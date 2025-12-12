import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import math
import os
import time
import json
import yaml
from collections import defaultdict


from simdino import SimDINOModel
from data.dataset import FMRIDataset, simple_custom_collate
from util.util import get_lr_schedule_with_warmup
from linear_probe import run_linear_probe
from util.param_groups import get_params_groups_with_decay, fuse_params_groups
from model.model_wrapper import ModelWrapper
from finetune import run_finetuning

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def is_main_process(rank):
    return rank == 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_network_loss_weight(step, final_weight, netloss_delay_steps=0,
                           netloss_weight_decay_steps=0, netloss_decay_schedule='linear'):
    if step < netloss_delay_steps:
        return 0.0

    if netloss_weight_decay_steps > 0:
        steps_since_delay = step - netloss_delay_steps
        if steps_since_delay >= netloss_weight_decay_steps:
            return 0.0
        decay_progress = steps_since_delay / netloss_weight_decay_steps
        if netloss_decay_schedule == 'cosine':
            return final_weight * 0.5 * (1 + math.cos(math.pi * decay_progress))
        return final_weight * (1.0 - decay_progress)
    return final_weight

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from: {config_path}")
    return config

def main(config):
    is_distributed, rank, world_size, local_rank = setup_distributed()
    device_id = local_rank if is_distributed else None

    def run_linear_probe_if_needed(epoch_num, model, probe_config, config, output_dir, device):
        if probe_config.get('enabled', False) and epoch_num in probe_config['probe_at_epochs']:
            if is_main_process(rank):
                print(f"Running linear probe at epoch {epoch_num}...")
            probe_model_instance = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
            probe_model_instance.eval()
            
            main_data_config_for_probe = config['data']
            probe_results = run_linear_probe(
                ssl_model=probe_model_instance,
                probe_config=probe_config,
                main_data_config=main_data_config_for_probe,
                device=device)
            if is_main_process(rank):
                probe_results_file = os.path.join(output_dir, f'linear_probe_results_epoch_{epoch_num}.yaml')
                with open(probe_results_file, 'w') as f:
                    yaml.dump(probe_results, f, default_flow_style=False)
                print(f"Linear probing results saved to: {probe_results_file}")
            
    def run_finetuning_if_needed(epoch_num, model, finetune_config, config, output_dir, device):
        if finetune_config.get('enabled', False) and epoch_num in finetune_config.get('probe_at_epochs', []):
            if is_main_process(rank):
                print(f"Running finetuning evaluation at epoch {epoch_num}...")
            probe_model_instance = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
            probe_model_instance.eval()
            
            # Create model wrapper for finetuning
            model_type = config.get('model_type', 'simdino')
            model_wrapper = ModelWrapper(probe_model_instance, model_type, config)
            
            main_data_config_for_probe = config['data']
            finetune_results = run_finetuning(
                model_wrapper=model_wrapper,
                finetune_config=finetune_config,
                main_data_config=main_data_config_for_probe,
                device=device,
                config=config)
            if is_main_process(rank):
                finetune_results_file = os.path.join(output_dir, f'finetune_results_epoch_{epoch_num}.yaml')
                with open(finetune_results_file, 'w') as f:
                    yaml.dump(finetune_results, f, default_flow_style=False)
                print(f"Finetuning results saved to: {finetune_results_file}")

    if config['training']['use_cuda'] and torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            if is_main_process(rank):
                print(f"Using distributed training with {world_size} GPUs.")
        else:
            device = torch.device("cuda")
            if is_main_process(rank):
                print(f"Using CUDA with {torch.cuda.device_count()} GPUs.")
        scaler = GradScaler()
    else:
        device = torch.device("cpu")
        if is_main_process(rank):
            print("Using CPU.")
        scaler = None

    seed = config['training']['seed']
    torch.manual_seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)

    model_cfg = config['model']
    dino_cfg = config['dino']
    ssl_cfg = config['ssl']
    train_cfg = config['training']
    model_type = config.get('model_type', 'hierarchical')
    data_cfg = config['data']
    mask_cfg = config['masking']
    gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    data_cfg_with_masking = data_cfg.copy()
    data_cfg_with_masking['canonical_network_masks'] = mask_cfg.get('canonical_network_masks', False)

    train_dataset = FMRIDataset(
        modality_config=data_cfg_with_masking,
        mode='train', 
        augment_level=data_cfg.get('augment_level', [0,0,0,0]),  
        crop_starts=data_cfg['crop_starts'],
        masking_ratio=mask_cfg['masking_ratio'],
        masking_type=mask_cfg['masking_type'],
    )

    # Setup distributed sampler if using DDP with multiple GPUs
    if is_distributed and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False  
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=simple_custom_collate
    )
    if is_main_process(rank):
        print(f"Dataset size: {len(train_dataset)}, Loader size: {len(train_loader)}")

    mlp_dim = int(model_cfg['embedding_dim'] * 4)
    heads = int(model_cfg['embedding_dim'] / 64)
    if is_main_process(rank):
        print("Overwriting mlp_dim to D*4 and heads to D/64.")

    # Build atlas names from explicit config and calculate total ROIs
    atlas_names = []
    atlas_network_counts = []
    total_rois = 0
    for atlas_type in ['schaefer', 'tian', 'buckner']:
        atlas_name = data_cfg.get(f'{atlas_type}_atlas')
        if atlas_name is not None:  # Atlas is used
            atlas_names.append(atlas_name)
            atlas_network_counts.append(data_cfg[f'{atlas_type}_networks'])
            total_rois += data_cfg[f'{atlas_type}_rois']
    
    # Get spatial mode flags
    max_spatial = data_cfg.get('max_spatial', False)
    min_spatial = data_cfg.get('min_spatial', False)

    if model_type in ['simple_dino_tf', 'semantoks_dino_tf']:
        model = SimDINOModel(
            # model params
            patch_size=data_cfg['patch_size'],
            target_time_length=data_cfg['target_signal_length'],
            embedding_dim=model_cfg['embedding_dim'],
            depth=model_cfg['depth'],
            mlp_dim=mlp_dim,
            global_pooling=model_cfg['global_pooling'],
            # dropout
            drop_path_rate=model_cfg['drop_path_rate'],
            layer_scale_init_value=model_cfg.get('layer_scale_init_value', None),
            emb_dropout=model_cfg['emb_dropout'],
            # network info
            network_data_path=data_cfg['network_map_path'],
            atlas_names=atlas_names,
            # predictor
            heads=heads, #model_cfg['heads'],
            projection_hidden_dim=model_cfg['projection_hidden_dim'],
            projection_output_dim=model_cfg['projection_output_dim'],
            projection_bottleneck_dim=model_cfg['projection_bottleneck_dim'],
            projection_nlayers=model_cfg['projection_nlayers'],
            # DINO params
            base_teacher_momentum=dino_cfg['base_teacher_momentum'],
            coeff=dino_cfg['coeff'],
            # masking params
            do_masking=mask_cfg['masking_frequency'] > 0,
            # ssl
            mask_loss_weight=ssl_cfg['mask_loss_weight'],
            use_separate_mask_predictor=dino_cfg['use_separate_mask_predictor'],
            network_loss_weight=ssl_cfg.get('network_loss_weight', 0.0),
            backbone_type=model_cfg.get('backbone_type', 'cnn_tf'),
            # Spatial resolution modes
            max_spatial=max_spatial,
            min_spatial=min_spatial,
            total_rois=total_rois,
            atlas_network_counts=atlas_network_counts,
            # Tokenizer configuration
            tokenizer_config=model_cfg.get('tokenizer', {}).get('config'),
            tokenizer_final_norm=model_cfg.get('tokenizer', {}).get('final_norm', 'layer')
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    opt_cfg = config['optimizer']
    train_cfg = config['training']
    save_model = train_cfg.get('save_model', False)

    if is_distributed:
        effective_batch_size = train_cfg['batch_size'] * world_size * gradient_accumulation_steps
    elif isinstance(model, nn.DataParallel):
        effective_batch_size = train_cfg['batch_size'] * torch.cuda.device_count() * gradient_accumulation_steps
    else:
        effective_batch_size = train_cfg['batch_size'] * gradient_accumulation_steps

    base_lr = opt_cfg['base_lr_scale'] * effective_batch_size / 256
    if is_main_process(rank):
        print(f"Effective Batch Size: {effective_batch_size} (batch_size={train_cfg['batch_size']} x grad_accum={gradient_accumulation_steps}), Base LR: {base_lr:.6f}")

    wd_start = opt_cfg['weight_decay']
    wd_end = opt_cfg.get('weight_decay_end', wd_start) 
    lr_decay_rate = opt_cfg.get('lr_decay_rate', 1.0)

    if opt_cfg['type'].lower() == 'adamw':
        if lr_decay_rate < 1.0:
            # Use layerwise learning rate decay
            all_param_groups = get_params_groups_with_decay(model, lr_decay_rate=lr_decay_rate)
            fused_param_groups = fuse_params_groups(all_param_groups)
            
            # Create optimizer with parameter groups
            param_groups_for_optimizer = []
            for group in fused_param_groups:
                param_groups_for_optimizer.append({
                    'params': group['params'],
                    'lr': base_lr * group['lr_multiplier'],
                    'weight_decay': wd_start * group['wd_multiplier']
                })
            
            optimizer = optim.AdamW(param_groups_for_optimizer)
        else:
            # Standard optimizer without layerwise decay
            optimizer = optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=wd_start
            )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_cfg['type']}")

    sched_cfg = config['scheduler']
    total_epochs = train_cfg['epochs']
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * total_epochs
    num_warmup_steps = steps_per_epoch * sched_cfg['warmup_epochs']
    if is_main_process(rank):
        print(f"Total training steps: {num_training_steps}")
        print(f"Warmup steps: {num_warmup_steps}")

    scheduler = get_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    train_cfg = config['training']
    base_output_dir = train_cfg['output_dir']
    probe_config = config.get('linear_probe', {})
    finetune_config = config.get('finetune', {})
    run_name = config.get('run_name', 'unnamed_run') # Get run_name, default if missing

    # Create the specific output directory for this run
    output_dir = os.path.join(base_output_dir, run_name)
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Outputs will be saved to: {output_dir}")

        # Save config used for this run inside the run's directory
        config_save_path = os.path.join(output_dir, 'config_used.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved current configuration to: {config_save_path}")
    
    # Synchronize processes to ensure directory is created before proceeding
    if is_distributed:
        # Initialize CUDA context on the correct device to avoid barrier warning
        _ = torch.cuda.FloatTensor(1, device=device)
        dist.barrier()

    start_epoch = 0
    # Check for resume path within the specific run directory first (optional)
    potential_resume_path = train_cfg.get('resume_checkpoint', None)
    if potential_resume_path and os.path.isabs(potential_resume_path):
         resume_path = potential_resume_path # Use absolute path if provided
    elif potential_resume_path: # Assume relative path is within the run's output dir
         resume_path = os.path.join(output_dir, potential_resume_path)
    else:
         resume_path = None

    # resume_path = train_cfg.get('resume_checkpoint', None) # Original logic - might point outside run dir
    if resume_path and os.path.exists(resume_path):
        if is_main_process(rank):
            print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # Load into unwrapped model (before DDP/DataParallel wrapping)
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if it exists in checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Load scaler state if resuming on CUDA and scaler exists in checkpoint
        if device.type == 'cuda' and scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if is_main_process(rank):
                print("Loaded GradScaler state.")
        elif device.type == 'cuda' and scaler is not None:
            if is_main_process(rank):
                print("Warning: Resuming on CUDA but GradScaler state not found in checkpoint.")

        if is_main_process(rank):
            print(f"Resumed from epoch {start_epoch}")
            if 'config' in checkpoint:
                print("Loaded config from checkpoint.")
                # config = checkpoint['config'] # Or decide how to merge/prioritize
    elif train_cfg.get('resume_checkpoint', None): # Check original config value for warning
        if is_main_process(rank):
            print(f"Warning: Checkpoint file not found at specified path: {train_cfg.get('resume_checkpoint')}. Starting training from scratch.")

    if is_distributed and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process(rank):
            print(f"Using DistributedDataParallel with {world_size} GPUs.")
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if is_main_process(rank):
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
    else:
        if is_main_process(rank):
            print(f"Using single GPU without wrapper.")

    all_epoch_losses = [] # Initialize list to store epoch losses

    if mask_cfg['masking_frequency'] > 0:
        assert data_cfg["number_of_crops"] == 2, "Masking requires number_of_crops to be 2"

    # Linear probe and finetuning at epoch 0 (before training, eg with random weights or pretrained)
    run_linear_probe_if_needed(0, model, probe_config, config, output_dir, device)
    run_finetuning_if_needed(0, model, finetune_config, config, output_dir, device)

    for epoch in range(start_epoch, total_epochs):

        # Set sampler epoch for distributed training
        if is_distributed and world_size > 1:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses_sum = defaultdict(float)
        epoch_batches = 0
        epoch_losses = defaultdict(list)

        model_instance = model.module if isinstance(model, (nn.DataParallel, DDP)) else model

        for batch_idx, batch in enumerate(train_loader):
            current_step = epoch * steps_per_epoch + batch_idx

            if wd_start != wd_end: # Cosine schedule for weight decay
                current_wd = wd_end + 0.5 * (wd_start - wd_end) * (1 + math.cos(math.pi * current_step / num_training_steps))
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = current_wd

            do_masking = (mask_cfg['masking_frequency'] > torch.rand(1).item()) # and (mask_loss_weight > 0)

            atlas_idx = 0
            atlas_2_idx = 0
            view1, view2 = batch['signal'][atlas_idx][0].to(device), batch['signal'][atlas_2_idx][1].to(device)
            atlases = [atlas_idx, atlas_2_idx]
            mask = [m.to(device) for m in batch['mask']] if do_masking else [None, None]
            
            # Compute network loss weight with optional delay and decay
            current_network_loss_weight = get_network_loss_weight(
                step=current_step,
                final_weight=ssl_cfg.get('network_loss_weight', 0.0),
                netloss_delay_steps=ssl_cfg.get('netloss_delay_epochs', 0) * steps_per_epoch,
                netloss_weight_decay_steps=ssl_cfg.get('netloss_weight_decay_epochs', 0) * steps_per_epoch,
                netloss_decay_schedule=ssl_cfg.get('netloss_decay_schedule', 'linear')
            )
            model_instance.current_network_loss_weight = current_network_loss_weight

            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                model_instance = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
                total_loss, log_data = model_instance.forward(view1, view2, atlases, mask)

            scaled_loss = total_loss / gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Optimizer step only at the end of accumulation cycle or at the end of epoch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()

                # Update teacher only after optimizer step
                model_instance = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
                model_instance.update_teacher(current_step, num_training_steps)
            for loss_name, loss_val in log_data.items():
                 if isinstance(loss_val, (float, int)) or (torch.is_tensor(loss_val) and loss_val.numel() == 1):
                     epoch_losses_sum[loss_name] += float(loss_val) # Accumulate each named loss
                     epoch_losses[loss_name].append(float(loss_val)) # Store each batch loss
            epoch_batches += 1

            if (batch_idx + 1) % train_cfg['log_interval'] == 0 and is_main_process(rank):
                current_lr = optimizer.param_groups[0]['lr']
                log_str = (f"Epoch: {epoch+1}/{total_epochs} | "
                           f"Batch: {batch_idx+1}/{len(train_loader)} | "
                           f"LR: {current_lr:.6f} | ")
                # Log current batch individual losses from log_data
                for loss_name, loss_val in log_data.items():
                    if 'loss' in loss_name:
                         log_str += f"{loss_name}: {float(loss_val):.4f} | "
                print(log_str)
            
        avg_epoch_losses = {name: total / epoch_batches for name, total in epoch_losses_sum.items()}

        log_str_epoch = ""
        for name, avg_val in avg_epoch_losses.items():
            # print(f"Average {name}: {avg_val:.4f}")
            log_str_epoch += f"Avg {name}: {avg_val:.4f} | "
        # print(f"Epoch Time: {epoch_time:.2f}s")

        # Store epoch average losses (including individual ones)
        epoch_data_to_save = {'epoch': epoch + 1, 'batch_losses': epoch_losses}
        epoch_data_to_save.update(avg_epoch_losses) # Add all calculated averages
        all_epoch_losses.append(epoch_data_to_save)

        # Checkpointing
        if ((epoch + 1) % train_cfg['checkpoint_interval'] == 0 or (epoch + 1) == total_epochs) and is_main_process(rank):
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Saving checkpoint to {checkpoint_path}...")
            model_state = model.module.state_dict() if isinstance(model, (nn.DataParallel, DDP)) else model.state_dict()
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_losses.get('total_loss', None),
                'config': config,
                "lr": optimizer.param_groups[0]['lr'],
                "wd": optimizer.param_groups[0]['weight_decay'],
                "mask_loss_weight": model.module.mask_loss_weight if isinstance(model, (nn.DataParallel, DDP)) else model.mask_loss_weight,
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            if save_model:
                torch.save(save_dict, checkpoint_path)

            losses_save_path = os.path.join(output_dir, 'training_losses.json')
            try:
                with open(losses_save_path, 'w') as f:
                    json.dump(all_epoch_losses, f, indent=4)
                print(f"Saved epoch average and batch losses to: {losses_save_path}")
            except Exception as e:
                print(f"Error saving epoch losses: {e}")

            print("Checkpoint saved.")

        run_linear_probe_if_needed(epoch + 1, model, probe_config, config, output_dir, device)
        run_finetuning_if_needed(epoch + 1, model, finetune_config, config, output_dir, device)
    
    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical SSL Model Training from YAML Config")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file (e.g., config/uni_base.yaml)')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)