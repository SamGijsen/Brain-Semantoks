"""
Full finetuning module for downstream tasks.
Borrows multi-task infrastructure from linear_probe.py but allows
training of both encoder and classification/regression heads.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from torch.cuda.amp import GradScaler
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import os

from data.dataset import FMRIDataset, simple_custom_collate
from linear_probe import calculate_metrics
from util.param_groups import get_params_groups_with_decay, fuse_params_groups

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _reset_encoder_weights(model_wrapper, config, device):
    """Reset encoder weights to original checkpoint state.
    
    This ensures independent evaluation across all experiments by resetting
    the encoder to its pretrained state before each repetition and label.
    
    Args:
        model_wrapper: ModelWrapper instance containing the encoder
        config: Configuration dict containing checkpoint path
        device: Device to load checkpoint on
    """
    checkpoint_path = config['training']['resume_checkpoint']
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logging.warning("Cannot reset encoder: checkpoint path not found")
        return
        
    logging.info(f"Resetting encoder weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if it exists (from DataParallel training)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Reset the underlying model weights
    model_wrapper._unwrapped_model.load_state_dict(state_dict)
    logging.info("Successfully reset encoder weights to checkpoint state")


def create_data_splits(probe_ds_info: Dict, rep_seed: int, n_reps: int) -> Dict:
    """Extract subject splitting logic into reusable function."""
    split_keys = ['probe_train_subject_ids_path', 'probe_val_subject_ids_path', 'probe_test_subject_ids_path']
    provided_splits = [key for key in split_keys if key in probe_ds_info and probe_ds_info[key] is not None]
    
    current_probe_ds_info = probe_ds_info.copy()
    
    if len(provided_splits) >= 3:
        return current_probe_ds_info
        
    # Load stratification and split data
    stratify_vars = current_probe_ds_info.get('stratify', [])
    split_ratio = current_probe_ds_info.get('split_ratio', [0.6, 0.2, 0.2] if len(provided_splits) == 1 else [0.75, 0.25])
    provided_train_subjects = list(np.load(current_probe_ds_info['probe_train_subject_ids_path'], allow_pickle=True))
    
    subject_stratify = None
    if stratify_vars:
        with h5py.File(current_probe_ds_info['data_path'], 'r') as f:
            all_subject_ids = [s.decode('utf-8') for s in f["long_subject_id"][:]]
            stratify_arrays = []
            for var in stratify_vars:
                if var in f:
                    stratify_arrays.append(f[var][:])
                else:
                    logging.warning(f"Stratification variable '{var}' not found")
            
            if stratify_arrays:
                stratify_data = np.column_stack(stratify_arrays)
                stratify_data = np.array(['_'.join(map(str, row)) for row in stratify_data])
                subject_to_stratify = {subj: stratify_data[i] for i, subj in enumerate(all_subject_ids) if subj in provided_train_subjects}
                subject_stratify = [subject_to_stratify.get(subj) for subj in provided_train_subjects]

        # Filter invalid stratification
        if subject_stratify:
            valid_pairs = [(subj, strat) for subj, strat in zip(provided_train_subjects, subject_stratify) if strat and strat != 'nan']
            if valid_pairs:
                provided_train_subjects, subject_stratify = map(list, zip(*valid_pairs))
            else:
                provided_train_subjects, subject_stratify = [], None
    
    # Perform splits
    if len(provided_splits) == 1:
        train_ratio, val_ratio, test_ratio = split_ratio
        temp_subjects, test_subjects = train_test_split(provided_train_subjects, test_size=test_ratio, random_state=rep_seed, stratify=subject_stratify)
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        temp_stratify = [subject_to_stratify[subj] for subj in temp_subjects] if subject_stratify else None
        train_subjects, val_subjects = train_test_split(temp_subjects, test_size=val_size_adjusted, random_state=rep_seed, stratify=temp_stratify)
        current_probe_ds_info.update({
            '_generated_train_subjects': train_subjects,
            '_generated_val_subjects': val_subjects,
            '_generated_test_subjects': test_subjects
        })
    else:  # len(provided_splits) == 2
        train_ratio, val_ratio = split_ratio
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_subjects, val_subjects = train_test_split(provided_train_subjects, test_size=val_size_adjusted, random_state=rep_seed, stratify=subject_stratify)
        current_probe_ds_info.update({
            '_generated_train_subjects': train_subjects,
            '_generated_val_subjects': val_subjects
        })
    
    return current_probe_ds_info


def create_split_datasets(probe_ds_info: Dict, main_data_config: Dict, finetune_config: Dict, 
                         dataset_label_names: List[str], validation_needed: bool = True, test_time_crops: int = 1) -> Dict:
    """Create datasets for train/val/test splits."""
    splits_to_process = ['probe_train', 'probe_test']
    if validation_needed:
        splits_to_process.insert(1, 'probe_val')
    
    split_datasets = {}
    for split in splits_to_process:
        subject_id_path_key = f"{split}_subject_ids_path"
        generated_subjects_key = f"_generated_{split.replace('probe_', '')}_subjects"
        
        target_subject_ids = None
        if generated_subjects_key in probe_ds_info:
            target_subject_ids = list(probe_ds_info[generated_subjects_key])
        elif subject_id_path_key in probe_ds_info and probe_ds_info[subject_id_path_key] is not None:
            target_subject_ids = list(np.load(probe_ds_info[subject_id_path_key], allow_pickle=True))
        
        if not target_subject_ids:
            split_datasets[split] = None
            continue
            
        # Use test_time_crops for test split, 1 for others
        num_crops = test_time_crops if split == 'probe_test' else 1
            
        # Create modality config
        split_modality_config = {
            **{k: main_data_config[k] for k in ['name', 'target_signal_length', 'patch_size', 'network_map_path'] if k in main_data_config},
            # Add explicit atlas specification
            'schaefer_atlas': main_data_config.get('schaefer_atlas'),
            'schaefer_rois': main_data_config.get('schaefer_rois'),
            'schaefer_networks': main_data_config.get('schaefer_networks'),
            'tian_atlas': main_data_config.get('tian_atlas'),
            'tian_rois': main_data_config.get('tian_rois'),
            'tian_networks': main_data_config.get('tian_networks'),
            'buckner_atlas': main_data_config.get('buckner_atlas'),
            'buckner_rois': main_data_config.get('buckner_rois'),
            'buckner_networks': main_data_config.get('buckner_networks'),
            'max_spatial': main_data_config.get('max_spatial', False),
            'min_spatial': main_data_config.get('min_spatial', False),
            'number_of_crops': num_crops,
            'channels': main_data_config.get('channels', 'all'),
            'min_crop_distance': 0,
            'max_crop_distance': 0,
            'datasets': [{
                'name': probe_ds_info['name'],
                'data_path': probe_ds_info['data_path'],
                'raw_signal_length': probe_ds_info['raw_signal_length'],
                '_target_subject_ids': target_subject_ids
            }]
        }
        
        split_datasets[split] = FMRIDataset(
            modality_config=split_modality_config,
            mode='probe_train',
            probe_label_names=dataset_label_names,
            use_augmentation=split == 'probe_train',
            augment_level=[0, 1, 1, 1] if split == 'probe_train' else [0, 0, 0, 0],
            crop_starts='random_mismatch'
        )
    
    # Set validation to None if not needed
    if not validation_needed:
        split_datasets['probe_val'] = None
        
    return split_datasets


def create_optimizer_scheduler(model: nn.Module, lr: float, finetune_config: Dict, max_epochs: int, use_grid_search: bool):
    """Create optimizer and scheduler based on config."""
    encoder_lr_multiplier = finetune_config.get('encoder_lr_multiplier', 1.0)
    weight_decay = finetune_config.get('weight_decay', 0.01)
    optimizer_type = finetune_config.get('optimizer', 'adamw')
    momentum = finetune_config.get('momentum', 0.9)
    lr_decay_rate = finetune_config.get('lr_decay_rate', 1.0)
    
    if lr_decay_rate < 1.0:
        # Use layerwise learning rate decay for encoder
        encoder_param_groups = get_params_groups_with_decay(model.encoder, lr_decay_rate=lr_decay_rate)
        fused_encoder_groups = fuse_params_groups(encoder_param_groups)
        
        # Create parameter groups with layerwise decay
        param_groups = []
        for i, group in enumerate(fused_encoder_groups):
            final_lr = lr * encoder_lr_multiplier * group['lr_multiplier']
            param_groups.append({
                'params': group['params'],
                'lr': final_lr,
                'weight_decay': weight_decay * group['wd_multiplier']
            })
        
        # Add head parameters (no decay)
        param_groups.append({
            'params': model.head.parameters(), 
            'lr': lr,
            'weight_decay': weight_decay
        })
    else:
        # Standard parameter groups without layerwise decay
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': lr * encoder_lr_multiplier},
            {'params': model.head.parameters(), 'lr': lr}
        ]
    
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay if lr_decay_rate >= 1.0 else 0.0)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay if lr_decay_rate >= 1.0 else 0.0)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
    if use_grid_search:
        min_lr = float(finetune_config.get('min_lr', 0))
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
    else:
        warmup_epochs = finetune_config.get('warmup_epochs', 5)
        scheduler_type = finetune_config.get('scheduler_type', 'cosine')
        min_lr = float(finetune_config.get('min_lr', 1e-6))
        
        if scheduler_type == 'cosine':
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
                main_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=min_lr)
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
        elif scheduler_type == 'linear':
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
                decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr/lr, total_iters=max_epochs - warmup_epochs)
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_epochs])
            else:
                scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr/lr, total_iters=max_epochs)
        elif scheduler_type == 'constant':
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=max_epochs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return optimizer, scheduler


def process_batch_data(batch: Dict, label_name: str, device: torch.device, is_classification: bool):
    """Extract and process batch data consistently."""
    batch_data = batch["signal"][0][0].to(device)  # [B, C, T]
    batch_res = batch["resolutions"].to(device) if "resolutions" in batch else None
    batch_labels = batch["labels"][label_name]
    
    if is_classification:
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.long().to(device)
        else:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    else:
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.float().to(device).unsqueeze(-1)
        else:
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32, device=device).unsqueeze(-1)
            
    return batch_data, batch_res, batch_labels


class FineTuningHead(nn.Module):    
    def __init__(self, feature_dim: int, num_classes: int,
                 head_type: str = 'linear', hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1):
        """
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of output classes (1 for regression)
            head_type: 'linear' or 'mlp'
            hidden_dims: Hidden dimensions for MLP head
            dropout: Dropout rate
        """
        super().__init__()
        
        # If MLP but no hidden dims, revert to linear
        if head_type == 'mlp' and (hidden_dims is None or len(hidden_dims) == 0):
            head_type = 'linear'
        
        if head_type == 'linear':
            
            self.head = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Linear(feature_dim, num_classes)
            )
            nn.init.trunc_normal_(self.head[1].weight, std=2e-5)
            
        elif head_type == 'mlp':
            layers = []
            current_dim = feature_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, num_classes))
            self.head = nn.Sequential(*layers)
            
            for i, layer in enumerate(self.head):
                if isinstance(layer, nn.Linear):
                    if i == len(self.head) - 1:  # Final layer
                        nn.init.trunc_normal_(layer.weight, std=2e-5)
                    else:  
                        nn.init.trunc_normal_(layer.weight, std=0.02)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(self, x):
        return self.head(x)


class FineTuningModel(nn.Module):
    """Combined encoder + head model for finetuning."""
    
    def __init__(self, encoder: nn.Module, head: nn.Module, 
                 feature_extractor_fn=None, freeze_encoder: bool = False, feature_type: str = 'cls_avg'):
        """
        Args:
            encoder: The encoder module
            head: The classification/regression head
            feature_extractor_fn: Function to extract features from encoder output
            freeze_encoder: Whether to freeze encoder initially
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.feature_extractor_fn = feature_extractor_fn
        self.feature_type = feature_type
        
        if freeze_encoder:
            self.freeze_encoder()
            
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
            
    def forward(self, x, atlas_idx=0, res=None):
        """Forward pass through encoder and head."""

        encoder_output = self.encoder(x, atlas=None, mask=None)
            
        # Extract features based on output type and feature_type
        if isinstance(encoder_output, dict):
            if 'global_cls' in encoder_output and 'tokens' in encoder_output:
                if self.feature_type == 'cls_avg':
                    # Combine cls and average tokens
                    avg = encoder_output['tokens'][:, 1:].mean(dim=1)
                    features = torch.cat([avg, encoder_output['global_cls']], dim=1)
                elif self.feature_type == 'cls':
                    # Use only CLS token
                    features = encoder_output['global_cls']
                elif self.feature_type == 'avg':
                    # Use only average tokens
                    features = encoder_output['tokens'][:, 1:].mean(dim=1)
                else:
                    raise ValueError(f"Unknown feature_type: {self.feature_type}")
            elif 'tokens' in encoder_output:
                # Use CLS token
                features = encoder_output['tokens'][:, 0]
            elif 'features' in encoder_output:
                features = encoder_output['features']
            else:
                raise ValueError(f"Cannot extract features from encoder output: {encoder_output.keys()}")
        else:
            features = encoder_output
            
        # Pass through head
        return self.head(features)


def _train_eval_finetune_model(
    model: FineTuningModel, optimizer: optim.Optimizer, scheduler: Optional[Any], criterion: nn.Module,
    train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device,
    label_name: str, is_classification: bool, early_stopping_patience: Optional[int] = None,
    freeze_encoder_epochs: int = 0, gradient_clip: float = 1.0, use_amp: bool = True
) -> Tuple[Dict[str, float], FineTuningModel]:
    """Train and evaluate a finetuning model."""
    best_val_metric = -float('inf') if is_classification else float('inf')
    best_model_state = None
    patience_counter = 0
    
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        if epoch == freeze_encoder_epochs and freeze_encoder_epochs > 0:
            logging.info(f"Unfreezing encoder at epoch {epoch}")
            model.unfreeze_encoder()
            
        # Training phase
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            batch_data, batch_res, batch_labels = process_batch_data(batch, label_name, device, is_classification)
            optimizer.zero_grad()
            
            def forward_pass():
                outputs = model(batch_data, atlas_idx=0, res=batch_res)
                return criterion(outputs, batch_labels)
                
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    loss = forward_pass()
                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_pass()
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            
            train_loss_sum += loss.item()
            
        train_losses.append(train_loss_sum / len(train_loader))
        
        # Validation phase
        val_metrics = {'dummy_metric': 0.0}
        primary_metric = 0.0
        
        if val_loader is not None:
            model.eval()
            val_outputs, val_labels_list, val_loss_sum = [], [], 0.0
            
            with torch.no_grad():
                # Use autocast for compatibility with FlashAttention
                with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', enabled=(device.type == 'cuda')):
                    for batch in val_loader:
                        batch_data, batch_res, batch_labels = process_batch_data(batch, label_name, device, is_classification)
                        outputs = model(batch_data, atlas_idx=0, res=batch_res)
                        val_loss_sum += criterion(outputs, batch_labels).item()
                        val_outputs.append(outputs.cpu())
                        val_labels_list.append(batch_labels.cpu())
            
            val_outputs = torch.cat(val_outputs, dim=0)
            val_labels_tensor = torch.cat(val_labels_list, dim=0)
            val_losses.append(val_loss_sum / len(val_loader))
            val_metrics = calculate_metrics(val_outputs, val_labels_tensor, is_classification)
            primary_metric = val_metrics.get('balanced_accuracy' if is_classification else 'mae', -1.0)
        else:
            val_losses.append(None)
        
        if scheduler is not None:
            scheduler.step()
            
        # Early stopping
        is_better = (primary_metric > best_val_metric) if is_classification else (primary_metric < best_val_metric)
        if is_better:
            best_val_metric = primary_metric
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    val_metrics.update({'train_losses': train_losses, 'val_losses': val_losses})
    return val_metrics, model


def run_single_label_experiment(model_wrapper, probe_ds_info: Dict, main_data_config: Dict, 
                               finetune_config: Dict, device: torch.device, label_name: str, 
                               rep_seed: int, use_grid_search: bool) -> Dict:
    """Run finetuning experiment for a single label."""
    # Setup parameters
    n_class = dict(zip(probe_ds_info['label_names'], probe_ds_info['n_class']))[label_name]
    is_classification = n_class > 1
    num_classes = n_class if is_classification else 1
    criterion = nn.CrossEntropyLoss() if is_classification else nn.L1Loss()
    
    # Handle data splits
    current_probe_ds_info = create_data_splits(probe_ds_info, rep_seed, 1)
    validation_needed = use_grid_search or finetune_config.get('early_stopping_patience') is not None
    test_time_crops = finetune_config.get('test_time_crops', 1)
    split_datasets = create_split_datasets(current_probe_ds_info, main_data_config, finetune_config, 
                                          probe_ds_info['label_names'], validation_needed, test_time_crops)
    
    if split_datasets['probe_train'] is None:
        logging.warning(f"No training data for {label_name}. Skipping.")
        return {}
    
    # Grid search parameters
    learning_rates = finetune_config.get('learning_rates', [0.001]) if use_grid_search else [finetune_config.get('fixed_lr', 0.0001)]
    max_epochs = finetune_config.get('max_epochs_grid', 30) if use_grid_search else finetune_config.get('fixed_epochs', 50)
    
    best_val_metric = -float('inf') if is_classification else float('inf')
    best_config_params = {'lr': None}
    best_model, best_val_metrics = None, {}
    
    # Create data loaders once (outside learning rate loop to avoid excessive worker warnings)
    batch_size = finetune_config.get('batch_size', 256)
    # batch_size = int(max(1, min(batch_size, len(split_datasets['probe_train'])/8)))
    num_workers = finetune_config.get('num_workers', 10)
    
    train_loader = DataLoader(split_datasets['probe_train'], batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, collate_fn=simple_custom_collate)
    val_loader = DataLoader(split_datasets['probe_val'], batch_size=batch_size*2, shuffle=False,
                          num_workers=num_workers, collate_fn=simple_custom_collate) if split_datasets['probe_val'] else None
    test_loader = DataLoader(split_datasets['probe_test'], batch_size=batch_size*2, shuffle=False,
                           num_workers=num_workers, collate_fn=simple_custom_collate) if split_datasets['probe_test'] else None
    
    # Grid search over learning rates
    for lr in learning_rates:
        # Scale learning rate by batch size
        scaled_lr = lr * (batch_size / 256)
        print(f"Original LR: {lr}, Batch size: {batch_size}, Scaled LR: {scaled_lr}")
        
        # Create model components
        feature_dim = model_wrapper.get_feature_dim(finetune_config.get('feature_type', 'cls_avg'))
        encoder = model_wrapper.prepare_for_finetuning(
            finetune_config.get('freeze_encoder_epochs', 0), 
            finetune_config.get('use_teacher', False)
        )
        head = FineTuningHead(
            feature_dim, num_classes,
            head_type=finetune_config.get('head_type', 'linear'),
            hidden_dims=finetune_config.get('hidden_dims'),
            dropout=finetune_config.get('dropout', 0.1)
        ).to(device)
        
        ft_model = FineTuningModel(
            encoder, head,
            freeze_encoder=finetune_config.get('freeze_encoder_epochs', 0) > 0,
            feature_type=finetune_config.get('feature_type', 'cls_avg')
        ).to(device)
        
        # Create optimizer and scheduler with scaled learning rate
        optimizer, scheduler = create_optimizer_scheduler(ft_model, scaled_lr, finetune_config, max_epochs, use_grid_search)
        
        # Train and evaluate
        val_metrics, trained_model = _train_eval_finetune_model(
            ft_model, optimizer, scheduler, criterion, train_loader, val_loader, max_epochs, device,
            label_name, is_classification, finetune_config.get('early_stopping_patience'),
            finetune_config.get('freeze_encoder_epochs', 0), finetune_config.get('gradient_clip', 1.0),
            finetune_config.get('use_amp', True)
        )
        
        primary_metric = val_metrics.get('balanced_accuracy' if is_classification else 'mae', -1.0)
        is_better = (primary_metric > best_val_metric) if is_classification else (primary_metric < best_val_metric)
        
        if is_better:
            best_val_metric = primary_metric
            best_config_params['lr'] = scaled_lr
            best_model = trained_model
            best_val_metrics = val_metrics
    
    # Test evaluation
    test_metrics, test_primary_metric = {}, -1.0
    if best_model and test_loader:
        test_time_crops = finetune_config.get('test_time_crops', 1)
        
        # Evaluate on test set
        best_model.eval()
        test_outputs, test_labels_list, test_subject_ids = [], [], []
        
        with torch.no_grad():
            # Use autocast for compatibility with FlashAttention
            with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', enabled=(device.type == 'cuda')):
                for batch in test_loader:
                    batch_labels = batch["labels"][label_name]
                    batch_subjects = batch.get("id", [])
                    
                    # Handle multiple crops
                    num_crops = len(batch["signal"][0])
                    for crop_idx in range(num_crops):
                        batch_data, batch_res, _ = process_batch_data(
                            {**batch, "signal": [[batch["signal"][0][crop_idx]]]}, 
                            label_name, device, is_classification
                        )
                        outputs = best_model(batch_data, atlas_idx=0, res=batch_res)
                        test_outputs.append(outputs.cpu())
                        test_labels_list.extend(batch_labels)
                        test_subject_ids.extend(batch_subjects)
        
        test_outputs = torch.cat(test_outputs, dim=0)
        test_labels_array = np.array(test_labels_list)
        
        # Handle TTA by averaging predictions per subject
        if test_time_crops > 1 and test_subject_ids:
            df = pd.DataFrame({'subject_id': test_subject_ids, 'index': range(len(test_subject_ids))})
            df = df.sort_values('subject_id', kind='mergesort')
            sorted_indices = df['index'].values
            
            test_outputs = test_outputs[sorted_indices]
            test_labels_array = test_labels_array[sorted_indices]
            test_subject_ids = [test_subject_ids[i] for i in sorted_indices]
            
            # Average by subject
            if is_classification:
                df_out = pd.DataFrame(test_outputs.numpy())
                df_out['subject_id'] = test_subject_ids
                averaged_outputs = torch.tensor(df_out.groupby('subject_id').mean().values)
            else:
                df_out = pd.DataFrame({'output': test_outputs.squeeze().numpy(), 'subject_id': test_subject_ids})
                averaged_outputs = torch.tensor(df_out.groupby('subject_id')['output'].mean().values).unsqueeze(1)
            
            df_labels = pd.DataFrame({'label': test_labels_array, 'subject_id': test_subject_ids})
            unique_labels = torch.tensor(df_labels.groupby('subject_id')['label'].first().values, 
                                       dtype=torch.int64 if is_classification else torch.float32)
            
            test_metrics = calculate_metrics(averaged_outputs, unique_labels, is_classification)
            test_metrics = {f'ensembled_{k}': v for k, v in test_metrics.items()}
        else:
            test_labels_tensor = torch.from_numpy(test_labels_array.astype(np.int64 if is_classification else np.float32))
            test_metrics = calculate_metrics(test_outputs, test_labels_tensor, is_classification)
        
        test_primary_metric = test_metrics.get(f"{'ensembled_' if test_time_crops > 1 else ''}{'balanced_accuracy' if is_classification else 'mae'}", -1.0)
    
    return {
        'best_val_primary_metric': float(best_val_metric) if abs(best_val_metric) != float('inf') else None,
        'test_primary_metric': float(test_primary_metric) if test_primary_metric != -1.0 else None,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_lr': float(best_config_params['lr']) if best_config_params['lr'] else None,
        'head_type': finetune_config.get('head_type', 'linear'),
        'train_losses': best_val_metrics.get('train_losses', []),
        'val_losses': best_val_metrics.get('val_losses', [])
    }


def run_finetuning(model_wrapper, finetune_config: Dict, main_data_config: Dict, 
                   device: torch.device, config: Optional[Dict] = None) -> Dict:
    """Run full finetuning evaluation pipeline."""
    start_time = time.time()
    logging.info("--- Starting Finetuning Evaluation ---")
    
    training_seed = config.get('training', {}).get('seed', 0) if config else 0
    n_repetitions = finetune_config.get('n_repetitions', 1)
    use_grid_search = finetune_config.get('use_grid_search', True)
    
    logging.info(f"Training mode: {'Grid search' if use_grid_search else 'Fixed schedule'}")
    
    all_results = {}
    
    for probe_ds_info in finetune_config.get('probe_datasets', []):
        ds_name = probe_ds_info['name']
        logging.info(f"--- Processing Dataset: {ds_name} ---")
        
        # Validate required dataset fields
        required_keys = ['data_path', 'raw_signal_length', 'label_names', 'n_class']
        if not all(key in probe_ds_info for key in required_keys):
            logging.warning(f"Skipping dataset '{ds_name}': missing required keys")
            continue
            
        dataset_label_names = probe_ds_info['label_names']
        
        # Determine number of repetitions needed
        split_keys = ['probe_train_subject_ids_path', 'probe_val_subject_ids_path', 'probe_test_subject_ids_path']
        provided_splits = [key for key in split_keys if key in probe_ds_info and probe_ds_info[key] is not None]
        n_reps_to_run = 1 if len(provided_splits) == 3 else n_repetitions
        
        dataset_repetition_results = []
        
        # Run repetitions
        for rep in range(n_reps_to_run):
            if n_reps_to_run > 1:
                logging.info(f"--- Running repetition {rep + 1}/{n_reps_to_run} for {ds_name} ---")
            
            rep_seed = training_seed + rep
            repetition_label_results = {}
            
            # Process each label
            for label_idx, label_name in enumerate(dataset_label_names):
                logging.info(f"--- Finetuning for Label: {label_name} (rep {rep + 1}) ---")
                
                # Reset encoder weights before each label experiment
                # This ensures independent evaluation across ALL experiments
                if rep > 0 or label_idx > 0:  # Reset after first experiment
                    _reset_encoder_weights(model_wrapper, config, device)
                
                result = run_single_label_experiment(
                    model_wrapper, probe_ds_info, main_data_config, finetune_config, 
                    device, label_name, rep_seed, use_grid_search
                )
                if result:
                    repetition_label_results[label_name] = result
                    logging.info(f"  Best LR: {result['best_lr']} (Val: {result['best_val_primary_metric']:.2f})")
                    # Log key classification metrics for better visibility
                    test_metrics = result['test_metrics']
                    n_class = dict(zip(probe_ds_info['label_names'], probe_ds_info['n_class']))[label_name]
                    if n_class > 1:  # Classification
                        key_metrics = {k: v for k, v in test_metrics.items() if any(m in k for m in ['accuracy', 'f1', 'precision', 'recall', 'auroc'])}
                        logging.info(f"  Test metrics: {key_metrics}")
                    else:  # Regression
                        logging.info(f"  Test metrics: {test_metrics}")
            
            dataset_repetition_results.append(repetition_label_results)
        
        # Aggregate results across repetitions
        all_results[ds_name] = {}
        for label_name in dataset_label_names:
            if n_reps_to_run == 1:
                if dataset_repetition_results and label_name in dataset_repetition_results[0]:
                    all_results[ds_name][label_name] = dataset_repetition_results[0][label_name]
            else:
                # Multiple repetitions - compute summary statistics
                repetition_data = [rep_results[label_name] for rep_results in dataset_repetition_results if label_name in rep_results]
                if repetition_data:
                    test_metrics = [r['test_primary_metric'] for r in repetition_data if r['test_primary_metric']]
                    val_metrics = [r['best_val_primary_metric'] for r in repetition_data if r['best_val_primary_metric']]
                    
                    summary = {}
                    if test_metrics:
                        summary.update({
                            'mean_test_primary_metric': float(np.mean(test_metrics)),
                            'std_test_primary_metric': float(np.std(test_metrics))
                        })
                    if val_metrics:
                        summary.update({
                            'mean_val_primary_metric': float(np.mean(val_metrics)),
                            'std_val_primary_metric': float(np.std(val_metrics))
                        })
                    
                    all_results[ds_name][label_name] = {'repetitions': repetition_data, 'summary': summary}
                    
    logging.info(f"--- Finetuning Finished ({time.time() - start_time:.2f}s) ---")
    return all_results