import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# from model.cnn_tf_comm import CNN_TF
from model.semantoks import CNN_TF
from model.dino_head import DINOHead

# Backbone selection imports
def get_backbone_class(backbone_type: str = 'cnn_tf'):
    """Get backbone class based on configuration"""
    if backbone_type == 'semantoks':
        try:
            from model.semantoks import Semantoks
            return Semantoks
        except ImportError as e:
            raise ImportError(f"Failed to import Semantoks: {e}")
    elif backbone_type == 'cnn_tf':
        return CNN_TF
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
from model.simdino_loss import MCRLoss, CosinePatchLoss
from typing import Optional

class SimDINOModel(nn.Module): 
    """
    A simplified DINO (Distillation with No Labels) model using 1D CNN encoders.
    
    Args:
        embedding_dim (int): Dimension of the encoder output embedding (D_out)
        kernel_sizes (list): List of kernel sizes for the CNN encoder
        base_channels (int): Base number of channels for the first conv layer
        projection_hidden_dim (int): Hidden dimension of the projection head
        projection_output_dim (int): Output dimension of the projection head (K)
        student_temp (float): Temperature for the student softmax
        teacher_temp (float): Temperature for the teacher softmax
        center_momentum (float): Momentum for updating the center
        base_teacher_momentum (float): Base momentum for teacher EMA updates
    """
    def __init__(
        self,
        # Arch
        patch_size,
        target_time_length,
        embedding_dim=200,
        depth=4,
        heads=5,
        mlp_dim=200*5,
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        emb_dropout=0.0,
        network_data_path=None,
        atlas_names=None,
        global_pooling='cls',
        # Backbone selection
        backbone_type='cnn_tf',  # 'cnn_tf' or 'semantoks'
        semantoks_config=None,   # Configuration dict for Semantoks
        # DINO
        projection_hidden_dim=512,
        projection_output_dim=2048,
        projection_bottleneck_dim=128,
        projection_nlayers=3,
        center_momentum=0.9,
        base_teacher_momentum=0.996,
        mask_loss_weight=0.0,
        network_loss_weight=0.0,
        do_masking=False,
        use_separate_mask_predictor=False,
        coeff=1,
        # Spatial resolution modes
        max_spatial=False, # If True, each ROI is its own network
        min_spatial=False, # If True, all ROIs constitute a single network
        total_rois=None, # Total ROI count when using spatial modes
        atlas_network_counts=None, # Network counts per atlas for standard mode
        # Tokenizer configuration
        tokenizer_config=None,  # Configuration for FlexibleNetworkTokenizer
        tokenizer_final_norm='layer',  # Final normalization for tokenizer
    ):
        super().__init__()
        
        self.center_momentum = center_momentum
        self.base_teacher_momentum = base_teacher_momentum
        self.projection_output_dim = projection_output_dim
        self.mask_loss_weight = mask_loss_weight
        self.network_loss_weight = network_loss_weight
        self.current_network_loss_weight = network_loss_weight  
        self.use_separate_mask_predictor = use_separate_mask_predictor
        self.coeff = coeff    
        
        # Get backbone class based on configuration
        BackboneClass = get_backbone_class(backbone_type)
        
        # Common parameters for both backbones
        common_params = {
            'target_time_length': target_time_length,
            'patch_size': patch_size,
            'network_data_path': network_data_path,
            'embedding_dim': embedding_dim,
            'depth': depth,
            'heads': heads,
            'mlp_dim': mlp_dim,
            'drop_path_rate': drop_path_rate,
            'layer_scale_init_value': layer_scale_init_value,
            'emb_dropout': emb_dropout,
            'global_pooling': global_pooling,
            'num_classes': 0,
            'atlas_names': atlas_names,
            'do_masking': do_masking,
            # Spatial resolution modes
            'max_spatial': max_spatial,
            'min_spatial': min_spatial, 
            'total_rois': total_rois,
            'atlas_network_counts': atlas_network_counts,
            # Tokenizer configuration
            'tokenizer_config': tokenizer_config,
            'tokenizer_final_norm': tokenizer_final_norm,
        }
        
            
        # Initialize student encoder
        self.student_encoder = BackboneClass(**common_params)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        
        # Store encoder dimensions for network loss
        self.num_networks = self.student_encoder.inter_network_transformer.num_networks
        self.num_patches = self.student_encoder.inter_network_transformer.num_patches
        self.embedding_dim = embedding_dim

        self.student_predictor = DINOHead(
            in_dim=embedding_dim,
            out_dim=projection_output_dim,
            hidden_dim=projection_hidden_dim,
            bottleneck_dim=projection_bottleneck_dim,
            nlayers=projection_nlayers,
            remove_last_layer=True,
            normalize=True
        )
        self.teacher_predictor = DINOHead( 
            in_dim=embedding_dim,
            out_dim=projection_output_dim,
            hidden_dim=projection_hidden_dim,
            bottleneck_dim=projection_bottleneck_dim,
            nlayers=projection_nlayers,
            remove_last_layer=True,
            normalize=True
        ) # deepcopy doesnt work here due to weight_norm
        self.teacher_predictor.load_state_dict(self.student_predictor.state_dict()) 
        
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_predictor.parameters():
            p.requires_grad = False
        
        if self.use_separate_mask_predictor: # Initialize separate mask predictor if requested
            self.student_mask_predictor = DINOHead(
                in_dim=embedding_dim,
                out_dim=projection_output_dim,
                hidden_dim=projection_hidden_dim,
                bottleneck_dim=projection_bottleneck_dim,
                nlayers=projection_nlayers,
                remove_last_layer=True,
                normalize=True
            )
            self.teacher_mask_predictor = DINOHead(
                in_dim=embedding_dim,
                out_dim=projection_output_dim,
                hidden_dim=projection_hidden_dim,
                bottleneck_dim=projection_bottleneck_dim,
                nlayers=projection_nlayers,
                remove_last_layer=True,
                normalize=True
            )
            self.teacher_mask_predictor.load_state_dict(self.student_mask_predictor.state_dict())
            for p in self.teacher_mask_predictor.parameters():
                p.requires_grad = False
        else:
            self.student_mask_predictor = self.student_predictor
            self.teacher_mask_predictor = self.teacher_predictor
    
    def forward(self, view1, view2, atlas, mask=[None, None]):
        """
        Forward pass through the DINO model with symmetric loss.

        Args:
            view1 (torch.Tensor): First view of the input data, shape (B, C, T)
            view2 (torch.Tensor): Second view of the input data, shape (B, C, T)
            atlas: Atlas identifiers for each view
            mask: Optional masks for each view

        Returns:
            tuple: (total_loss, log_data)
                total_loss (torch.Tensor): The combined DINO distillation loss
                log_data (dict): Dictionary with logging information
        """
        do_masking = mask[0] is not None and mask[1] is not None and (mask[0].any() or mask[1].any())

        patch_mask1 = mask[0] if do_masking else None
        patch_mask2 = mask[1] if do_masking else None
        batch_size = view1.shape[0] if isinstance(view1, torch.Tensor) else view1[0].shape[0]
        device = view1.device if isinstance(view1, torch.Tensor) else view1[0].device

        with torch.no_grad():
            t_embedding1 = self.teacher_encoder(view1, atlas[0])["tokens"]
            t_embedding2 = self.teacher_encoder(view2, atlas[1])["tokens"]

            t_cls_proj1 = self.teacher_predictor(t_embedding1[:, 0])
            t_cls_proj2 = self.teacher_predictor(t_embedding2[:, 0])

            # Teacher mask projections (computed here to stay in no_grad context)
            t_mask_proj1, t_mask_proj2 = None, None
            if do_masking and self.mask_loss_weight > 0:
                teacher_mask1_flat = patch_mask1.view(batch_size, -1)
                teacher_mask2_flat = patch_mask2.view(batch_size, -1)

                t_mask_tokens1 = t_embedding1[:, 1:][teacher_mask1_flat]
                t_mask_tokens2 = t_embedding2[:, 1:][teacher_mask2_flat]
                t_mask_proj1 = self.teacher_mask_predictor(t_mask_tokens1)
                t_mask_proj2 = self.teacher_mask_predictor(t_mask_tokens2)

        s_embedding1 = self.student_encoder(view1, atlas[0], patch_mask1)["tokens"]
        s_embedding2 = self.student_encoder(view2, atlas[1], patch_mask2)["tokens"]

        s_cls_proj1 = self.student_predictor(s_embedding1[:, 0])
        s_cls_proj2 = self.student_predictor(s_embedding2[:, 0])

        # CLS loss
        student_feat_list = [s_cls_proj1, s_cls_proj2]
        teacher_feat_list = [t_cls_proj1, t_cls_proj2]
        cls_loss, _ = MCRLoss(out_dim=256, expa_type=1, reduce_cov=0,
                              eps=0.05, coeff=self.coeff, center=False).forward(
                                  student_feat_list, teacher_feat_list, no_diag=True, normalized=True)

        network_loss = torch.tensor(0.0, device=device)
        weight_is_positive = (torch.any(self.current_network_loss_weight > 0)
                             if isinstance(self.current_network_loss_weight, torch.Tensor)
                             else self.current_network_loss_weight > 0)

        if weight_is_positive:
            # Extract and reshape network-patch tokens from student embeddings
            s_tokens1_no_cls = s_embedding1[:, 1:]  # [B, N*P, D]
            s_tokens2_no_cls = s_embedding2[:, 1:]  # [B, N*P, D]

            # Reshape to 2D grid: [B, N*P, D] -> [B, N, P, D]
            s_tokens1_reshaped = s_tokens1_no_cls.reshape(batch_size, self.num_networks, self.num_patches, -1)
            s_tokens2_reshaped = s_tokens2_no_cls.reshape(batch_size, self.num_networks, self.num_patches, -1)

            # Average across time patches: [B, N, P, D] -> [B, N, D]
            s_net_avg1 = s_tokens1_reshaped.mean(dim=2)
            s_net_avg2 = s_tokens2_reshaped.mean(dim=2)

            # Project network tokens through predictor
            s_net_proj1 = self.student_predictor(s_net_avg1.view(-1, self.embedding_dim))
            s_net_proj2 = self.student_predictor(s_net_avg2.view(-1, self.embedding_dim))
            s_net_proj1 = s_net_proj1.view(batch_size, self.num_networks, -1)
            s_net_proj2 = s_net_proj2.view(batch_size, self.num_networks, -1)

            # Same processing for teacher tokens
            t_tokens1_no_cls = t_embedding1[:, 1:]  # [B, N*P, D]
            t_tokens2_no_cls = t_embedding2[:, 1:]  # [B, N*P, D]

            t_tokens1_reshaped = t_tokens1_no_cls.reshape(batch_size, self.num_networks, self.num_patches, -1)
            t_tokens2_reshaped = t_tokens2_no_cls.reshape(batch_size, self.num_networks, self.num_patches, -1)

            t_net_avg1 = t_tokens1_reshaped.mean(dim=2)
            t_net_avg2 = t_tokens2_reshaped.mean(dim=2)

            t_net_proj1 = self.teacher_predictor(t_net_avg1.view(-1, self.embedding_dim))
            t_net_proj2 = self.teacher_predictor(t_net_avg2.view(-1, self.embedding_dim))
            t_net_proj1 = t_net_proj1.view(batch_size, self.num_networks, -1)
            t_net_proj2 = t_net_proj2.view(batch_size, self.num_networks, -1)

            # Flatten network features for MCRLoss: [B, N, proj_dim] -> [B*N, proj_dim]
            student_net_feat_list = [s_net_proj1.reshape(-1, s_net_proj1.shape[-1]),
                                     s_net_proj2.reshape(-1, s_net_proj2.shape[-1])]
            teacher_net_feat_list = [t_net_proj1.reshape(-1, t_net_proj1.shape[-1]),
                                     t_net_proj2.reshape(-1, t_net_proj2.shape[-1])]

            network_loss, _ = MCRLoss(out_dim=256, expa_type=1, reduce_cov=0,
                                     eps=0.05, coeff=self.coeff, center=False).forward(
                                         student_net_feat_list, teacher_net_feat_list,
                                         no_diag=True, normalized=True)
            network_loss = network_loss * self.current_network_loss_weight

        mask_loss = torch.tensor(0.0, device=device)
        patch_loss = torch.tensor(0.0, device=device)
        network_mask_loss = torch.tensor(0.0, device=device)

        compute_mask_loss = self.mask_loss_weight > 0 and do_masking
        if compute_mask_loss:
            patch_mask1_flat = patch_mask1.reshape(batch_size, -1)
            patch_mask2_flat = patch_mask2.reshape(batch_size, -1)

            # Extract masked tokens from student embeddings
            s_mask_tokens1 = s_embedding1[:, 1:][patch_mask1_flat]
            s_mask_tokens2 = s_embedding2[:, 1:][patch_mask2_flat]

            # Generate predictions for masked tokens
            s_mask_preds1 = self.student_mask_predictor(s_mask_tokens1)
            s_mask_preds2 = self.student_mask_predictor(s_mask_tokens2)

            student_patch_tokens_masked = torch.cat([s_mask_preds1, s_mask_preds2], dim=0)
            teacher_patch_tokens_masked = torch.cat([t_mask_proj1, t_mask_proj2], dim=0)

            student_masks_flat = torch.cat([patch_mask1_flat, patch_mask2_flat], dim=0)
            n_masked_patches = student_patch_tokens_masked.shape[0]

            masks_weight = ((1 / student_masks_flat.sum(-1).clamp(min=1.0))
                           .unsqueeze(-1).expand_as(student_masks_flat)[student_masks_flat])

            ibot_patch_loss = CosinePatchLoss(patch_out_dim=256, center=False)
            patch_loss, _ = ibot_patch_loss.forward_masked(
                student_patch_tokens_masked, teacher_patch_tokens_masked,
                student_masks_flat, n_masked_patches, masks_weight)

            mask_loss = (patch_loss + network_mask_loss) * self.mask_loss_weight

        total_loss = cls_loss + network_loss + mask_loss
        if isinstance(total_loss, torch.Tensor) and total_loss.numel() > 1:
            total_loss = total_loss.mean()

        # Handle both scalar and tensor network_loss for logging
        network_loss_log = (network_loss.mean().item()
                           if isinstance(network_loss, torch.Tensor) and network_loss.numel() > 1
                           else network_loss.item())

        log_data = {
            'cls_loss': cls_loss.item(),
            'network_loss': network_loss_log,
            'teacher_proj_std': torch.std(t_cls_proj1.float(), dim=0).mean().item(),
            'student_proj_std': torch.std(s_cls_proj1.float(), dim=0).mean().item(),
            'within_view_loss': patch_loss.item() if do_masking else 0.0,
            'network_mask_loss': network_mask_loss.item(),
            'mask_loss': mask_loss.item() if do_masking else 0.0,
            'total_loss': total_loss.item(),
        }

        return total_loss, log_data
    
    def _get_teacher_momentum(self, current_step, total_steps):
        if total_steps <= 1:
            return self.base_teacher_momentum
        
        # Cosine schedule from base_momentum to 1.0
        cos_val = math.cos(math.pi * current_step / total_steps) * 0.5 + 0.5
        return 1.0 - (1.0 - self.base_teacher_momentum) * cos_val
    
    def update_teacher(self, current_step, total_steps):
        # Get current momentum value
        m = self._get_teacher_momentum(current_step, total_steps)
        
        # Update teacher encoder parameters
        for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                               self.teacher_encoder.parameters()):
            teacher_param.data.mul_(m).add_((1 - m) * student_param.detach().data)
        
        # Update teacher predictor parameters
        for student_param, teacher_param in zip(self.student_predictor.parameters(), 
                                               self.teacher_predictor.parameters()):
            teacher_param.data.mul_(m).add_((1 - m) * student_param.detach().data)
        
        # Update teacher mask predictor parameters if using separate predictor
        if self.use_separate_mask_predictor:
            for student_param, teacher_param in zip(self.student_mask_predictor.parameters(),
                                                   self.teacher_mask_predictor.parameters()):
                teacher_param.data.mul_(m).add_((1 - m) * student_param.detach().data)
    
    def extract_features(self, view, atlas_name, use_teacher=True, feature_type=None, network_mask=None):
        """
        Extract features from the encoder (student or teacher).

        Args:
            view (torch.Tensor): Input data, shape (B, C, T)
            atlas_name: Atlas identifier
            use_teacher (bool): Whether to use the teacher encoder (default: True)
            feature_type (str): Type of features to extract ('cls', 'cls_avg', etc.)
            res (torch.Tensor): Resolution tensor
            network_mask (Optional[int]): If provided, mask all networks EXCEPT this one (0-indexed).
                                        Creates in-distribution masking like during pretraining.

        Returns:
            torch.Tensor: Extracted features, shape (B, D_out)
        """
        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        B = view.shape[0] if isinstance(view, torch.Tensor) else view[0].shape[0]
        device = view.device if isinstance(view, torch.Tensor) else view[0].device

        # Create network masking tensor if specified
        mask_tensor = None
        if network_mask is not None:
            # Get network dimensions from the encoder
            N = encoder.inter_network_transformer.num_networks
            P = encoder.inter_network_transformer.num_patches

            # Create mask: [B, N, P] where True = masked, False = keep
            # Mask all networks except the specified one, expanded for batch dimension
            mask_tensor = torch.ones(B, N, P, dtype=torch.bool, device=device)
            mask_tensor[:, network_mask, :] = False  # Keep only specified network unmasked

        with torch.no_grad():
            # Use autocast for compatibility with FlashAttention
            with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', enabled=(device.type == 'cuda')):
                dict = encoder(view, atlas_name, mask=mask_tensor)
        if feature_type == 'cls':
            features = dict['global_cls']
        elif feature_type == 'cls_avg':
            avg = dict['tokens'][:,1:].mean(dim=1)
            features = torch.cat([avg, dict['global_cls']], dim=1)
        return features 