"""
Model wrapper for clean encoder extraction and unified interface.
Provides a consistent API for different model architectures to be used
in downstream tasks (linear probing, finetuning, etc.)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
import copy
import os


class ModelWrapper:
    """
    Unified interface for different model architectures.
    Provides clean access to encoders without projection heads.
    """
    
    def __init__(self, model: nn.Module, model_type: str = 'simdino', config: Optional[Dict] = None):
        """
        Args:
            model: The full model (e.g., SimDINOModel)
            model_type: Type of model ('simdino', 'mae', 'clip', etc.)
            config: Optional config dict with model-specific parameters
        """
        self.model = model
        self.model_type = model_type
        self.config = config or {}
        
        # Handle DataParallel wrapper if present
        self._unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
        
    def get_encoder(self, use_teacher: bool = False) -> nn.Module:
        """
        Extract encoder without projection heads.
        
        Args:
            use_teacher: For models with teacher-student, whether to use teacher encoder
            
        Returns:
            The encoder module
        """
        if self.model_type in ['simdino', 'simple_dino_tf']:
            if use_teacher:
                return self._unwrapped_model.teacher_encoder
            else:
                return self._unwrapped_model.student_encoder
        elif self.model_type == 'mae':
            # Example for other architectures - no teacher/student distinction
            return self._unwrapped_model.encoder
        else:
            # Fallback: assume the model itself is the encoder
            return self._unwrapped_model
            
    def extract_features(
        self, 
        x: Union[torch.Tensor, List[torch.Tensor]], 
        atlas_idx: int = 0,
        use_teacher: bool = True,
        feature_type: str = 'cls_avg',
        res: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract features from the encoder.
        
        Args:
            x: Input data
            atlas_idx: Atlas index for multi-atlas models
            use_teacher: Whether to use teacher encoder
            feature_type: Type of features ('cls', 'cls_avg', 'avg', etc.)
            res: Resolution tensor for multi-resolution models
            **kwargs: Additional model-specific arguments
            
        Returns:
            Extracted features
        """
        if self.model_type == 'simdino':
            # Use the model's built-in extract_features method
            return self._unwrapped_model.extract_features(
                x, atlas_idx, use_teacher=use_teacher, 
                feature_type=feature_type, res=res
            )
        else:
            # Generic feature extraction
            encoder = self.get_encoder(use_teacher)
            encoder.eval()
            
            with torch.no_grad():
                if hasattr(encoder, 'extract_features'):
                    features = encoder.extract_features(x, **kwargs)
                else:
                    # Standard forward pass
                    output = encoder(x, **kwargs)
                    
                    # Handle different output types
                    if isinstance(output, dict):
                        if feature_type == 'cls' and 'cls' in output:
                            features = output['cls']
                        elif feature_type == 'cls_avg' and 'tokens' in output:
                            cls = output.get('cls', output['tokens'][:, 0])
                            avg = output['tokens'][:, 1:].mean(dim=1)
                            features = torch.cat([avg, cls], dim=1)
                        else:
                            features = output.get('features', output.get('tokens', output))
                    else:
                        features = output
                        
            return features
            
    def get_feature_dim(self, feature_type: str = 'cls_avg') -> int:
        """
        Get the dimensionality of extracted features.
        
        Args:
            feature_type: Type of features
            
        Returns:
            Feature dimension
        """
        if self.model_type in ['simdino', 'simple_dino_tf']:
            try:
                # Try to get from encoder first
                encoder = self.get_encoder(use_teacher=True)  # Use teacher by default for dim inference
                base_dim = encoder.embedding_dim
            except:
                # Fallback to config for simple_dino_tf
                if 'model' in self.config and 'embedding_dim' in self.config['model']:
                    base_dim = self.config['model']['embedding_dim']
                elif 'embedding_dim' in self.config:
                    base_dim = self.config['embedding_dim']
                else:
                    raise ValueError(f"No embedding_dim found in config for model type {self.model_type}")
            
            if feature_type == 'cls_avg':
                return base_dim * 2  # cls + avg concatenated
            else:
                return base_dim
        else:
            # Try to infer from config
            if 'embedding_dim' in self.config:
                base_dim = self.config['embedding_dim']
                return base_dim * 2 if feature_type == 'cls_avg' else base_dim
            else:
                raise NotImplementedError(f"Cannot infer feature dim for model type {self.model_type}")
                
    def prepare_for_finetuning(self, freeze_encoder_epochs: int = 0, use_teacher: bool = False) -> nn.Module:
        """
        Prepare encoder for finetuning.
        
        Args:
            freeze_encoder_epochs: Number of initial epochs to freeze encoder
            use_teacher: Whether to use teacher or student encoder
            
        Returns:
            Encoder ready for finetuning
        """
        encoder = self.get_encoder(use_teacher=use_teacher)
        
        # Remove any hooks or special training modes
        encoder.train()
        
        # Always unfreeze first, then optionally freeze if needed
        for param in encoder.parameters():
            param.requires_grad = True
        
        # Optionally freeze initially  
        if freeze_encoder_epochs > 0:
            for param in encoder.parameters():
                param.requires_grad = False
                
        return encoder
        
    def unfreeze_encoder(self, encoder: nn.Module):
        """Unfreeze encoder parameters for training."""
        for param in encoder.parameters():
            param.requires_grad = True
            
    @staticmethod
    def from_checkpoint(
        checkpoint_path: str, 
        config: Dict,
        device: torch.device = torch.device('cpu'),
        use_teacher: bool = True
    ) -> 'ModelWrapper':
        """
        Create a ModelWrapper from a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint
            config: Model configuration
            device: Device to load model on
            use_teacher: Whether to extract teacher or student
            
        Returns:
            ModelWrapper instance
        """
        # Import here to avoid circular dependency
        from simdino import SimDINOModel
        
        model_type = config.get('model_type', 'simdino')
        
        if model_type == 'simdino':
            # Create model instance
            model_cfg = config['model']
            data_cfg = config['data']
            dino_cfg = config['dino']
            ssl_cfg = config['ssl']
            mask_cfg = config['masking']
            
            # Process atlas names
            atlas_names = data_cfg['atlas_combo'][0].split(",")
            for i, an in enumerate(atlas_names):
                suffix = f"_{data_cfg['num_networks'][i]}n" if "schaefer" in an else ""
                atlas_names[i] = an.replace("timeseries/", "") + suffix
            
            # Calculate derived parameters
            mlp_dim = int(model_cfg['embedding_dim'] * 4)
            heads = int(model_cfg['embedding_dim'] / 64)
            
            model = SimDINOModel(
                patch_size=data_cfg['patch_size'],
                target_time_length=data_cfg['target_signal_length'],
                cnn_dim=model_cfg['cnn_dim'],
                cnn_final_norm=model_cfg['cnn_final_norm'],
                embedding_dim=model_cfg['embedding_dim'],
                depth=model_cfg['depth'],
                mlp_dim=mlp_dim,
                heads=heads,
                global_pooling=model_cfg['global_pooling'],
                drop_path_rate=model_cfg['drop_path_rate'],
                layer_scale_init_value=model_cfg.get('layer_scale_init_value', None),
                emb_dropout=model_cfg['emb_dropout'],
                network_data_path=data_cfg['network_map_path'],
                atlas_names=atlas_names,
                atlas_resolutions=data_cfg['atlas_resolutions'],
                projection_hidden_dim=model_cfg['projection_hidden_dim'],
                projection_output_dim=model_cfg['projection_output_dim'],
                projection_bottleneck_dim=model_cfg['projection_bottleneck_dim'],
                projection_nlayers=model_cfg['projection_nlayers'],
                center_momentum=dino_cfg['center_momentum'],
                base_teacher_momentum=dino_cfg['base_teacher_momentum'],
                coeff=dino_cfg['coeff'],
                mask_loss_views=dino_cfg.get('mask_loss_views', 'straight'),
                do_masking=mask_cfg['masking_frequency'] > 0,
                mask_loss_weight=ssl_cfg['mask_loss_weight'],
                use_separate_mask_predictor=dino_cfg['use_separate_mask_predictor'],
                backbone_type=model_cfg.get('backbone_type', 'cnn_tf'),
                max_crop_distance=data_cfg.get('max_crop_distance', 80.0),
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            # Handle DataParallel keys
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
        else:
            raise NotImplementedError(f"Model type {model_type} not yet supported")
            
        return ModelWrapper(model, model_type, config)
    
    def eval(self):
        """Set the wrapped model to evaluation mode."""
        return self.model.eval()
        
    def train(self, mode: bool = True):
        """Set the wrapped model to training mode."""
        return self.model.train(mode)
        
    def to(self, device):
        """Move the wrapped model to the specified device."""
        self.model = self.model.to(device)
        self._unwrapped_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return self
        
    def cuda(self):
        """Move the wrapped model to CUDA."""
        return self.to(torch.device('cuda'))
        
    def cpu(self):
        """Move the wrapped model to CPU."""
        return self.to(torch.device('cpu'))

