import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from .tokenizer import FlexibleNetworkTokenizer

try:
    from flash_attn import flash_attn_qkvpacked_func
    if not torch.cuda.is_available():
        _FLASH_ATTN_AVAILABLE = False
    else:
        _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class CNN_TF(nn.Module):
    def __init__(
        self,
        target_time_length: int,
        patch_size: int,
        network_data_path: str,
        atlas_names: List[str],
        embedding_dim: int = 128,
        depth: int = 3,
        heads: int = 4,
        mlp_dim: int = 512,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        global_pooling: str = 'cls', # For InterNetworkTransformer
        emb_dropout: float = 0.0,   # For InterNetworkTransformer
        num_classes: int = 0,
        head_dropout: float = 0.0,
        do_masking: bool = False,    # For InterNetworkTransformer input tokens
                
        # Tokenizer options
        tokenizer_config: Optional[List[dict]] = None,  # Configuration for FlexibleNetworkTokenizer
        tokenizer_final_norm: str = "layer",  # Final normalization for tokenizer
        tokenizer_pooling_type: str = "mean",  # Pooling type: 'mean', 'max', 'attention'
        
        # Flash attention option  
        flash_attention: str = "auto",  # "auto", "flash_attn", or "normal"
        
        # Spatial resolution modes (bypass network map loading)
        max_spatial: bool = False,  # If True, each ROI is its own network
        min_spatial: bool = False,  # If True, all ROIs constitute a single network  
        total_rois: Optional[int] = None,  # Total ROI count when using spatial modes
        atlas_network_counts: Optional[List[int]] = None,  # Network counts per atlas for standard mode
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.atlas_names = atlas_names

        self.patch_size = patch_size

        self.tokenizer_config = tokenizer_config
        self.tokenizer_final_norm = tokenizer_final_norm
        self.tokenizer_pooling_type = tokenizer_pooling_type

        # Handle spatial modes - skip network map loading if using max/min spatial
        if max_spatial or min_spatial:
            if total_rois is None:
                raise ValueError("total_rois must be provided when using max_spatial or min_spatial modes")
            
            self.total_num_rois = total_rois
            self.is_max_spatial = max_spatial
            self.is_min_spatial = min_spatial
            
            if max_spatial:
                self.num_networks = total_rois  # Each ROI = 1 network
                self.network_map = torch.arange(total_rois, dtype=torch.long)  # [0,1,2,...,total_rois-1]
            else:  # min_spatial
                self.num_networks = 1  # All ROIs = 1 network  
                self.network_map = torch.zeros(total_rois, dtype=torch.long)  # All ROIs → network 0
        else:
            # Standard mode - load from network map file using network counts like dataset does
            if atlas_network_counts is None:
                raise ValueError("atlas_network_counts must be provided for standard mode")
                
            network_data = dict(np.load(network_data_path))
            network_maps_list = []
            max_network_id = -1
            for atlas_name, network_count in zip(self.atlas_names, atlas_network_counts):
                # Use same logic as dataset: suffix if network_count > 1
                if network_count > 1:
                    map_key_in_npz = f"network_map_{atlas_name}_{network_count}n"
                else:
                    map_key_in_npz = f"network_map_{atlas_name}"
                    
                if map_key_in_npz not in network_data:
                    raise ValueError(f"Network map for atlas {atlas_name} (key: {map_key_in_npz}) not found in {network_data_path}")
                nwd = torch.tensor(network_data[map_key_in_npz], dtype=torch.long)
                nwd = nwd - (nwd.min()-max_network_id) + 1
                max_network_id = nwd.max()
                network_maps_list.append(nwd)
            self.network_map = torch.cat(network_maps_list)

            self.num_networks = len(torch.unique(self.network_map))
            self.total_num_rois = len(self.network_map)
            
            self.is_max_spatial = (self.num_networks == self.total_num_rois)  # Each ROI = 1 network
            self.is_min_spatial = (self.num_networks == 1)  # All ROIs = 1 network

        self.patch_size = patch_size
        self.patch_stride = patch_size
        # self.num_patches = target_time_length // patch_size
        self.num_patches = (target_time_length - self.patch_size) // self.patch_stride + 1
        self.global_pooling = global_pooling # For InterNetworkTransformer

        network_roi_counts = []
        for unique_network in torch.unique(self.network_map):
            network_roi_counts.append((self.network_map == unique_network).sum())
        self.network_roi_counts = network_roi_counts

        self.shared_roi_projection = None
        self.network_tokenizers = None

        if self.is_max_spatial:
            # Max spatial: Use shared linear projection for all ROIs
            self.shared_roi_projection = nn.Linear(self.patch_size, embedding_dim)
            print(f"Using shared linear projection for max spatial mode ({self.total_num_rois} ROIs)")
        else: 
            # Standard mode: Create network-specific tokenizers
            if self.tokenizer_config is None:
                # Fallback to simple linear tokenizer if no config provided
                self.tokenizer_config = [{'type': 'dense', 'kernel_size': 3, 'depthwise': False, 'out_channels': embedding_dim}]

            self.network_tokenizers = nn.ModuleDict()
            for atlas_name_iter in self.atlas_names:
                self.network_tokenizers[atlas_name_iter] = nn.ModuleList([
                    FlexibleNetworkTokenizer(
                        num_rois_in_network=self.network_roi_counts[i],
                        config=self.tokenizer_config,
                        final_norm=self.tokenizer_final_norm,
                        pooling_type=self.tokenizer_pooling_type
                    )
                    for i in range(self.num_networks)
                ])

        # Calculate actual embedding dimension based on tokenizer config or fallback
        if self.is_max_spatial:
            actual_embedding_dim = embedding_dim
        elif self.tokenizer_config is not None:
            actual_embedding_dim = sum(branch['out_channels'] for branch in self.tokenizer_config)
        else:
            actual_embedding_dim = embedding_dim

        num_effective_atlases_for_inter_tf = 1 # backward compatible
        self.inter_network_transformer = InterNetworkTransformer(
            num_networks=self.num_networks,
            num_patches=self.num_patches,
            embedding_dim=actual_embedding_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            drop_path_rate=drop_path_rate,
            emb_dropout=emb_dropout,
            num_atlases=num_effective_atlases_for_inter_tf, # This reflects input to InterNetworkTransformer
            do_masking=do_masking,
            layer_scale_init_value=layer_scale_init_value,
            flash_attention=flash_attention,
        )

        self.norm = nn.LayerNorm(actual_embedding_dim)
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(actual_embedding_dim, num_classes)
            )
        else:
            self.head = None

    def _forward_max_spatial(self, x_patches, batch_size, device):
        """Efficient processing when each ROI = 1 network (maximum spatial resolution)"""
        B, P, C, patch_size = x_patches.shape
        
        # Reshape all patches for single pass through linear projection
        # [B, P, C, patch_size] -> [B*P*C, patch_size]
        reshaped = x_patches.reshape(B * P * C, patch_size)
        
        # Single linear projection for all ROIs
        linear_output = self.shared_roi_projection(reshaped)  # [B*P*C, D]
        
        # Reshape to [B, C, P, D] (networks=C, patches=P)
        output_tokens = linear_output.reshape(B, P, C, -1).transpose(1, 2)  # [B, C, P, D]
        
        return output_tokens

    def _forward_standard(self, x_patches, batch_size, device):
        """Standard processing using network-specific tokenizers"""
        # Calculate total output dimension from tokenizer config
        total_output_dim = sum(branch['out_channels'] for branch in self.tokenizer_config)
        output_tokens = torch.zeros(batch_size, self.num_networks, self.num_patches, total_output_dim, device=device)

        for network_idx in range(self.num_networks):
            network_mask = (self.network_map == network_idx)
            num_rois_in_network = network_mask.sum().item()
            current_tokenizer = self.network_tokenizers[self.atlas_names[0]][network_idx]

            network_patch_data = x_patches[:, :, network_mask, :]
            actual_patch_size = network_patch_data.shape[-1]
            reshaped_for_tokenizer = network_patch_data.reshape(
                batch_size * self.num_patches, num_rois_in_network, actual_patch_size
            )
            tokenizer_output = current_tokenizer(reshaped_for_tokenizer)
            tokenizer_output_final = tokenizer_output.reshape(batch_size, self.num_patches, total_output_dim)
            output_tokens[:, network_idx, :, :] = tokenizer_output_final

        return output_tokens

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], atlas: Optional[int] = 0, mask: Optional[torch.Tensor] = None,):
        if isinstance(x, list):
            x = x[0]
        single_atlas_data_tensor = x
        batch_size = single_atlas_data_tensor.shape[0]
        device = single_atlas_data_tensor.device

        # Process this single atlas data
        x_current_atlas_patched = single_atlas_data_tensor.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        x_current_atlas_patched_permuted = x_current_atlas_patched.permute(0, 2, 1, 3) # [B, num_patches, num_rois_atlas, patch_size]
        
        # Choose processing mode based on network configuration
        if self.is_max_spatial:
            output_tokens = self._forward_max_spatial(x_current_atlas_patched_permuted, batch_size, device)
        else:
            output_tokens = self._forward_standard(x_current_atlas_patched_permuted, batch_size, device)

        inter_network_transformer_atlas_pe_idx = 0
        inter_tokens = self.inter_network_transformer(output_tokens, atlas=inter_network_transformer_atlas_pe_idx, mask=mask)

        inter_tokens = self.norm(inter_tokens) # Norm after InterNetworkTransformer
        cls_token = inter_tokens[:, 0]

        logits = None
        if self.head is not None:
            logits = self.head(cls_token)

        return {
            'global_cls': cls_token,
            'tokens': inter_tokens, 
            'logits': logits
        }


class InterNetworkTransformer(nn.Module):
    def __init__(
        self,
        num_networks,
        num_patches,
        embedding_dim=128,
        depth=2,
        heads=4,
        mlp_dim=512,
        drop_path_rate=0.0,
        emb_dropout=0.0,
        mlp_dropout=0.0, 
        num_atlases=1, 
        do_masking=False,
        layer_scale_init_value: Optional[float] = None,
        flash_attention: str = "auto",  # "auto", "flash_attn", or "normal"
    ):
        super().__init__()

        # Resolve flash attention setting
        if flash_attention == "auto":
            attn_mode = 'flash_attn' if _FLASH_ATTN_AVAILABLE else 'normal'
            if not _FLASH_ATTN_AVAILABLE:
                warnings.warn("Flash attention not available, using standard attention. Install with: pip install flash-attn", UserWarning)
        elif flash_attention == "flash_attn":
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError("Flash attention requested but not available. Install with: pip install flash-attn")
            attn_mode = 'flash_attn'
        else:  # "normal"
            attn_mode = 'normal'
        
        self.attn_mode = attn_mode

        self.num_effective_atlases = num_atlases 
        self.num_networks = num_networks
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        if do_masking:
            self.mask_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
            nn.init.normal_(self.mask_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        if embedding_dim % 2 != 0:
            raise ValueError(
                f"Embedding dimension ({embedding_dim}) must be even for sincos positional encoding."
            )

        temporal_indices = torch.arange(num_patches, dtype=torch.float)
        temporal_pos_embedding = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(embedding_dim, temporal_indices.numpy())
        ).float()
        self.register_buffer('temporal_pos_embedding', temporal_pos_embedding)
        
        self.network_learnable_pos_embedding = nn.Embedding(num_networks, embedding_dim)
        nn.init.normal_(self.network_learnable_pos_embedding.weight, std=0.02)

        self.dropout = nn.Dropout(emb_dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embedding_dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=mlp_dropout,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                use_flash_attention=(attn_mode == 'flash_attn'),
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embedding_dim) 

    def forward(self, x, atlas=0, mask=None):
        if isinstance(x, list):
            x = x[0]
        batch_size = x.shape[0]

        if mask is not None and hasattr(self, 'mask_embedding'):
            x[mask] = self.mask_embedding.to(x.dtype)
        

        network_indices = torch.arange(self.num_networks, device=x.device)
        network_pe = self.network_learnable_pos_embedding(network_indices)
        
        temporal_pe = self.temporal_pos_embedding
        
        pos_embedding = network_pe.unsqueeze(1) + temporal_pe.unsqueeze(0)
        
        # Reshape x to match positional embedding structure: [B, N*P, D] -> [B, N, P, D]
        x = x.view(batch_size, self.num_networks, self.num_patches, -1)
        
        # pos_embedding is [N, P, D] or [N, P, 1, D], need to broadcast with [B, N, P, D]
        if pos_embedding.dim() == 4:
            pos_embedding = pos_embedding.squeeze(-2)  # Remove singleton dimension if present
        
        # Add positional embeddings: [B, N, P, D] + [N, P, D] (broadcast)
        x = x + pos_embedding.unsqueeze(0)
        
        # Reshape back to sequence: [B, N*P, D]
        x = x.reshape(batch_size, self.num_networks * self.num_patches, -1)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
                
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0., attn_mode='normal'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_mode = attn_mode

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        if self.attn_mode == 'flash_attn':
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:  # normal attention
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2)
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0, drop_path=0.0, layer_scale_init_value: Optional[float] = None, use_flash_attention: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Use unified attention class
        attn_mode = 'flash_attn' if use_flash_attention else 'normal'
        self.attn = Attention(dim, num_heads=heads, dropout=dropout, attn_mode=attn_mode)
            
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        if layer_scale_init_value:
            self.ls1 = LayerScale(dim, init_values=layer_scale_init_value)
            self.ls2 = LayerScale(dim, init_values=layer_scale_init_value)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_gamma=None, attn_beta=None, mlp_gamma=None, mlp_beta=None):
        # Attention block
        norm_x = self.norm1(x)
        if attn_gamma is not None and attn_beta is not None:
            norm_x = norm_x * (1 + attn_gamma.unsqueeze(1)) + attn_beta.unsqueeze(1)
        attn_output = self.attn(norm_x)
        x = x + self.drop_path1(self.ls1(attn_output))

        # MLP block
        norm_x = self.norm2(x)
        if mlp_gamma is not None and mlp_beta is not None:
            norm_x = norm_x * (1 + mlp_gamma.unsqueeze(1)) + mlp_beta.unsqueeze(1)
        mlp_output = self.mlp(norm_x)
        x = x + self.drop_path2(self.ls2(mlp_output))

        return x

