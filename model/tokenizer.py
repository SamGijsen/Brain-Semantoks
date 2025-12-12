import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math




class SLConv(nn.Module):
    """
    From: https://github.com/mackelab/neural_timeseries_diffusion
    Structured Long Convolutional layer.
    Adapted from https://github.com/ctlllll/SGConv

    Args:
        kernel_size: Kernel size used to build convolution.
        num_channels: Number of channels.
        num_scales: Number of scales.
            Overall length will be: kernel_size * (2 ** (num_scales - 1))
        decay_min: Minimum decay. Advanced option.
        decay_max: Maximum decay. Advanced option.
        heads: Number of heads.
        padding_mode: Padding mode. Either "zeros" or "circular".
        use_fft_conv: Whether to use FFT convolution.
        interpolate_mode: Interpolation mode. Either "nearest" or "linear". Advanced option.
    """

    def __init__(
        self,
        kernel_size,
        num_channels,
        num_scales,
        decay_min=2.0,
        decay_max=2.0,
        heads=1,
        padding_mode="zeros",
        use_fft_conv=False,
        interpolate_mode="nearest",
    ):
        super().__init__()
        assert decay_min <= decay_max

        self.h = num_channels
        self.num_scales = num_scales
        self.kernel_length = kernel_size * (2 ** (num_scales - 1))

        self.heads = heads

        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.use_fft_conv = use_fft_conv
        self.interpolate_mode = interpolate_mode

        self.D = nn.Parameter(torch.randn(self.heads, self.h))

        total_padding = self.kernel_length - 1
        left_pad = total_padding // 2
        self.pad = [left_pad, total_padding - left_pad]

        # Init of conv kernels. There are more options here.
        # Full kernel is always normalized by initial kernel norm.
        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            kernel = nn.Parameter(torch.randn(self.heads, self.h, kernel_size))
            self.kernel_list.append(kernel)

        # Support multiple scales. Only makes sense in non-sparse setting.
        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1),
        )
        self.register_buffer("kernel_norm", torch.ones(self.heads, self.h, 1))
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

    def forward(self, x):
        signal_length = x.size(-1)

        kernel_list = []
        for i in range(self.num_scales):
            kernel = F.interpolate(
                self.kernel_list[i],
                scale_factor=2 ** (max(0, i - 1)),
                mode=self.interpolate_mode,
            ) * self.multiplier ** (self.num_scales - i - 1)
            kernel_list.append(kernel)
        k = torch.cat(kernel_list, dim=-1)

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device
            )

        assert k.size(-1) <= signal_length
        if self.use_fft_conv:
            k = F.pad(k, (0, signal_length - k.size(-1)))

        k = k / self.kernel_norm

        # Convolution
        if self.use_fft_conv:
            if self.padding_mode == "constant":
                factor = 2
            elif self.padding_mode == "circular":
                factor = 1

            k_f = torch.fft.rfft(k, n=factor * signal_length)  # (C H L)
            u_f = torch.fft.rfft(x, n=factor * signal_length)  # (B H L)
            y_f = torch.einsum("bhl,chl->bchl", u_f, k_f)
            slice_start = self.kernel_length // 2
            y = torch.fft.irfft(y_f, n=factor * signal_length)

            if self.padding_mode == "constant":
                y = y[..., slice_start : slice_start + signal_length]  # (B C H L)
            elif self.padding_mode == "circular":
                y = torch.roll(y, -slice_start, dims=-1)
            y = rearrange(y, "b c h l -> b (h c) l")
        else:
            # Pytorch implements convolutions as cross-correlations! flip necessary
            y = F.conv1d(
                F.pad(x, self.pad, mode=self.padding_mode),
                rearrange(k.flip(-1), "c h l -> (h c) 1 l"),
                groups=self.h,
            )

        # Compute D term in state space equation - essentially a skip connection
        y = y + rearrange(
            torch.einsum("bhl,ch->bchl", x, self.D),
            "b c h l -> b (h c) l",
        )

        return y

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding="same", dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SLConvBranch(nn.Module):
    def __init__(self, num_rois_in_network, out_channels, slconv_config):
        super().__init__()
        # 1. Calculate heads to ensure output is >= desired out_channels
        heads = math.ceil(out_channels / num_rois_in_network)
        
        self.slconv = SLConv(
            kernel_size=slconv_config['kernel_size'],
            num_scales=slconv_config['num_scales'],
            num_channels=num_rois_in_network,
            heads=heads,
            decay_min=slconv_config['decay_min'],
            decay_max=slconv_config['decay_max'],
        )
        
        # 2. Linear layer for projection and channel mixing
        slconv_out_features = heads * num_rois_in_network
        self.projection = nn.Linear(slconv_out_features, out_channels)

    def forward(self, x):
        # Input x shape: (batch, channels, length)
        x = self.slconv(x)
        
        # To apply Linear to the channel dimension, we need to permute
        # (batch, channels, length) -> (batch, length, channels)
        x = x.permute(0, 2, 1)
        
        x = self.projection(x)
        
        # Permute back to the standard conv1d format
        # (batch, length, channels) -> (batch, channels, length)
        x = x.permute(0, 2, 1)
        
        return x

class AttentionPooling1d(nn.Module):
    """Single-head attention pooling over temporal dimension."""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, x):
        # x: [B, C, L] -> attention over L dimension
        # Transpose to [B, L, C] for linear layer
        x_transposed = x.transpose(1, 2)  # [B, L, C]
        attention_scores = self.attention(x_transposed)  # [B, L, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [B, L, 1]
        # Apply weights and sum over temporal dimension
        weighted = x * attention_weights.transpose(1, 2)  # [B, C, L]
        return torch.sum(weighted, dim=2)  # [B, C]


class FlexibleNetworkTokenizer(nn.Module):
    def __init__(self, num_rois_in_network, config, final_norm='layer', pooling_type='mean'):
        super().__init__()
        self.branches = nn.ModuleList()
        total_out_channels = 0

        for branch_cfg in config:
            total_out_channels += branch_cfg['out_channels']
            branch_type = branch_cfg['type']

            # print(branch_cfg, branch_type)
            if branch_type == 'sgconv':
                self.branches.append(
                    SLConvBranch(
                        num_rois_in_network,
                        branch_cfg['out_channels'],
                        branch_cfg
                    )
                )
            
            elif branch_type in ['dense', 'dilated']:
                is_depthwise = branch_cfg.get('depthwise', False)
                kernel_size = branch_cfg['kernel_size']
                dilation = branch_cfg.get('dilation', 1)
                
                if is_depthwise:
                    self.branches.append(
                        DepthwiseSeparableConv1d(num_rois_in_network, branch_cfg['out_channels'], kernel_size, dilation)
                    )
                else:
                    self.branches.append(
                        nn.Conv1d(num_rois_in_network, branch_cfg['out_channels'], kernel_size, padding='same', dilation=dilation)
                    )
            
        self.activation = nn.GELU()

        # Initialize pooling based on type
        if pooling_type == 'mean':
            self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        elif pooling_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        elif pooling_type == 'attention':
            self.pool = AttentionPooling1d(total_out_channels)
        else:
            raise ValueError(f"Invalid pooling_type: {pooling_type}. Must be 'mean', 'max', or 'attention'")

        self.pooling_type = pooling_type

        if final_norm == "layer":
            self.norm = nn.LayerNorm(total_out_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        
        x = torch.cat(branch_outputs, dim=1)
        
        x = self.activation(x)
        x = self.pool(x)

        if self.pooling_type in ['mean', 'max']:
            x = x.squeeze(-1)

        x = self.norm(x)
        
        return x

    
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding="same", dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))