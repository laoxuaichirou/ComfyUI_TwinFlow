# Local implementations for Z-Image components
# These provide fallbacks when diffusers doesn't include Z-Image support

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any, Dict

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    HAS_DIFFUSERS_MIXINS = True
except ImportError:
    # Fallback if these aren't available
    ConfigMixin = object
    ModelMixin = nn.Module
    HAS_DIFFUSERS_MIXINS = False
    def register_to_config(func):
        return func

# Try to import FromSingleFileMixin for loading single checkpoint files
try:
    from diffusers.loaders import FromSingleFileMixin
    HAS_FROM_SINGLE_FILE = True
except ImportError:
    FromSingleFileMixin = object
    HAS_FROM_SINGLE_FILE = False


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1-D Tensor of N indices, one per batch element
        embedding_dim: the dimension of the output
        flip_sin_to_cos: if True, flip sin and cos order
        downscale_freq_shift: frequency shift for downscaling
        scale: scale factor
        max_period: controls the minimum frequency of the embeddings
    
    Returns:
        Tensor of shape [N, embedding_dim] with positional embeddings
    """
    assert len(timesteps.shape) == 1, "Timesteps should be 1D"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb

    # concat sin and cos embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sin and cos
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None) # gguf need to know the compute_dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def _build_base_classes():
    """Build the base classes tuple dynamically based on available imports."""
    bases = []
    if HAS_DIFFUSERS_MIXINS:
        bases.append(ModelMixin)
        bases.append(ConfigMixin)
    if HAS_FROM_SINGLE_FILE:
        bases.append(FromSingleFileMixin)
    if not bases:
        bases.append(nn.Module)
    return tuple(bases)


class ZImageTransformer2DModelBase(*_build_base_classes()):
    """
    Base class for ZImageTransformer2DModel when the real one isn't available in diffusers.
    
    This provides the minimal interface needed for the wrapper to work.
    The actual model weights and forward logic come from loading pretrained weights.
    """
    
    # Required by FromSingleFileMixin
    _optional_components = []
    _no_split_modules = []
    
    # These will be set by config or during loading
    all_patch_size: Tuple[int, ...] = (2,)
    all_f_patch_size: Tuple[int, ...] = (1,)
    t_scale: float = 1000.0
    gradient_checkpointing: bool = False
    
    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ):
        super().__init__()
        
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.n_layers = n_layers
        self.n_refiner_layers = n_refiner_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.cap_feat_dim = cap_feat_dim
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        
        # Create the timestep embedder
        self.t_embedder = TimestepEmbedder(min(dim, 256), mid_size=1024)
        
        # Placeholder for components that will be loaded from weights
        # These would normally be created by the full ZImageTransformer2DModel
        self.noise_refiner = nn.ModuleList()
        self.context_refiner = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.all_x_embedder = nn.ModuleDict()
        self.all_final_layer = nn.ModuleDict()
        self.cap_embedder = nn.Identity()
        self.rope_embedder = nn.Identity()
        self.x_pad_token = nn.Parameter(torch.zeros(1))
        self.cap_pad_token = nn.Parameter(torch.zeros(1))
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
    
    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
    
    def patchify_and_embed(self, x, cap_feats, patch_size, f_patch_size):
        """Placeholder - actual implementation comes from loaded weights"""
        raise NotImplementedError(
            "This base class doesn't implement the full ZImageTransformer2DModel. "
            "Please update diffusers to version 0.36.0+ for full Z-Image support."
        )
    
    def unpatchify(self, unified, x_size, patch_size, f_patch_size):
        """Placeholder - actual implementation comes from loaded weights"""
        raise NotImplementedError(
            "This base class doesn't implement the full ZImageTransformer2DModel. "
            "Please update diffusers to version 0.36.0+ for full Z-Image support."
        )
    
    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        target_timestep=None,
    ):
        """Placeholder forward - actual implementation in wrapper or loaded model"""
        raise NotImplementedError(
            "This base class doesn't implement the full ZImageTransformer2DModel forward. "
            "Please update diffusers to version 0.36.0+ for full Z-Image support."
        )

