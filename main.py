"""MoonViT — Native-resolution vision encoder from the Kimi-VL Technical Report.

Reference
---------
Kimi Team. "Kimi-VL Technical Report." arXiv:2504.07491v3.
  https://arxiv.org/abs/2504.07491

What this file implements
-------------------------
A single-file, typed, PyTorch reference implementation of MoonViT and its
Kimi-VL MLP projector. Architectural choices follow the paper:

* **Initialized from SigLIP-SO-400M.** The default config matches
  ``google/siglip-so400m-patch14-384`` (hidden 1152, 27 layers, 16 heads,
  head-dim 72, MLP 4304, patch 14, pre-norm, GELU-tanh, LayerNorm, QKV-bias).

* **Native resolution via NaViT-style packing.** Input is a *list* of images
  at arbitrary resolutions. Each image is patch-embedded to its own
  ``(grid_h, grid_w)`` grid, flattened, then all images are concatenated
  along the sequence dimension into a single 1-D packed sequence.
  ``cu_seqlens`` drives variable-length attention so tokens never attend
  across image boundaries.

* **Dual positional encoding (additive).** The paper is explicit that both
  are kept: SigLIP's learnable absolute positional embedding
  (bicubically interpolated from the pretraining grid to each image's grid)
  is *added* to the patch embeddings, and **2-D RoPE** is applied inside the
  attention. Head-dim is split in half — the first half rotates with the
  H-axis position, the second half with the W-axis position.

* **Variable-length attention.** Uses ``flash_attn_varlen_func`` when
  available (and on CUDA); otherwise a block-diagonal per-image SDPA
  fallback runs standalone on CPU/MPS/CUDA without the flash-attn dep.

* **Kimi-VL MLP projector.** 2×2 pixel-shuffle (space-to-depth: spatial
  halves, channels 4×), then a two-layer MLP into the LLM hidden size.
  Included as a separate ``MLPProjector`` module — the encoder output is
  usable on its own.

Usage
-----
>>> import torch
>>> from main import MoonViT, MoonViTConfig, MLPProjector
>>> encoder = MoonViT(MoonViTConfig())
>>> imgs = [torch.randn(3, 224, 280), torch.randn(3, 140, 196)]
>>> out = encoder(imgs)
>>> out.last_hidden_state.shape  # (total_patches, 1152)

Notes
-----
* Patch and grid dimensions are cropped to the largest multiples of
  ``patch_size`` (and ``2 * patch_size`` when using the projector, since the
  2×2 pixel-shuffle requires even grids).
* Images in the same call may have different ``(H, W)``; there is no
  padding and no cross-image attention leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # FlashAttention is optional; we fall back to SDPA transparently.
    from flash_attn import flash_attn_varlen_func  # type: ignore

    _HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover - environment-dependent
    flash_attn_varlen_func = None  # type: ignore
    _HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MoonViTConfig:
    """Configuration for :class:`MoonViT`.

    Defaults reproduce SigLIP-SO-400M (``google/siglip-so400m-patch14-384``),
    the checkpoint MoonViT is initialized from in the Kimi-VL paper.
    """

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    patch_size: int = 14
    num_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    # Side length of the square grid used by the original SigLIP absolute
    # positional embedding. For SO-400M @ 384 with patch-14, this is
    # 384 // 14 == 27 (i.e. 729 positions).
    abs_pos_embed_grid_size: int = 27

    # 2-D RoPE base frequency.
    rope_theta: float = 10000.0

    # Upper bound used as ``max_seqlen`` by the flash-attn varlen kernel.
    # 3.2 M pixels / 14^2 ≈ 16327; round up.
    max_num_patches: int = 16384


# ---------------------------------------------------------------------------
# Patch embedding & positional encodings
# ---------------------------------------------------------------------------


class MoonViTPatchEmbed(nn.Module):
    """Strided Conv2d patch embedding (equivalent to a linear on flattened patches).

    Input is a single image at its native resolution — shape ``(1, C, H, W)``
    or ``(C, H, W)``. Spatial dims are cropped to the largest multiples of
    ``patch_size``.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, int, int]:
        """Return ``(patches, grid_h, grid_w)`` where ``patches`` is ``(N, D)``."""
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() != 4 or pixel_values.size(0) != 1:
            raise ValueError(
                f"MoonViTPatchEmbed expects a single image (1, C, H, W); "
                f"got shape {tuple(pixel_values.shape)}"
            )
        ps = self.patch_size
        _, _, h, w = pixel_values.shape
        gh, gw = h // ps, w // ps
        if gh == 0 or gw == 0:
            raise ValueError(
                f"Image is smaller than the patch size ({ps}); got H={h}, W={w}"
            )
        pixel_values = pixel_values[:, :, : gh * ps, : gw * ps]
        x = self.proj(pixel_values)  # (1, D, gh, gw)
        x = x.flatten(2).transpose(1, 2).squeeze(0).contiguous()  # (gh*gw, D)
        return x, gh, gw


class AbsolutePosEmbedInterpolator(nn.Module):
    """SigLIP learnable absolute positional embedding, bicubically interpolated
    per-image to match each image's native ``(grid_h, grid_w)`` grid.

    The raw parameter is shaped ``(grid_size * grid_size, hidden_size)`` so it
    loads directly from a SigLIP checkpoint (which stores positions as a single
    flat table).
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.grid_size = config.abs_pos_embed_grid_size
        self.hidden_size = config.hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(self.grid_size * self.grid_size, self.hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=config.initializer_range)

    def forward(self, grid_h: int, grid_w: int) -> Tensor:
        """Return interpolated positional embedding of shape ``(grid_h * grid_w, D)``."""
        g = self.grid_size
        if grid_h == g and grid_w == g:
            return self.pos_embed
        # Reshape the flat table to a 2-D grid, interpolate, flatten back.
        pe = self.pos_embed.reshape(g, g, self.hidden_size)  # (g, g, D)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1, D, g, g)
        pe = F.interpolate(
            pe,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )
        pe = pe.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, self.hidden_size)
        return pe


class RotaryEmbedding2D(nn.Module):
    """2-D rotary positional embedding split across the head dimension.

    The first half of ``head_dim`` is rotated with the H-axis position, and
    the second half with the W-axis position. Within each half we use the
    standard Llama-style "duplicated-halves + rotate_half" form.

    ``build_cos_sin`` constructs packed ``cos`` / ``sin`` tensors aligned with
    a packed NaViT sequence — one ``(gh * gw, head_dim)`` block per image,
    concatenated along the sequence dimension.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2-D RoPE; got {head_dim}"
            )
        self.head_dim = head_dim
        self.axis_dim = head_dim // 2  # dims per spatial axis
        self.theta = theta
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_cos_sin(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        freqs = positions[:, None].to(self.inv_freq.dtype) * self.inv_freq[None, :]
        cos = freqs.cos()
        sin = freqs.sin()
        # Duplicate halves so "rotate_half" rotation matches standard Llama RoPE.
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin

    def build_cos_sin(
        self,
        grid_shapes: Sequence[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Build ``(L_total, head_dim)`` cos/sin packed across the given image grids."""
        cos_chunks: List[Tensor] = []
        sin_chunks: List[Tensor] = []
        for gh, gw in grid_shapes:
            h_pos = torch.arange(gh, device=device)
            w_pos = torch.arange(gw, device=device)
            h_cos, h_sin = self._axis_cos_sin(h_pos)  # (gh, axis_dim)
            w_cos, w_sin = self._axis_cos_sin(w_pos)  # (gw, axis_dim)

            # Broadcast each 1-D axis over the 2-D grid.
            h_cos = h_cos[:, None, :].expand(gh, gw, self.axis_dim)
            h_sin = h_sin[:, None, :].expand(gh, gw, self.axis_dim)
            w_cos = w_cos[None, :, :].expand(gh, gw, self.axis_dim)
            w_sin = w_sin[None, :, :].expand(gh, gw, self.axis_dim)

            cos = torch.cat([h_cos, w_cos], dim=-1).reshape(gh * gw, self.head_dim)
            sin = torch.cat([h_sin, w_sin], dim=-1).reshape(gh * gw, self.head_dim)
            cos_chunks.append(cos)
            sin_chunks.append(sin)

        cos_all = torch.cat(cos_chunks, dim=0).to(dtype=dtype)
        sin_all = torch.cat(sin_chunks, dim=0).to(dtype=dtype)
        return cos_all, sin_all


def _rotate_half_axis(x: Tensor) -> Tensor:
    """Llama-style half-rotation used within a single spatial-axis slice."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rotary_2d(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """Apply 2-D RoPE to ``q`` and ``k``.

    Parameters
    ----------
    q, k:
        Shape ``(L, num_heads, head_dim)``.
    cos, sin:
        Shape ``(L, head_dim)``. The first ``head_dim // 2`` entries are the
        H-axis rotation; the last ``head_dim // 2`` entries are the W-axis
        rotation (see :class:`RotaryEmbedding2D`).
    """
    hd = q.shape[-1]
    ad = hd // 2  # per-axis slice size

    cos = cos.unsqueeze(1)  # (L, 1, head_dim)
    sin = sin.unsqueeze(1)

    cos_h, cos_w = cos[..., :ad], cos[..., ad:]
    sin_h, sin_w = sin[..., :ad], sin[..., ad:]

    def _rot(x: Tensor) -> Tensor:
        xh, xw = x[..., :ad], x[..., ad:]
        xh_rot = xh * cos_h + _rotate_half_axis(xh) * sin_h
        xw_rot = xw * cos_w + _rotate_half_axis(xw) * sin_w
        return torch.cat([xh_rot, xw_rot], dim=-1)

    return _rot(q), _rot(k)


# ---------------------------------------------------------------------------
# Attention, MLP, encoder layer
# ---------------------------------------------------------------------------


def _varlen_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    dropout_p: float,
) -> Tensor:
    """Per-image SDPA fallback for variable-length attention.

    Equivalent to calling ``scaled_dot_product_attention`` on each image's
    slice in isolation — which is exactly the block-diagonal mask that
    ``flash_attn_varlen_func`` would realize.
    """
    outputs: List[Tensor] = []
    for i in range(cu_seqlens.numel() - 1):
        s = int(cu_seqlens[i].item())
        e = int(cu_seqlens[i + 1].item())
        if e == s:
            continue
        # (seq, heads, head_dim) -> (1, heads, seq, head_dim)
        qi = q[s:e].transpose(0, 1).unsqueeze(0)
        ki = k[s:e].transpose(0, 1).unsqueeze(0)
        vi = v[s:e].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, dropout_p=dropout_p, scale=scale
        )
        oi = oi.squeeze(0).transpose(0, 1).contiguous()  # (seq, heads, head_dim)
        outputs.append(oi)
    return torch.cat(outputs, dim=0)


class MoonViTAttention(nn.Module):
    """Multi-head self-attention with 2-D RoPE and NaViT-packed var-length masking."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.attn_dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        l_total, dim = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(l_total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each (L, H, D_h)

        q, k = apply_rotary_2d(q, k, rope_cos, rope_sin)

        dropout_p = self.attn_dropout if self.training else 0.0

        if _HAS_FLASH_ATTN and q.is_cuda:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=int(max_seqlen),
                max_seqlen_k=int(max_seqlen),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
                causal=False,
            )
        else:
            out = _varlen_sdpa(q, k, v, cu_seqlens, self.scale, dropout_p)

        out = out.reshape(l_total, dim)
        return self.proj(out)


def _get_activation(name: str) -> nn.Module:
    if name == "gelu_pytorch_tanh":
        return nn.GELU(approximate="tanh")
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class MoonViTMLP(nn.Module):
    """SigLIP-style two-layer MLP (no gating)."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.act = _get_activation(config.hidden_act)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoonViTEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (SigLIP layout)."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = MoonViTAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MoonViTMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens,
            max_seqlen,
            rope_cos,
            rope_sin,
        )
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# MoonViT
# ---------------------------------------------------------------------------


@dataclass
class MoonViTOutput:
    """Output bundle from :class:`MoonViT`.

    Attributes
    ----------
    last_hidden_state:
        Packed patch features, shape ``(L_total, hidden_size)``.
    cu_seqlens:
        Cumulative sequence lengths, shape ``(batch_size + 1,)``, dtype int32.
        Entry ``i`` marks the start of image ``i`` in ``last_hidden_state``.
    grid_shapes:
        Per-image ``(grid_h, grid_w)``, one tuple per input image.
    """

    last_hidden_state: Tensor
    cu_seqlens: Tensor
    grid_shapes: List[Tuple[int, int]] = field(default_factory=list)


class MoonViT(nn.Module):
    """Native-resolution vision encoder (Kimi-VL, §2.1).

    ``forward`` accepts a list of images at arbitrary resolutions and returns
    a packed sequence of patch tokens (NaViT-style). Positions are encoded
    additively by (a) interpolated SigLIP absolute pos embed and (b) 2-D RoPE
    inside attention. Attention is variable-length, so tokens never cross
    image boundaries.
    """

    def __init__(self, config: Optional[MoonViTConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else MoonViTConfig()
        cfg = self.config

        head_dim = cfg.hidden_size // cfg.num_attention_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "MoonViT requires head_dim divisible by 4 (2-D RoPE splits the "
                f"head dim in half per axis); got head_dim={head_dim}"
            )

        self.patch_embed = MoonViTPatchEmbed(cfg)
        self.pos_embed = AbsolutePosEmbedInterpolator(cfg)
        self.rope_2d = RotaryEmbedding2D(head_dim, theta=cfg.rope_theta)

        self.layers = nn.ModuleList(
            [MoonViTEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.apply(self._init_weights)

    # -- initialization --------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        ir = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=ir)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=ir)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # -- forward ---------------------------------------------------------

    def forward(self, pixel_values_list: Sequence[Tensor]) -> MoonViTOutput:
        """Encode a list of native-resolution images.

        Parameters
        ----------
        pixel_values_list:
            Sequence of image tensors. Each is ``(C, H, W)`` or ``(1, C, H, W)``;
            ``H`` and ``W`` may differ across entries. Spatial dims are cropped
            to the largest multiples of ``patch_size``.
        """
        if len(pixel_values_list) == 0:
            raise ValueError("pixel_values_list must contain at least one image")

        patch_chunks: List[Tensor] = []
        grid_shapes: List[Tuple[int, int]] = []
        seqlens: List[int] = []

        for img in pixel_values_list:
            patches, gh, gw = self.patch_embed(img)  # (N, D)
            pos = self.pos_embed(gh, gw).to(dtype=patches.dtype, device=patches.device)
            patches = patches + pos
            patch_chunks.append(patches)
            grid_shapes.append((gh, gw))
            seqlens.append(gh * gw)

        hidden_states = torch.cat(patch_chunks, dim=0)  # (L_total, D)
        device = hidden_states.device

        seqlens_t = torch.tensor(seqlens, dtype=torch.int32, device=device)
        cu_seqlens = F.pad(torch.cumsum(seqlens_t, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = max(seqlens)

        rope_cos, rope_sin = self.rope_2d.build_cos_sin(
            grid_shapes, device=device, dtype=hidden_states.dtype
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens, max_seqlen, rope_cos, rope_sin
            )
        hidden_states = self.post_layernorm(hidden_states)

        return MoonViTOutput(
            last_hidden_state=hidden_states,
            cu_seqlens=cu_seqlens,
            grid_shapes=grid_shapes,
        )


# ---------------------------------------------------------------------------
# Kimi-VL MLP projector (pixel shuffle + 2-layer MLP)
# ---------------------------------------------------------------------------


class PixelShuffle2x(nn.Module):
    """2×2 space-to-depth: halve spatial grid, 4× channels.

    Operates over a *packed* sequence: a flat ``(N, D)`` block with an
    associated ``(grid_h, grid_w)`` shape becomes ``(N / 4, 4 * D)`` with
    shape ``(grid_h / 2, grid_w / 2)``. Both grid dims must be even.
    """

    def forward(self, x: Tensor, grid_h: int, grid_w: int) -> Tuple[Tensor, int, int]:
        if grid_h % 2 != 0 or grid_w % 2 != 0:
            raise ValueError(
                f"PixelShuffle2x needs an even grid; got ({grid_h}, {grid_w})"
            )
        n, d = x.shape
        if n != grid_h * grid_w:
            raise ValueError(
                f"Token count {n} does not match grid {grid_h}x{grid_w}={grid_h * grid_w}"
            )
        # (gh, gw, d) -> (gh/2, 2, gw/2, 2, d) -> (gh/2, gw/2, 2, 2, d)
        x = x.view(grid_h, grid_w, d)
        x = x.view(grid_h // 2, 2, grid_w // 2, 2, d)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view((grid_h // 2) * (grid_w // 2), 4 * d)
        return x, grid_h // 2, grid_w // 2


class MLPProjector(nn.Module):
    """Kimi-VL vision-to-LLM projector (§2.1).

    1. 2×2 pixel-shuffle compression of MoonViT patch features.
    2. Two-layer MLP into the LLM hidden size.

    Applied per-image on the packed sequence so that ``cu_seqlens`` and
    per-image grid shapes are recomputed for the downstream LLM.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "gelu_pytorch_tanh",
    ) -> None:
        super().__init__()
        shuffled_dim = 4 * vision_hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else llm_hidden_size
        )
        self.shuffle = PixelShuffle2x()
        self.fc1 = nn.Linear(shuffled_dim, self.intermediate_size, bias=True)
        self.act = _get_activation(hidden_act)
        self.fc2 = nn.Linear(self.intermediate_size, llm_hidden_size, bias=True)

    def forward(
        self,
        vision_features: Tensor,
        grid_shapes: Sequence[Tuple[int, int]],
        cu_seqlens: Tensor,
    ) -> Tuple[Tensor, List[Tuple[int, int]], Tensor]:
        """Project packed vision features to LLM-space tokens.

        Parameters
        ----------
        vision_features:
            ``(L_total, vision_hidden_size)`` from :class:`MoonViT`.
        grid_shapes:
            Per-image ``(grid_h, grid_w)``. Each grid dim must be even.
        cu_seqlens:
            Cumulative sequence lengths from :class:`MoonViT`.

        Returns
        -------
        features:
            ``(L_total / 4, llm_hidden_size)``, packed.
        new_grid_shapes:
            Per-image ``(grid_h / 2, grid_w / 2)``.
        new_cu_seqlens:
            Updated cumulative sequence lengths for the compressed tokens.
        """
        chunks: List[Tensor] = []
        new_grids: List[Tuple[int, int]] = []
        new_seqlens: List[int] = []
        for i, (gh, gw) in enumerate(grid_shapes):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            block = vision_features[s:e]
            shuffled, nh, nw = self.shuffle(block, gh, gw)
            chunks.append(shuffled)
            new_grids.append((nh, nw))
            new_seqlens.append(nh * nw)

        x = torch.cat(chunks, dim=0)
        x = self.fc2(self.act(self.fc1(x)))

        seqlens_t = torch.tensor(new_seqlens, dtype=torch.int32, device=x.device)
        new_cu_seqlens = F.pad(torch.cumsum(seqlens_t, dim=0), (1, 0)).to(torch.int32)
        return x, new_grids, new_cu_seqlens


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Tiny end-to-end check: two mixed-size fake images through MoonViT and the projector."""
    # Use a tiny config so CPU smoke tests stay fast.
    cfg = MoonViTConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=14,
        abs_pos_embed_grid_size=8,
    )
    encoder = MoonViT(cfg).eval()
    projector = MLPProjector(
        vision_hidden_size=cfg.hidden_size,
        llm_hidden_size=96,
        intermediate_size=128,
    ).eval()

    # Grids: (224/14, 280/14) = (16, 20)  -> 320 tokens
    #        (140/14, 196/14) = (10, 14)  -> 140 tokens
    imgs = [
        torch.randn(3, 224, 280),
        torch.randn(3, 140, 196),
    ]

    with torch.no_grad():
        out = encoder(imgs)
        assert out.last_hidden_state.shape == (320 + 140, cfg.hidden_size)
        assert out.cu_seqlens.tolist() == [0, 320, 460]
        assert out.grid_shapes == [(16, 20), (10, 14)]

        proj_x, proj_grids, proj_cu = projector(
            out.last_hidden_state, out.grid_shapes, out.cu_seqlens
        )
        # 2x2 pixel shuffle: tokens /= 4; each grid dim halved.
        assert proj_x.shape == ((320 + 140) // 4, 96)
        assert proj_grids == [(8, 10), (5, 7)]
        assert proj_cu.tolist() == [0, 80, 80 + 35]

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[smoke] encoder parameters: {n_params:,}")
    print(f"[smoke] encoder output: {tuple(out.last_hidden_state.shape)}")
    print(f"[smoke] projector output: {tuple(proj_x.shape)}")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
