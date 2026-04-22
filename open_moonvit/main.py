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

    Attributes
    ----------
    hidden_size : int
        Dimensionality of the patch token embeddings and all hidden states
        throughout the Transformer. Default 1152 (SigLIP-SO-400M).
    intermediate_size : int
        Width of the two-layer MLP inside each encoder layer. Default 4304.
    num_hidden_layers : int
        Number of stacked :class:`MoonViTEncoderLayer` blocks. Default 27.
    num_attention_heads : int
        Number of attention heads. ``hidden_size`` must be divisible by this
        value. Default 16 → ``head_dim = 72``.
    patch_size : int
        Side length (in pixels) of each square patch. The Conv2d tokenizer
        uses this as both kernel size and stride, so patches are
        non-overlapping. Default 14.
    num_channels : int
        Number of input image channels. Default 3 (RGB).
    hidden_act : str
        Activation function used in :class:`MoonViTMLP`. Passed to
        :func:`_get_activation`. Default ``"gelu_pytorch_tanh"``.
    layer_norm_eps : float
        Epsilon for all ``nn.LayerNorm`` layers. Default ``1e-6``.
    attention_dropout : float
        Dropout probability applied to attention weights during training.
        Default ``0.0``.
    initializer_range : float
        Standard deviation of the truncated-normal distribution used to
        initialize ``nn.Linear`` and ``nn.Conv2d`` weight matrices.
        Default ``0.02``.
    abs_pos_embed_grid_size : int
        Side length of the square grid used by the original SigLIP absolute
        positional embedding. For SO-400M @ 384px with patch-14 this is
        ``384 // 14 == 27`` (729 total positions). Default 27.
    rope_theta : float
        Base frequency for 2-D Rotary Positional Encoding. Default ``10000.0``.
    max_num_patches : int
        Upper bound on the number of patches in a single image, used as the
        ``max_seqlen`` argument to the FlashAttention varlen kernel. Sized
        for 3.2 M pixels at patch-14 (≈ 16 327 patches). Default 16 384.
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
        """Initialize the Conv2d patch projection layer.

        Parameters
        ----------
        config : MoonViTConfig
            Model configuration. Uses ``config.num_channels``,
            ``config.hidden_size``, and ``config.patch_size`` to construct
            the projection kernel. Both kernel size and stride are set to
            ``patch_size`` so the convolution is equivalent to a per-patch
            linear projection with no overlap between patches.
        """
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
        """Embed a single image into a flat sequence of patch tokens.

        Accepts a 3-D ``(C, H, W)`` or 4-D ``(1, C, H, W)`` image tensor.
        Spatial dimensions are silently cropped to ``(gh * patch_size,
        gw * patch_size)`` before the projection, discarding any
        incomplete border pixels.

        Parameters
        ----------
        pixel_values : Tensor
            Image to embed. Shape ``(C, H, W)`` or ``(1, C, H, W)``. The
            batch dimension, if present, must be exactly 1.

        Returns
        -------
        patches : Tensor
            Shape ``(grid_h * grid_w, hidden_size)``. Contiguous float
            tensor on the same device as ``pixel_values``.
        grid_h : int
            Number of patch rows after cropping (``H // patch_size``).
        grid_w : int
            Number of patch columns after cropping (``W // patch_size``).

        Raises
        ------
        ValueError
            If the input is not 3-D or 4-D, if the batch dimension is not 1,
            or if either spatial dimension is smaller than ``patch_size``
            (which would produce an empty grid).
        """
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
        """Initialize the learnable positional embedding parameter.

        The embedding table is stored flat as ``(grid_size², hidden_size)``
        to match the SigLIP checkpoint layout exactly, enabling direct weight
        loading without any reshaping. Values are initialized with a truncated
        normal distribution and subsequently updated by fine-tuning.

        Parameters
        ----------
        config : MoonViTConfig
            Uses ``config.abs_pos_embed_grid_size`` (side length of the square
            pretraining grid; 27 for SigLIP-SO-400M at 384 px / patch-14),
            ``config.hidden_size``, and ``config.initializer_range`` (std for
            the truncated-normal init).
        """
        super().__init__()
        self.grid_size = config.abs_pos_embed_grid_size
        self.hidden_size = config.hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(self.grid_size * self.grid_size, self.hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=config.initializer_range)

    def forward(self, grid_h: int, grid_w: int) -> Tensor:
        """Return the positional embedding resized to the requested grid.

        When ``grid_h == grid_w == grid_size`` the raw parameter table is
        returned directly (no copy, no interpolation). For any other grid the
        flat parameter is reshaped to ``(grid_size, grid_size, D)``, treated
        as a spatial feature map, and bicubically interpolated to
        ``(grid_h, grid_w)`` before being re-flattened. Bicubic mode with
        ``antialias=False`` matches the SigLIP / ViT interpolation convention.

        Parameters
        ----------
        grid_h : int
            Target number of patch rows for the current image.
        grid_w : int
            Target number of patch columns for the current image.

        Returns
        -------
        Tensor
            Shape ``(grid_h * grid_w, hidden_size)``, on the same device and
            dtype as the underlying ``pos_embed`` parameter.
        """
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
        """Initialize inverse-frequency buffers for H and W axis rotations.

        The head dimension is split into two equal halves — one per spatial
        axis. Within each half the standard Llama-style rotate-half RoPE is
        applied, requiring the half-dim to itself be even (hence the divisible-
        by-4 constraint on ``head_dim``). The inverse-frequency vector
        ``inv_freq`` has shape ``(axis_dim // 2,)`` and is shared between both
        spatial axes; only the position indices differ.

        Parameters
        ----------
        head_dim : int
            Attention head dimension. Must be divisible by 4: two spatial
            axes each use half the head dimension, and the rotate-half
            formulation internally splits that half in two again.
        theta : float, optional
            RoPE base frequency. Larger values slow the rotation rate and
            extend the effective position range. Default ``10000.0``.

        Raises
        ------
        ValueError
            If ``head_dim`` is not divisible by 4.
        """
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
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_cos_sin(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute cos/sin rotation factors for a 1-D sequence of integer positions.

        Uses the duplicated-halves convention so that output can be applied
        with the standard ``_rotate_half_axis`` function: the first
        ``axis_dim // 2`` entries and the second ``axis_dim // 2`` entries of
        the output tensors are identical before being multiplied with the
        query/key vectors.

        Parameters
        ----------
        positions : Tensor
            1-D integer tensor of position indices, e.g.
            ``torch.arange(grid_h)``. Placed on the same device as
            ``self.inv_freq`` inside the outer loop.

        Returns
        -------
        cos : Tensor
            Shape ``(len(positions), axis_dim)``. Each row holds the cosine
            rotation factors for that position, with both halves duplicated.
        sin : Tensor
            Shape ``(len(positions), axis_dim)``. Same structure as ``cos``.
        """
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
        """Build packed cos/sin rotation tensors for an entire NaViT-style sequence.

        For each image grid ``(gh, gw)`` the H-axis and W-axis cos/sin factors
        are independently computed, broadcast over the full 2-D patch grid,
        then concatenated along the head-dim axis to form a
        ``(gh * gw, head_dim)`` block. All such blocks are concatenated along
        the sequence axis to produce tensors that align 1-to-1 with the packed
        sequence that :class:`MoonViTAttention` operates on.

        The first ``head_dim // 2`` entries of every row encode the H-axis
        rotation; the last ``head_dim // 2`` encode the W-axis rotation.
        :func:`apply_rotary_2d` expects exactly this layout.

        Parameters
        ----------
        grid_shapes : Sequence[Tuple[int, int]]
            Per-image ``(grid_h, grid_w)`` for every image in the current
            forward pass, in the same order as the packed sequence.
        device : torch.device
            Device on which to allocate the integer position-index tensors.
            Should match the device of the hidden states.
        dtype : torch.dtype
            Output dtype. Internal computation is done in float32 for
            numerical accuracy and cast at the end.

        Returns
        -------
        cos : Tensor
            Shape ``(L_total, head_dim)`` where
            ``L_total = sum(gh * gw for gh, gw in grid_shapes)``.
        sin : Tensor
            Same shape and device as ``cos``.
        """
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
    """Apply the Llama-style rotate-half operation to a single spatial-axis slice.

    Splits the last dimension of ``x`` into two equal halves ``[x1, x2]`` and
    returns ``[-x2, x1]``. This realizes the 2-D orthogonal rotation
    ``[cos θ, -sin θ; sin θ, cos θ]`` when combined with the standard
    ``x * cos + rotate_half(x) * sin`` formula, without materializing a full
    rotation matrix.

    Parameters
    ----------
    x : Tensor
        Arbitrary shape; the last dimension must be even.

    Returns
    -------
    Tensor
        Same shape as ``x``, with the second half negated and moved to the
        front of the last dimension.
    """
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rotary_2d(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """Apply 2-D RoPE to query and key tensors in a NaViT-packed sequence.

    The head dimension is treated as two independent halves, one per spatial
    axis. Each half is rotated by the corresponding axis position using the
    standard rotate-half formulation. Because the rotation is applied
    element-wise per token the operation is compatible with the packed
    (variable-length) sequence layout — no padding or masking is needed.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape ``(L_total, num_heads, head_dim)``.
    k : Tensor
        Key tensor of shape ``(L_total, num_heads, head_dim)``.
    cos : Tensor
        Shape ``(L_total, head_dim)``. The first ``head_dim // 2`` entries
        hold the H-axis cosines; the last ``head_dim // 2`` hold the W-axis
        cosines. Produced by :meth:`RotaryEmbedding2D.build_cos_sin`.
    sin : Tensor
        Shape ``(L_total, head_dim)``. Same layout as ``cos``.

    Returns
    -------
    q_rotated : Tensor
        Shape ``(L_total, num_heads, head_dim)``.
    k_rotated : Tensor
        Shape ``(L_total, num_heads, head_dim)``.
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
    """Per-image SDPA fallback for variable-length attention without FlashAttention.

    Iterates over images defined by ``cu_seqlens`` and calls
    ``F.scaled_dot_product_attention`` on each image's token slice in
    isolation. This is mathematically equivalent to applying a block-diagonal
    attention mask (one block per image) over the full packed sequence — which
    is exactly the attention pattern that ``flash_attn_varlen_func`` realizes
    more efficiently on CUDA.

    Used automatically when FlashAttention is not installed or the tensors are
    not on CUDA (e.g. CPU inference, MPS).

    Parameters
    ----------
    q : Tensor
        Shape ``(L_total, num_heads, head_dim)``.
    k : Tensor
        Shape ``(L_total, num_heads, head_dim)``.
    v : Tensor
        Shape ``(L_total, num_heads, head_dim)``.
    cu_seqlens : Tensor
        Shape ``(num_images + 1,)``, dtype int32. Cumulative token counts;
        image ``i`` occupies rows ``cu_seqlens[i]:cu_seqlens[i+1]``.
    scale : float
        Softmax temperature, typically ``head_dim ** -0.5``.
    dropout_p : float
        Attention dropout probability (non-zero only during training).

    Returns
    -------
    Tensor
        Shape ``(L_total, num_heads, head_dim)``. Concatenation of per-image
        attention outputs in the original packed order.
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
        """Initialize multi-head attention projection weights.

        Queries, keys, and values are computed with a single fused linear
        layer ``qkv`` of output size ``3 * hidden_size``, then split after
        the projection. A separate output projection ``proj`` maps the
        concatenated head outputs back to ``hidden_size``.

        Parameters
        ----------
        config : MoonViTConfig
            Uses ``config.hidden_size``, ``config.num_attention_heads``, and
            ``config.attention_dropout``.

        Raises
        ------
        ValueError
            If ``hidden_size`` is not evenly divisible by
            ``num_attention_heads``.
        """
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
        """Compute multi-head self-attention over a NaViT-packed token sequence.

        Projects hidden states to Q/K/V, applies 2-D RoPE to Q and K, then
        runs variable-length attention. Uses ``flash_attn_varlen_func`` when
        FlashAttention is installed and the input is on CUDA; falls back to
        :func:`_varlen_sdpa` otherwise. In both cases the block-diagonal
        attention structure implied by ``cu_seqlens`` prevents any token from
        attending to tokens belonging to a different image in the packed batch.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(L_total, hidden_size)``. The packed sequence of all
            images concatenated along the sequence dimension.
        cu_seqlens : Tensor
            Shape ``(num_images + 1,)``, dtype int32. Cumulative token counts;
            image ``i`` spans rows ``cu_seqlens[i]:cu_seqlens[i+1]``.
        max_seqlen : int
            Length of the longest individual image sequence in the batch.
            Required as an upper bound by the FlashAttention kernel.
        rope_cos : Tensor
            Shape ``(L_total, head_dim)``. Precomputed from
            :meth:`RotaryEmbedding2D.build_cos_sin`.
        rope_sin : Tensor
            Shape ``(L_total, head_dim)``. Same layout as ``rope_cos``.

        Returns
        -------
        Tensor
            Shape ``(L_total, hidden_size)``. Attention output after the
            output projection.
        """
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
    """Instantiate and return an activation function module by string name.

    Supported names
    ---------------
    ``"gelu_pytorch_tanh"``
        ``nn.GELU(approximate="tanh")`` — fast tanh-approximated GELU that
        matches JAX/SigLIP behaviour. Default throughout MoonViT.
    ``"gelu"``
        ``nn.GELU()`` — exact GELU computed via the error function.
    ``"relu"``
        ``nn.ReLU()``.
    ``"silu"``
        ``nn.SiLU()`` (also known as swish: ``x * sigmoid(x)``).

    Parameters
    ----------
    name : str
        One of the supported activation names listed above.

    Returns
    -------
    nn.Module
        The corresponding activation module, freshly instantiated.

    Raises
    ------
    ValueError
        If ``name`` does not match any supported activation.
    """
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
        """Initialize the two-layer feed-forward network.

        Architecture: ``hidden_size → intermediate_size → hidden_size`` with
        biases on both projections. Unlike SwiGLU or GeGLU variants used in
        many LLMs, there is no gating — the activation is applied directly to
        the output of ``fc1``.

        Parameters
        ----------
        config : MoonViTConfig
            Uses ``config.hidden_size``, ``config.intermediate_size``, and
            ``config.hidden_act`` (passed to :func:`_get_activation`).
        """
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.act = _get_activation(config.hidden_act)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Run the feed-forward network.

        Parameters
        ----------
        x : Tensor
            Shape ``(..., hidden_size)``. Typically ``(L_total, hidden_size)``
            when called from :class:`MoonViTEncoderLayer`.

        Returns
        -------
        Tensor
            Same shape as ``x``; result of ``fc2(act(fc1(x)))``.
        """
        return self.fc2(self.act(self.fc1(x)))


class MoonViTEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (SigLIP layout)."""

    def __init__(self, config: MoonViTConfig) -> None:
        """Initialize a single pre-norm Transformer encoder layer.

        Follows the SigLIP / ViT pre-norm layout:
        ``x = x + Attn(LayerNorm1(x))``
        ``x = x + MLP(LayerNorm2(x))``

        Parameters
        ----------
        config : MoonViTConfig
            Passed through to :class:`MoonViTAttention` and
            :class:`MoonViTMLP`. Also uses ``config.hidden_size`` and
            ``config.layer_norm_eps`` for the two ``nn.LayerNorm`` sub-modules.
        """
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
        """Apply one pre-norm encoder layer to the packed token sequence.

        Residual connections wrap both the attention and MLP sub-blocks:
        ``hidden_states = hidden_states + self_attn(layer_norm1(hidden_states))``
        ``hidden_states = hidden_states + mlp(layer_norm2(hidden_states))``

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(L_total, hidden_size)``. Packed sequence of all images.
        cu_seqlens : Tensor
            Shape ``(num_images + 1,)``, dtype int32. Forwarded unchanged to
            :class:`MoonViTAttention` to enforce per-image attention masking.
        max_seqlen : int
            Maximum per-image sequence length; forwarded to attention.
        rope_cos : Tensor
            Shape ``(L_total, head_dim)``; forwarded to attention.
        rope_sin : Tensor
            Same shape as ``rope_cos``; forwarded to attention.

        Returns
        -------
        Tensor
            Shape ``(L_total, hidden_size)``.
        """
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
        """Construct the full MoonViT encoder.

        Assembles the following sub-modules:

        * ``patch_embed`` — :class:`MoonViTPatchEmbed` (strided Conv2d
          tokenizer; one token per non-overlapping patch).
        * ``pos_embed`` — :class:`AbsolutePosEmbedInterpolator` (SigLIP
          learnable absolute positional embedding, bicubically interpolated
          per image to its native grid size).
        * ``rope_2d`` — :class:`RotaryEmbedding2D` (2-D RoPE applied inside
          every attention layer; not a stored embedding, just inv-freq buffers).
        * ``layers`` — ``nn.ModuleList`` of ``num_hidden_layers``
          :class:`MoonViTEncoderLayer` blocks.
        * ``post_layernorm`` — final ``nn.LayerNorm`` applied after the last
          encoder layer.

        All learnable parameters are initialized via :meth:`_init_weights`:
        truncated normal for ``nn.Linear`` / ``nn.Conv2d`` weights, zeros for
        biases, ones / zeros for ``nn.LayerNorm`` scale / bias.

        Parameters
        ----------
        config : MoonViTConfig, optional
            Model hyperparameters. Defaults to :class:`MoonViTConfig()`, which
            reproduces the SigLIP-SO-400M architecture described in the
            Kimi-VL paper.

        Raises
        ------
        ValueError
            If the derived ``head_dim = hidden_size // num_attention_heads``
            is not divisible by 4, which is required by 2-D RoPE (the head
            dim is split into two equal halves, each further split in the
            rotate-half formulation).
        """
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
        """Apply weight initialization rules to a single sub-module.

        Called recursively by ``self.apply(self._init_weights)`` during
        construction. Rules by module type:

        * ``nn.Linear`` — weights drawn from truncated normal
          ``N(0, initializer_range²)``; biases zeroed.
        * ``nn.Conv2d`` — same as ``nn.Linear``.
        * ``nn.LayerNorm`` — scale initialized to 1, bias to 0. These are
          the standard pre-activation-norm starting values and are updated
          during fine-tuning.

        All other module types are silently skipped.

        Parameters
        ----------
        module : nn.Module
            Sub-module to initialize. Typically called via ``self.apply``
            rather than directly.
        """
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
        """Encode a list of native-resolution images into packed patch tokens.

        Each image is independently patch-embedded and augmented with its
        interpolated absolute positional embedding. The resulting token
        sequences are concatenated into a single packed sequence; 2-D RoPE
        cos/sin tensors are built to match. All ``num_hidden_layers`` encoder
        layers are applied to the full packed sequence, followed by a final
        LayerNorm. The variable-length structure (``cu_seqlens``) ensures that
        attention never crosses image boundaries.

        Parameters
        ----------
        pixel_values_list : Sequence[Tensor]
            One or more image tensors. Each must be ``(C, H, W)`` or
            ``(1, C, H, W)``; ``H`` and ``W`` may differ across entries.
            Spatial dimensions are cropped to the largest multiples of
            ``patch_size`` before tokenization.

        Returns
        -------
        MoonViTOutput
            ``last_hidden_state`` — shape ``(L_total, hidden_size)``.
            ``cu_seqlens`` — shape ``(len(pixel_values_list) + 1,)``, int32.
            ``grid_shapes`` — list of ``(grid_h, grid_w)`` per image.

        Raises
        ------
        ValueError
            If ``pixel_values_list`` is empty.
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
        """Compress a packed 2-D token grid by folding 2×2 spatial blocks into channels.

        Each non-overlapping 2×2 neighbourhood of tokens is concatenated along
        the feature dimension, reducing the spatial token count by 4× while
        quadrupling the feature dimension. This is equivalent to
        ``nn.PixelUnshuffle(2)`` on a spatial feature map, but operates
        directly on the flat packed sequence used throughout MoonViT, avoiding
        any reshape to a 4-D tensor.

        Parameters
        ----------
        x : Tensor
            Shape ``(grid_h * grid_w, D)``. The flat token sequence for a
            single image.
        grid_h : int
            Number of patch rows. Must be even.
        grid_w : int
            Number of patch columns. Must be even.

        Returns
        -------
        x_shuffled : Tensor
            Shape ``((grid_h // 2) * (grid_w // 2), 4 * D)``.
        new_grid_h : int
            ``grid_h // 2``.
        new_grid_w : int
            ``grid_w // 2``.

        Raises
        ------
        ValueError
            If either ``grid_h`` or ``grid_w`` is odd, or if ``x.shape[0]``
            does not equal ``grid_h * grid_w``.
        """
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
        """Initialize the pixel-shuffle and two-layer MLP projection.

        After pixel-shuffle the token dimension is ``4 * vision_hidden_size``
        (each 2×2 spatial block is concatenated into one token). The MLP maps
        this expanded dimension to ``llm_hidden_size`` via an optional
        intermediate bottleneck / expansion at ``intermediate_size``.

        Parameters
        ----------
        vision_hidden_size : int
            Feature dimension of MoonViT patch tokens (e.g. 1152 for the
            SigLIP-SO-400M backbone used in Kimi-VL).
        llm_hidden_size : int
            Hidden dimension of the downstream language model that will consume
            the projected tokens.
        intermediate_size : int, optional
            Width of the hidden layer inside the two-layer MLP. Defaults to
            ``llm_hidden_size`` when not specified, giving a square projection.
        hidden_act : str, optional
            Activation function for the MLP, passed to :func:`_get_activation`.
            Default ``"gelu_pytorch_tanh"`` to match MoonViT's own MLP layers.
        """
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

        Iterates over images in the packed sequence, applies 2×2
        :class:`PixelShuffle2x` to each image's token block, concatenates the
        compressed blocks, then passes the full sequence through the shared
        two-layer MLP. New ``cu_seqlens`` and grid shapes are computed to
        reflect the 4× reduction in token count.

        Parameters
        ----------
        vision_features : Tensor
            ``(L_total, vision_hidden_size)`` packed output from
            :class:`MoonViT`.
        grid_shapes : Sequence[Tuple[int, int]]
            Per-image ``(grid_h, grid_w)`` from :attr:`MoonViTOutput.grid_shapes`.
            Each spatial dimension must be even (required by
            :class:`PixelShuffle2x`).
        cu_seqlens : Tensor
            Cumulative sequence lengths from :attr:`MoonViTOutput.cu_seqlens`,
            shape ``(num_images + 1,)``, dtype int32.

        Returns
        -------
        features : Tensor
            Shape ``(L_total / 4, llm_hidden_size)``. Packed LLM-space tokens.
        new_grid_shapes : List[Tuple[int, int]]
            Per-image ``(grid_h // 2, grid_w // 2)`` after pixel-shuffle.
        new_cu_seqlens : Tensor
            Updated cumulative sequence lengths for the compressed token
            sequence, shape ``(num_images + 1,)``, dtype int32.
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
