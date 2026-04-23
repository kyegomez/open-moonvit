"""Comprehensive pytest suite for MoonViT (``main.py``).

Covers every public module and the private helpers used in the forward path:

* ``MoonViTConfig`` defaults                     — SigLIP-SO-400M sanity check
* ``MoonViTPatchEmbed``                          — shapes, cropping, input validation
* ``AbsolutePosEmbedInterpolator``               — identity + bicubic upscale
* ``RotaryEmbedding2D`` + ``apply_rotary_2d``    — shapes, zero-position identity,
                                                   per-image == packed
* ``_rotate_half_axis``                          — double-rotation negation
* ``_varlen_sdpa``                               — matches per-image SDPA bit-wise
* ``MoonViTAttention``                           — shape + block-diagonal isolation
* ``MoonViTMLP`` / ``MoonViTEncoderLayer``       — shape preservation
* ``MoonViT``                                    — full forward, packed==per-image,
                                                   error cases, gradient flow
* ``PixelShuffle2x`` / ``MLPProjector``          — shape, information preservation,
                                                   new grid / cu_seqlens
* ``_get_activation``                            — all known names + unknown raises

Logs are routed through loguru.

Run:
    $ pytest -v test_main.py
    $ RUN_SLOW_TESTS=1 pytest -v test_main.py      # includes the ~413M-param build
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from open_moonvit.main import (
    AbsolutePosEmbedInterpolator,
    MLPProjector,
    MoonViT,
    MoonViTAttention,
    MoonViTConfig,
    MoonViTEncoderLayer,
    MoonViTMLP,
    MoonViTOutput,
    MoonViTPatchEmbed,
    PixelShuffle2x,
    RotaryEmbedding2D,
    _get_activation,
    _rotate_half_axis,
    _varlen_sdpa,
    apply_rotary_2d,
)

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("LOGURU_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}",
    colorize=True,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cfg() -> MoonViTConfig:
    """Small config for fast CPU tests."""
    cfg = MoonViTConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=14,
        abs_pos_embed_grid_size=8,
    )
    logger.info(
        f"tiny_cfg: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"heads={cfg.num_attention_heads} patch={cfg.patch_size}"
    )
    return cfg


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg: MoonViTConfig) -> MoonViT:
    torch.manual_seed(0)
    model = MoonViT(tiny_cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    logger.info(f"tiny_model built: {n:,} parameters")
    return model


def _mk_images(shapes: List[Tuple[int, int]], seed: int = 7) -> List[Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3, h, w, generator=g) for (h, w) in shapes]


# ---------------------------------------------------------------------------
# MoonViTConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_match_siglip_so400m(self) -> None:
        cfg = MoonViTConfig()
        assert cfg.hidden_size == 1152
        assert cfg.intermediate_size == 4304
        assert cfg.num_hidden_layers == 27
        assert cfg.num_attention_heads == 16
        assert cfg.patch_size == 14
        assert cfg.num_channels == 3
        assert cfg.hidden_act == "gelu_pytorch_tanh"
        assert cfg.layer_norm_eps == pytest.approx(1e-6)
        assert cfg.abs_pos_embed_grid_size == 27
        assert cfg.rope_theta == pytest.approx(10000.0)
        assert cfg.max_num_patches >= 16327, "must cover paper's 3.2M-pixel limit"

    def test_default_head_dim_div4(self) -> None:
        cfg = MoonViTConfig()
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        assert head_dim == 72
        assert head_dim % 4 == 0


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class TestPatchEmbed:
    def test_exact_multiple(self, tiny_cfg: MoonViTConfig) -> None:
        pe = MoonViTPatchEmbed(tiny_cfg)
        img = torch.randn(1, 3, 56, 84)  # 4x6 grid
        patches, gh, gw = pe(img)
        assert (gh, gw) == (4, 6)
        assert patches.shape == (24, tiny_cfg.hidden_size)

    def test_crops_to_patch_multiple(self, tiny_cfg: MoonViTConfig) -> None:
        pe = MoonViTPatchEmbed(tiny_cfg)
        img = torch.randn(1, 3, 60, 88)  # 60->56, 88->84
        patches, gh, gw = pe(img)
        assert (gh, gw) == (4, 6)
        assert patches.shape[0] == 24

    def test_accepts_3d_input(self, tiny_cfg: MoonViTConfig) -> None:
        pe = MoonViTPatchEmbed(tiny_cfg)
        patches, gh, gw = pe(torch.randn(3, 28, 42))
        assert (gh, gw) == (2, 3)
        assert patches.shape == (6, tiny_cfg.hidden_size)

    def test_rejects_tiny_images(self, tiny_cfg: MoonViTConfig) -> None:
        pe = MoonViTPatchEmbed(tiny_cfg)
        with pytest.raises(ValueError, match="smaller than the patch size"):
            pe(torch.randn(1, 3, 10, 10))

    def test_rejects_multi_batch(self, tiny_cfg: MoonViTConfig) -> None:
        pe = MoonViTPatchEmbed(tiny_cfg)
        with pytest.raises(ValueError, match="single image"):
            pe(torch.randn(2, 3, 56, 56))


# ---------------------------------------------------------------------------
# Absolute position embedding interpolator
# ---------------------------------------------------------------------------


class TestAbsPosEmbed:
    def test_identity_when_same_grid(self, tiny_cfg: MoonViTConfig) -> None:
        layer = AbsolutePosEmbedInterpolator(tiny_cfg)
        g = tiny_cfg.abs_pos_embed_grid_size
        out = layer(g, g)
        assert torch.equal(out, layer.pos_embed)

    def test_interpolation_shape(self, tiny_cfg: MoonViTConfig) -> None:
        layer = AbsolutePosEmbedInterpolator(tiny_cfg)
        out = layer(4, 10)
        assert out.shape == (40, tiny_cfg.hidden_size)

    def test_interpolation_upscale_shape(self, tiny_cfg: MoonViTConfig) -> None:
        layer = AbsolutePosEmbedInterpolator(tiny_cfg)
        out = layer(16, 20)
        assert out.shape == (320, tiny_cfg.hidden_size)


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------


class TestRotary2D:
    def test_requires_div4(self) -> None:
        with pytest.raises(ValueError, match="divisible by 4"):
            RotaryEmbedding2D(head_dim=6)

    def test_cos_sin_shapes(self) -> None:
        r = RotaryEmbedding2D(head_dim=16)
        cos, sin = r.build_cos_sin(
            [(3, 4), (2, 5)], device=torch.device("cpu"), dtype=torch.float32
        )
        assert cos.shape == (3 * 4 + 2 * 5, 16)
        assert sin.shape == (3 * 4 + 2 * 5, 16)

    def test_zero_position_identity(self) -> None:
        """At position (0, 0), cos==1 and sin==0 for all frequencies."""
        r = RotaryEmbedding2D(head_dim=16)
        cos, sin = r.build_cos_sin(
            [(1, 1)], device=torch.device("cpu"), dtype=torch.float32
        )
        assert torch.allclose(cos, torch.ones_like(cos))
        assert torch.allclose(sin, torch.zeros_like(sin))

    def test_concat_per_image_matches_packed(self) -> None:
        r = RotaryEmbedding2D(head_dim=12)
        grids = [(3, 5), (2, 4)]
        cos_a, sin_a = r.build_cos_sin([grids[0]], torch.device("cpu"), torch.float32)
        cos_b, sin_b = r.build_cos_sin([grids[1]], torch.device("cpu"), torch.float32)
        cos, sin = r.build_cos_sin(grids, torch.device("cpu"), torch.float32)
        assert torch.allclose(cos, torch.cat([cos_a, cos_b]))
        assert torch.allclose(sin, torch.cat([sin_a, sin_b]))

    def test_head_dim_split_structure(self) -> None:
        """First head_dim/2 is H-axis, second half is W-axis."""
        r = RotaryEmbedding2D(head_dim=16)
        ad = 8  # axis dim
        # Grid (2, 1): H varies, W is constant at 0 — so the W half must be all identity.
        cos, sin = r.build_cos_sin([(2, 1)], torch.device("cpu"), torch.float32)
        assert torch.allclose(cos[:, ad:], torch.ones(2, ad))
        assert torch.allclose(sin[:, ad:], torch.zeros(2, ad))
        # And at row 0 (H==0) the H half is also identity.
        assert torch.allclose(cos[0, :ad], torch.ones(ad))
        assert torch.allclose(sin[0, :ad], torch.zeros(ad))
        # But at row 1 (H==1) the H half should differ from identity.
        assert not torch.allclose(sin[1, :ad], torch.zeros(ad))


# ---------------------------------------------------------------------------
# apply_rotary_2d
# ---------------------------------------------------------------------------


class TestApplyRotary2D:
    def test_shape_preserved(self) -> None:
        L, H, D = 5, 4, 16
        q, k = torch.randn(L, H, D), torch.randn(L, H, D)
        cos, sin = torch.randn(L, D), torch.randn(L, D)
        q_out, k_out = apply_rotary_2d(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_identity_at_zero_position(self) -> None:
        L, H, D = 3, 4, 16
        q, k = torch.randn(L, H, D), torch.randn(L, H, D)
        cos = torch.ones(L, D)
        sin = torch.zeros(L, D)
        q_out, k_out = apply_rotary_2d(q, k, cos, sin)
        assert torch.allclose(q_out, q)
        assert torch.allclose(k_out, k)


# ---------------------------------------------------------------------------
# _rotate_half_axis
# ---------------------------------------------------------------------------


class TestRotateHalfAxis:
    def test_basic(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = _rotate_half_axis(x)
        # cat([-x[2:], x[:2]]) = [-3, -4, 1, 2]
        assert torch.equal(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    def test_double_rotation_negates(self) -> None:
        x = torch.randn(3, 8)
        assert torch.allclose(_rotate_half_axis(_rotate_half_axis(x)), -x)


# ---------------------------------------------------------------------------
# _varlen_sdpa
# ---------------------------------------------------------------------------


class TestVarlenSDPA:
    def test_matches_per_image_sdpa(self) -> None:
        torch.manual_seed(0)
        L, H, D = 12, 4, 8
        q, k, v = torch.randn(L, H, D), torch.randn(L, H, D), torch.randn(L, H, D)
        cu = torch.tensor([0, 5, 12], dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        out = _varlen_sdpa(q, k, v, cu, scale=scale, dropout_p=0.0)
        assert out.shape == (L, H, D)
        for s, e in [(0, 5), (5, 12)]:
            qi = q[s:e].transpose(0, 1).unsqueeze(0)
            ki = k[s:e].transpose(0, 1).unsqueeze(0)
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)
            oi = oi.squeeze(0).transpose(0, 1)
            assert torch.allclose(out[s:e], oi, atol=1e-5)


# ---------------------------------------------------------------------------
# MoonViTAttention
# ---------------------------------------------------------------------------


class TestAttention:
    def test_shape(self, tiny_cfg: MoonViTConfig) -> None:
        attn = MoonViTAttention(tiny_cfg).eval()
        L, D = 20, tiny_cfg.hidden_size
        hd = D // tiny_cfg.num_attention_heads
        x = torch.randn(L, D)
        cu = torch.tensor([0, 12, 20], dtype=torch.int32)
        cos, sin = torch.randn(L, hd), torch.randn(L, hd)
        out = attn(x, cu, 12, cos, sin)
        assert out.shape == (L, D)

    def test_block_diagonal_isolation(self, tiny_cfg: MoonViTConfig) -> None:
        attn = MoonViTAttention(tiny_cfg).eval()
        D = tiny_cfg.hidden_size
        hd = D // tiny_cfg.num_attention_heads
        L1, L2 = 12, 8
        L = L1 + L2
        torch.manual_seed(1)
        x = torch.randn(L, D)
        cos, sin = torch.randn(L, hd), torch.randn(L, hd)
        cu = torch.tensor([0, L1, L], dtype=torch.int32)
        with torch.no_grad():
            out_packed = attn(x, cu, max(L1, L2), cos, sin)
            out_a = attn(
                x[:L1],
                torch.tensor([0, L1], dtype=torch.int32),
                L1,
                cos[:L1],
                sin[:L1],
            )
            out_b = attn(
                x[L1:],
                torch.tensor([0, L2], dtype=torch.int32),
                L2,
                cos[L1:],
                sin[L1:],
            )
        assert torch.allclose(out_packed[:L1], out_a, atol=1e-5)
        assert torch.allclose(out_packed[L1:], out_b, atol=1e-5)


# ---------------------------------------------------------------------------
# MoonViTMLP / MoonViTEncoderLayer
# ---------------------------------------------------------------------------


class TestMLP:
    def test_shape(self, tiny_cfg: MoonViTConfig) -> None:
        mlp = MoonViTMLP(tiny_cfg)
        x = torch.randn(10, tiny_cfg.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape


class TestEncoderLayer:
    def test_shape_preserved(self, tiny_cfg: MoonViTConfig) -> None:
        layer = MoonViTEncoderLayer(tiny_cfg).eval()
        L, D = 20, tiny_cfg.hidden_size
        hd = D // tiny_cfg.num_attention_heads
        x = torch.randn(L, D)
        cu = torch.tensor([0, 12, 20], dtype=torch.int32)
        cos, sin = torch.randn(L, hd), torch.randn(L, hd)
        out = layer(x, cu, 12, cos, sin)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# MoonViT (end-to-end)
# ---------------------------------------------------------------------------


class TestMoonViT:
    def test_forward_shapes(self, tiny_model: MoonViT, tiny_cfg: MoonViTConfig) -> None:
        images = _mk_images([(56, 84), (28, 42)])  # grids 4x6=24, 2x3=6
        out = tiny_model(images)
        assert isinstance(out, MoonViTOutput)
        assert out.last_hidden_state.shape == (30, tiny_cfg.hidden_size)
        assert out.cu_seqlens.dtype == torch.int32
        assert out.cu_seqlens.tolist() == [0, 24, 30]
        assert out.grid_shapes == [(4, 6), (2, 3)]

    def test_packed_equals_per_image(self, tiny_model: MoonViT) -> None:
        a, b = _mk_images([(56, 84), (42, 70)])
        with torch.no_grad():
            packed = tiny_model([a, b]).last_hidden_state
            oa = tiny_model([a]).last_hidden_state
            ob = tiny_model([b]).last_hidden_state
        assert torch.allclose(packed[: oa.size(0)], oa, atol=1e-5)
        assert torch.allclose(packed[oa.size(0) :], ob, atol=1e-5)

    def test_three_images_packed(self, tiny_model: MoonViT) -> None:
        imgs = _mk_images([(56, 84), (28, 42), (70, 70)])
        out = tiny_model(imgs)
        expected_counts = [4 * 6, 2 * 3, 5 * 5]
        assert out.grid_shapes == [(4, 6), (2, 3), (5, 5)]
        assert out.last_hidden_state.shape[0] == sum(expected_counts)
        assert out.cu_seqlens.tolist() == [
            0,
            expected_counts[0],
            sum(expected_counts[:2]),
            sum(expected_counts),
        ]

    def test_crops_to_patch_multiple(self, tiny_model: MoonViT) -> None:
        img = torch.randn(3, 60, 88)  # crops to 56x84 -> 4x6
        out = tiny_model([img])
        assert out.grid_shapes == [(4, 6)]
        assert out.last_hidden_state.shape[0] == 24

    def test_single_image(self, tiny_model: MoonViT) -> None:
        (img,) = _mk_images([(28, 42)])
        out = tiny_model([img])
        assert out.cu_seqlens.tolist() == [0, 6]
        assert out.grid_shapes == [(2, 3)]

    def test_empty_list_raises(self, tiny_model: MoonViT) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            tiny_model([])

    def test_tiny_image_raises(self, tiny_model: MoonViT) -> None:
        with pytest.raises(ValueError, match="smaller than the patch size"):
            tiny_model([torch.randn(3, 10, 10)])

    def test_gradient_flow(self, tiny_cfg: MoonViTConfig) -> None:
        model = MoonViT(tiny_cfg)
        img = torch.randn(3, 56, 56)
        out = model([img])
        out.last_hidden_state.sum().backward()
        zero_grads = [
            name
            for name, p in model.named_parameters()
            if p.grad is None or p.grad.abs().sum().item() == 0.0
        ]
        assert len(zero_grads) == 0, f"params with no gradient: {zero_grads[:5]}..."

    def test_dtype_propagation_float32(self, tiny_cfg: MoonViTConfig) -> None:
        model = MoonViT(tiny_cfg).eval()
        img = torch.randn(3, 56, 56, dtype=torch.float32)
        out = model([img])
        assert out.last_hidden_state.dtype == torch.float32

    def test_deterministic_with_seed(self, tiny_cfg: MoonViTConfig) -> None:
        torch.manual_seed(42)
        m1 = MoonViT(tiny_cfg).eval()
        torch.manual_seed(42)
        m2 = MoonViT(tiny_cfg).eval()
        img = torch.randn(3, 56, 56)
        with torch.no_grad():
            o1 = m1([img]).last_hidden_state
            o2 = m2([img]).last_hidden_state
        assert torch.equal(o1, o2)

    @pytest.mark.skipif(
        not os.environ.get("RUN_SLOW_TESTS"),
        reason="Set RUN_SLOW_TESTS=1 to instantiate the default ~413M-param model.",
    )
    def test_default_config_param_count(self) -> None:
        cfg = MoonViTConfig()
        model = MoonViT(cfg)
        n = sum(p.numel() for p in model.parameters())
        logger.info(f"Default MoonViT params: {n / 1e6:.1f}M")
        # Paper: "400M native-resolution MoonViT". Our SigLIP-SO-400M build is ~413M.
        assert 400e6 < n < 430e6


# ---------------------------------------------------------------------------
# PixelShuffle2x
# ---------------------------------------------------------------------------


class TestPixelShuffle:
    def test_shape(self) -> None:
        s = PixelShuffle2x()
        x = torch.randn(24, 64)
        out, nh, nw = s(x, grid_h=4, grid_w=6)
        assert out.shape == (6, 256)
        assert (nh, nw) == (2, 3)

    def test_information_preserved_bijection(self) -> None:
        """A 2x2 block of tokens at grid (0,0) — i.e. indices 0, 1, 4, 5 in a
        4x4 grid — should appear concatenated along the channel dim of the
        output's (0, 0) block."""
        s = PixelShuffle2x()
        x = torch.arange(4 * 4 * 2, dtype=torch.float32).view(16, 2)
        out, nh, nw = s(x, 4, 4)
        assert (nh, nw) == (2, 2)
        expected_block00 = torch.cat([x[0], x[1], x[4], x[5]])
        assert torch.equal(out[0], expected_block00)
        expected_block01 = torch.cat([x[2], x[3], x[6], x[7]])
        assert torch.equal(out[1], expected_block01)

    def test_odd_grid_raises(self) -> None:
        s = PixelShuffle2x()
        x = torch.randn(15, 8)
        with pytest.raises(ValueError, match="even grid"):
            s(x, 3, 5)

    def test_mismatched_token_count_raises(self) -> None:
        s = PixelShuffle2x()
        x = torch.randn(10, 8)
        with pytest.raises(ValueError, match="Token count"):
            s(x, 4, 4)


# ---------------------------------------------------------------------------
# MLPProjector
# ---------------------------------------------------------------------------


class TestMLPProjector:
    def test_shapes_and_grids(self) -> None:
        proj = MLPProjector(
            vision_hidden_size=64, llm_hidden_size=96, intermediate_size=128
        ).eval()
        x = torch.randn(24 + 16, 64)  # grids 4x6 and 4x4
        cu = torch.tensor([0, 24, 40], dtype=torch.int32)
        grids = [(4, 6), (4, 4)]
        out, new_grids, new_cu = proj(x, grids, cu)
        # Each grid halved, tokens quartered: 6 + 4 = 10
        assert out.shape == (10, 96)
        assert new_grids == [(2, 3), (2, 2)]
        assert new_cu.tolist() == [0, 6, 10]
        assert new_cu.dtype == torch.int32

    def test_default_intermediate_equals_llm_hidden(self) -> None:
        proj = MLPProjector(vision_hidden_size=64, llm_hidden_size=96)
        assert proj.intermediate_size == 96

    def test_end_to_end_with_moonvit(
        self, tiny_model: MoonViT, tiny_cfg: MoonViTConfig
    ) -> None:
        proj = MLPProjector(
            vision_hidden_size=tiny_cfg.hidden_size,
            llm_hidden_size=96,
            intermediate_size=128,
        ).eval()
        # 4x6=24 and 2x4=8 (grids must be even).
        imgs = _mk_images([(56, 84), (28, 56)])
        with torch.no_grad():
            enc_out = tiny_model(imgs)
            tokens, grids, cu = proj(
                enc_out.last_hidden_state, enc_out.grid_shapes, enc_out.cu_seqlens
            )
        assert tokens.shape == ((24 + 8) // 4, 96)
        assert grids == [(2, 3), (1, 2)]
        assert cu.tolist() == [0, 6, 8]


# ---------------------------------------------------------------------------
# _get_activation
# ---------------------------------------------------------------------------


class TestGetActivation:
    @pytest.mark.parametrize(
        "name, cls",
        [
            ("gelu_pytorch_tanh", nn.GELU),
            ("gelu", nn.GELU),
            ("relu", nn.ReLU),
            ("silu", nn.SiLU),
        ],
    )
    def test_known(self, name: str, cls: type) -> None:
        assert isinstance(_get_activation(name), cls)

    def test_gelu_tanh_is_approximate(self) -> None:
        act = _get_activation("gelu_pytorch_tanh")
        assert isinstance(act, nn.GELU)
        assert act.approximate == "tanh"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported activation"):
            _get_activation("no_such_activation")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
