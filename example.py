import torch

from open_moonvit.main import MLPProjector, MoonViT, MoonViTConfig

# Tiny end-to-end check: two mixed-size fake images through MoonViT and the projector.
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
print(f"encoder parameters: {n_params:,}")
print(f"encoder output: {tuple(out.last_hidden_state.shape)}")
print(f"projector output: {tuple(proj_x.shape)}")
print("OK")
