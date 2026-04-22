import torch

from open_moonvit.main import MLPProjector, MoonViT, MoonViTConfig

# Tiny end-to-end check: two mixed-size fake images through MoonViT and the projector.
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
    print(out)
    print(out.last_hidden_state.shape)
