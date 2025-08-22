from wan.modules.vae import _video_vae
model = _video_vae(z_dim=16, device='cuda')
for n, m in model.named_modules():
    print(f"Name: {n}")
    print(f"Module: {m}")
    print("-" * 50)