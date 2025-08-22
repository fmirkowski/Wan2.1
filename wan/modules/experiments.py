from wan.modules.vae import _video_vae
model = _video_vae(z_dim=16, device='cuda')
[print(n) for n, p in model.named_modules()]