
import torch, os, argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vqvae import VQVAE
from models.unet import UNet
import math

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


## THIS MODEL IS NOT USED, DID NOT WORK, EPIC FAIL. WENT TOO DEEP.  

def cosine_beta_schedule(timesteps):
    steps = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos((steps / timesteps) * math.pi / 2) ** 2
    betas = 1 - f[1:] / f[:-1]
    return torch.clamp(betas, 1e-4, 0.999)

def get_alphas(timesteps, device):
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, 0)
    return alphas, alphas_cum

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load pretrained vqvae encoder
    vq = VQVAE().to(device)
    vq.load_state_dict(torch.load(args.vq_ckpt, map_location=device))
    vq.eval()
    encoder = vq.encoder
    # diffusion model
    unet = UNet(base=64, in_ch=64).to(device)
    opt = torch.optim.AdamW(unet.parameters(), 1e-4, betas=(0.9, 0.95))
    alphas, alphas_cum = get_alphas(args.T, device)
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = datasets.CelebA(args.data_root, split='train', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    ema = UNet(base=64, in_ch=64).to(device)
    ema.load_state_dict(unet.state_dict())
    ema_decay = 0.9999

    loader_iter = iter(loader)
    losses = []

    for step in range(1, args.steps + 1):
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, _ = next(loader_iter)

        x = x.to(device)    

        with torch.no_grad():
            z = encoder(x)  # B 64 16 16
        t = torch.randint(0, args.T, (x.size(0),), device=device)
        noise = torch.randn_like(z)
        sqrt_acum = alphas_cum[t][:, None, None, None].sqrt()
        sqrt_one_minus = (1 - alphas_cum[t])[:, None, None, None].sqrt()
        z_t = sqrt_acum * z + sqrt_one_minus * noise
        pred = unet(z_t, t.float())
        loss = torch.nn.functional.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        # Update EMA
        with torch.no_grad():
            for p, q in zip(ema.parameters(), unet.parameters()):
                p.data = ema_decay * p.data + (1 - ema_decay) * q.data
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f}")

        if step % args.log_freq == 0:
            print(f"Step [{step}/{args.steps}]  Loss: {loss.item():.4f}")

        # Checkpointing
        if step % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(args.out, f"ema_step{step}.pt")
            torch.save(ema.state_dict(), ckpt_path)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  → Saved checkpoint: {ckpt_path} at {current_time}")

    os.makedirs(args.out, exist_ok=True)
    final_ckpt = os.path.join(args.out, 'ema_final.pt')
    torch.save(ema.state_dict(), final_ckpt)
    print(f"Training completed. Final model saved to '{final_ckpt}'.")
    loss_file_path = os.path.join(args.out, 'losses.txt')
    np.savetxt(loss_file_path, losses)
    print(f"Losses saved to {loss_file_path}")  
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.out, 'loss_plot_ld.png'))
        
    os.makedirs(args.out, exist_ok=True)
    torch.save(ema.state_dict(), os.path.join(args.out, 'unet_ema.pt'))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vq_ckpt", default="checkpoints/vqvae.pt")
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--checkpoint_freq", type=int, default=10000)
    main(p.parse_args())
