#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
import matplotlib.pyplot as plt

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Model, optimizer, loss
    model = VQVAE(num_codebook=args.num_codebook, dim=args.dim).to(device)
    model.train()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95)
    )
    recon_loss = nn.MSELoss()

    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = datasets.CelebA(
        args.data_root, split='train', download=False, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    loader_iter = iter(loader)
    losses = []

    for step in range(1, args.steps + 1):
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, _ = next(loader_iter)

        x = x.to(device)
        recon, commit = model(x)
        loss = recon_loss(recon, x) + commit

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        losses.append(loss.item())
        if step % args.log_freq == 0:
            print(f"Step [{step}/{args.steps}]  Loss: {loss.item():.4f}")

        # Checkpointing
        if args.ckpt_freq > 0 and step % args.ckpt_freq == 0:
            ckpt_path = os.path.join(args.out, f"vqvae_step{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint/plot
    os.makedirs(args.out, exist_ok=True)
    final_ckpt = os.path.join(args.out, 'vqvae_final.pt')
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training completed. Final model saved to '{final_ckpt}'.")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.out, 'loss_plot.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VQ-VAE')
    parser.add_argument('--data_root', type=str, default='data', help='Path to dataset')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency (in steps)')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--num_codebook', type=int, default=1024, help='Number of codebook embeddings')
    parser.add_argument('--dim', type=int, default=64, help='Dimensionality of latent vectors')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--out', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--ckpt_freq', type=int, default=10000, help='Checkpoint frequency (in steps), 0 to disable')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main(args)
