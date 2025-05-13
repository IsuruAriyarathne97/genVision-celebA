#!/usr/bin/env python


import os
import sys
import argparse
import torch
import numpy as np
from torchvision import datasets, transforms, utils as tv_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae import VQVAE
from PIL import Image
from tqdm import tqdm
from pytorch_fid import fid_score

def denormalize(tensor):
    return (tensor * 0.5) + 0.5

def save_comparison_grid(original, reconstructed, path, n=8):

    original = denormalize(original[:n]).cpu().numpy()
    reconstructed = denormalize(reconstructed[:n]).cpu().numpy()
    
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i+1)
        plt.imshow(np.transpose(original[i], (1, 2, 0)))
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        
        # Reconstructed
        plt.subplot(2, n, i+n+1)
        plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved comparison grid to {path}")

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and reconstructed images"""
    # Convert to 0-1 range
    original = denormalize(original)
    reconstructed = denormalize(reconstructed)
    
    mse = torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])

    psnr = 10 * torch.log10(1.0 / mse)

    return psnr.mean().item()

def save_images_for_fid(loader, model, device, folder, max_images=5000):

    os.makedirs(os.path.join(folder, 'original'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'reconstructed'), exist_ok=True)
    
    count = 0
    
    for batch, _ in tqdm(loader, desc="Generating images for FID"):
        if count >= max_images:
            break
        
        batch = batch.to(device)
        with torch.no_grad():
            reconstructed, _ = model(batch)
        
        for i in range(batch.size(0)):
            if count >= max_images:
                break
            
            # Save original image
            img = denormalize(batch[i]).cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(folder, 'original', f"{count:05d}.png"))
            
            # Save reconstructed image
            img = denormalize(reconstructed[i]).cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(folder, 'reconstructed', f"{count:05d}.png"))
            
            count += 1
    
    return count


def generate_samples(model: VQVAE, num_samples: int = 64, out_dir: str = "samples", device=None):


    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
      
        batch_size = 32  
        all_samples = []
        
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
           
            latent_h = 32 
            latent_w = 32
            indices = torch.randint(0, model.vq.codebook.num_embeddings, 
                                  (current_batch, latent_h, latent_w), 
                                  device=device)
            
            z_q_flat = model.vq.codebook(indices.view(-1))
            z_q = z_q_flat.view(current_batch, latent_h, latent_w, -1)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            
            samples = model.decoder(z_q)
            all_samples.append(samples)
            samples = torch.cat(all_samples, dim=0)
        
        grid = tv_utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
        
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        tv_utils.save_image(grid, out_path / "generated_samples.png")
        
        for i, sample in enumerate(samples):
            tv_utils.save_image(sample, out_path / f"sample_{i:03d}.png", normalize=True, value_range=(-1, 1))
        
        return samples
    
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = VQVAE(num_codebook=args.num_codebook, dim=args.dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    dataset = datasets.CelebA(
        args.data_root, split=args.split, download=False, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Generating reconstruction examples...")
    dataiter = iter(loader)
    images, _ = next(dataiter)
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    
    grid_path = os.path.join(args.output_dir, "reconstruction_grid.png")
    save_comparison_grid(images, reconstructed, grid_path)
    
    # Calculate PSNR
    psnr = calculate_psnr(images, reconstructed)
    print(f"PSNR on sample batch: {psnr:.2f} dB")
    
    mse = torch.nn.functional.mse_loss(denormalize(reconstructed), denormalize(images)).item()
    print(f"MSE on sample batch: {mse:.4f}")
    
    samples = generate_samples(
            model, 
            num_samples=args.num_samples,
            out_dir="samples",
            device=device
        )
    print(f"Generated {len(samples)} samples")
    if args.calculate_fid:
        try:
            print("Preparing images for FID calculation...")
            fid_img_dir = os.path.join(args.output_dir, "fid_images")
            num_images = save_images_for_fid(loader, model, device, fid_img_dir, args.max_fid_images)
            
            print(f"Calculating FID score using {num_images} images...")
            fid_value = fid_score.calculate_fid_given_paths(
                [os.path.join(fid_img_dir, 'original'), os.path.join(fid_img_dir, 'reconstructed')],
                args.batch,
                device,
                dims=2048
            )
            print(f"FID score: {fid_value:.2f}")
            
            with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
                f.write(f"Model: {args.model_path}\n")
                f.write(f"PSNR: {psnr:.2f} dB\n")
                f.write(f"MSE: {mse:.4f}\n")
                f.write(f"FID: {fid_value:.2f}\n")
                
        except Exception as e:
            print(f"Error calculating FID: {e}")
            print("Make sure pytorch-fid is installed: pip install pytorch-fid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VQVAE')
    parser.add_argument('--model_path', type=str, default='../checkpoints/vqvae.pt', help='Path to the trained model')
    parser.add_argument('--data_root', type=str, default='../data', help='Path to dataset')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (train/valid/test)')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--num_codebook', type=int, default=1024, help='Number of codebook embeddings')
    parser.add_argument('--dim', type=int, default=64, help='Dimensionality of latent vectors')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID score')
    parser.add_argument('--max_fid_images', type=int, default=5000, help='Maximum number of images to use for FID')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    args = parser.parse_args()
    main(args)