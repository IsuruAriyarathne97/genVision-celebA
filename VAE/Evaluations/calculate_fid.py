import argparse
import os
import sys
import math
import time
import shutil
from pathlib import Path

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as tv_utils
from pytorch_fid import fid_score


from ..models.vanilla_vae import VAE, Encoder, Decoder

def save_images_for_fid(images_tensor, base_path: Path, start_idx: int = 0):
    images_tensor = images_tensor.cpu()
    for i, img_tensor in enumerate(images_tensor):
        tv_utils.save_image(
            img_tensor,
            base_path / f"image_{start_idx + i:06d}.png",
            normalize=True,
            value_range=(-1, 1)
        )

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    fid_base_dir = Path("./fid_calculation_images")
    real_images_dir = fid_base_dir / "real"
    fake_images_dir = fid_base_dir / "fake"

    if fid_base_dir.exists():
        shutil.rmtree(fid_base_dir)
    real_images_dir.mkdir(parents=True, exist_ok=True)
    fake_images_dir.mkdir(parents=True, exist_ok=True)

    model = VAE(latent_dim=args.latent_dim).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
         model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"Loaded model from {args.checkpoint_path} (using VAE definition from train_v4.py)")

    print(f"Generating {args.num_samples} fake images...")
    generated_count = 0
 

    latent_z_dim_for_sampling = model.encoder.fc_mu.out_features

    with torch.no_grad():
        for i in range(0, args.num_samples, args.batch_size):
            batch_num_samples = min(args.batch_size, args.num_samples - i)
            if batch_num_samples <= 0:
                break
            
            z_to_decode = torch.randn(batch_num_samples, latent_z_dim_for_sampling, device=device)

            fake_imgs = model.decoder(z_to_decode)
            
            save_images_for_fid(fake_imgs, fake_images_dir, start_idx=generated_count)
            generated_count += batch_num_samples
            if (i // args.batch_size + 1) % 10 == 0:
                 print(f"  Generated {generated_count}/{args.num_samples} fake images...")
    print(f"Finished generating {generated_count} fake images in {fake_images_dir}")

    print(f"Preparing {args.num_samples} real images...")
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    real_dataset = datasets.CelebA(args.data_root, split="test", download=False, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    saved_real_count = 0
    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(real_loader):
            if saved_real_count >= args.num_samples:
                break
            num_to_save = min(real_imgs.size(0), args.num_samples - saved_real_count)
            save_images_for_fid(real_imgs[:num_to_save], real_images_dir, start_idx=saved_real_count)
            saved_real_count += num_to_save
            if (i + 1) % 10 == 0 :
                 print(f"  Saved {saved_real_count}/{args.num_samples} real images...")
    print(f"Finished saving {saved_real_count} real images in {real_images_dir}")

    if saved_real_count == 0 or generated_count == 0:
        print("Error: No real or generated images were saved. Cannot calculate FID.")
        if fid_base_dir.exists():
            shutil.rmtree(fid_base_dir)
        return

    print("Calculating FID score...")
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[str(real_images_dir), str(fake_images_dir)],
        batch_size=args.fid_batch_size,
        device=device,
        dims=args.dims,
        num_workers=args.num_workers
    )
    print(f"FID score: {fid_value:.4f}")

    if args.cleanup_images:
        print(f"Cleaning up image directory: {fid_base_dir}")
        shutil.rmtree(fid_base_dir)
    else:
        print(f"Generated images saved in {fid_base_dir} (real and fake subdirectories).")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calculate FID score for a VAE model, importing model from train_v4.py.")
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to VAE model checkpoint (e.g., vae_epoch_059.pt).")
    p.add_argument("--data_root", type=str, required=True, help="Path to CelebA root directory.")
    p.add_argument("--latent_dim", type=int, default=256, help="Latent dimension used to initialize VAE (must match the checkpoint's architecture).")
    p.add_argument("--image_size", type=int, default=128, help="Image size (VAE output and real images will be resized to this).")
    p.add_argument("--num_samples", type=int, default=10000, help="Number of real/fake samples to use for FID calculation.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for generating images and loading real images.")
    p.add_argument("--fid_batch_size", type=int, default=50, help="Batch size for FID calculation step (InceptionV3 forward pass).")
    p.add_argument("--dims", type=int, default=2048, help="Inception feature dimensions (e.g., 2048 for default InceptionV3 features).")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use ('cuda' or 'cpu').")
    p.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    p.add_argument("--cleanup_images", action='store_true', help="Remove temporary image directories after FID calculation.")

    args = p.parse_args()
    main(args)