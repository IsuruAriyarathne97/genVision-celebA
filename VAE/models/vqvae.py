
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, z_dim=64):
        super().__init__()
        ch = 64
        layers = [
            nn.Conv2d(in_ch, ch, 3, padding=1),
            ResBlock(ch), ResBlock(ch),
            nn.Conv2d(ch, ch*2, 4, stride=2, padding=1),
            ResBlock(ch*2), ResBlock(ch*2),
            nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1),
            ResBlock(ch*4), ResBlock(ch*4),
            nn.Conv2d(ch*4, z_dim, 1)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim=64, out_ch=3):
        super().__init__()
        ch = 256
        layers = [
            nn.Conv2d(z_dim, ch, 3, padding=1),
            ResBlock(ch), ResBlock(ch),
            nn.ConvTranspose2d(ch, ch//2, 4, stride=2, padding=1),
            ResBlock(ch//2), ResBlock(ch//2),
            nn.ConvTranspose2d(ch//2, ch//4, 4, stride=2, padding=1),
            ResBlock(ch//4), ResBlock(ch//4),
            nn.Conv2d(ch//4, out_ch, 3, padding=1),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_codebook, dim, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codebook, dim)
        self.beta = beta
        nn.init.uniform_(self.codebook.weight, -1./num_codebook, 1./num_codebook)

    def forward(self, z):
        # BCHW -> BHWC
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, z_perm.shape[-1])
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                + self.codebook.weight.pow(2).sum(1)
                - 2 * torch.matmul(z_flat, self.codebook.weight.t()))
        idx = dist.argmin(-1)
        z_q = self.codebook(idx).view(z_perm.shape).permute(0, 3, 1, 2).contiguous()
        commit_loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, commit_loss

class VQVAE(nn.Module):
    def __init__(self, num_codebook=1024, dim=64):
        super().__init__()
        self.encoder = Encoder(z_dim=dim)
        self.vq = VectorQuantizer(num_codebook, dim)
        self.decoder = Decoder(z_dim=dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, commit = self.vq(z)
        recon = self.decoder(z_q)
        return recon, commit
