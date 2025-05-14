
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        emb = torch.cat([torch.sin(t[:, None] * self.freqs), torch.cos(t[:, None] * self.freqs)], dim=-1)
        return self.mlp(emb)

class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = ch // heads
        self.to_qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).reshape(b, self.heads, 3 * self.head_dim, h * w)
        q, k, v = qkv.chunk(3, dim=2)
        attn = (q.transpose(-1, -2) @ k) / (self.head_dim ** 0.5)
        attn = attn.softmax(-1)
        out = (attn @ v.transpose(-1, -2)).reshape(b, c, h, w)
        return self.out(out) + x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, attn=False, down=False, up=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.attn = SelfAttention(out_ch) if attn else nn.Identity()
        if in_ch != out_ch or down or up:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        self.down = down
        self.up = up
        if down:
            self.scale = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        if up:
            self.scale = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.attn(h)
        
        
        skip = self.skip(x)   
        if self.down:
            h    = self.scale(h)        
            skip = self.scale(skip)      
        if self.up:
            h    = self.scale(h)         
            skip = self.scale(skip)

        return h + skip

class UNet(nn.Module):
    def __init__(self, base=64, time_dim=512, in_ch=64, depth=4):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        chs = [base * 2 ** i for i in range(depth)]
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.downs = nn.ModuleList()
        curr = base
        for ch in chs:
            self.downs.append(ResBlock(curr, ch, time_dim, attn=True, down=True))
            curr = ch
        self.mid = ResBlock(curr, curr, time_dim, attn=True)
        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(ResBlock(curr + ch, ch, time_dim, attn=True, up=True))
            curr = ch
        self.out_norm = nn.GroupNorm(32, curr)
        self.out_conv = nn.Conv2d(curr, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = []
        x = self.in_conv(x)
        for block in self.downs:
            x = block(x, t_emb)
            h.append(x)
        x = self.mid(x, t_emb)
        for block in self.ups:
            skip = h.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
        return self.out_conv(F.silu(self.out_norm(x)))
