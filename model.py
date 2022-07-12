# simple vit, from https://arxiv.org/abs/2205.01580
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device),
                          torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def  __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    def __init__(self, *, patch_size, num_classes=18, dim=8576, depth=8, heads=16, mlp_dim=1000, dim_head=64):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * \
        #     (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
        #               p1=patch_height, p2=patch_width),
        #     nn.Linear(patch_dim, dim),
        # )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # *_, h, w, dtype = *img.shape, img.dtype

        # x = self.to_patch_embedding(img)

        # require 2d h*w patches, where i dont know how 
        # pe = posemb_sincos_2d(x)
        # x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        x = self.linear_head(x)
        x = self.sigmoid(x)
        return x


class SimpleViT_Mask(nn.Module):
    def __init__(self, *, patch_size, num_classes=18, dim=8576, depth=8, heads=16, mlp_dim=1000, dim_head=64,
                 masking_ratio=0.75, masking_epoch=0):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * \
        #     (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)',
        #               p1=patch_height, p2=patch_width),
        #     nn.Linear(patch_dim, dim),
        # )
        assert masking_ratio >= 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        # self.masking_ratio = masking_ratio
        self.num_masked = int(masking_ratio * patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_size + 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.sigmoid = nn.Sigmoid()
        self.current_epoch = 0
        self.masking_epoch = masking_epoch
        print(f'masking epoch = {self.masking_epoch}; masking frame = {self.num_masked}')

    def update_epoch(self, epoch):
        # call update_epoch before computation please
        # if it is being updated, it means we train in different way
        self.current_epoch = epoch

    def forward(self, x):

        # *_, h, w, dtype = *img.shape, img.dtype

        # x = self.to_patch_embedding(img)

        # require 2d h*w patches, where i dont know how
        # pe = posemb_sincos_2d(x)
        # x = rearrange(x, 'b ... d -> b (...) d') + pe

        # inside forward, gpu is used, where gpu dont proceed if-else clause
        # if self.masking_ratio > 0:

        device = x.device
        # [BS, windows, 8576]
        batch, num_patches, *_ = x.shape

        # patch to encoder tokens and add positions
        # x = x + self.pos_embedding[:, 1:(num_patches + 1)]  # equal to [:,:]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        rand_indices = torch.rand(
            batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:,
                                           :self.num_masked], rand_indices[:, self.num_masked:]
        # TODO: check if important or not, seems position embedding already did the job
        unmasked_indices, indices = unmasked_indices.sort()  # make it follow temporal order, eg. [0,1,3,4]
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        if (self.current_epoch < self.masking_epoch) and (self.masking_epoch > 0):
            # masking_epoch > 0 and current epoch < masking_epoch
            x = x
        else:
            x = x[batch_range, unmasked_indices]  # select data from indices

        x = self.transformer(x)  # it work as patch_size is just channel
        x = x.mean(dim=1)

        x = self.to_latent(x)
        x = self.linear_head(x)
        x = self.sigmoid(x)
        return x

