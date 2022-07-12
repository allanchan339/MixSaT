import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helper methods


# Get Arguments
def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def get_deepnorm_coefficients(encoder_layers: int, decoder_layers: int):
    """
    See DeepNet_.
    Returns alpha and beta depending on the number of encoder and decoder layers,
    first tuple is for the  for the encoder and second for the decoder
    .. _DeepNet: https://arxiv.org/pdf/2203.00555v1.pdf
    """

    N = encoder_layers
    M = decoder_layers

    if decoder_layers == 0:
        # Encoder only
        return (2 * N) ** 0.25, (8 * N) ** -0.25

    elif encoder_layers == 0:
        # Decoder only
        return (2 * M) ** 0.25, (8 * M) ** -0.25
    else:
        # Encoder/decoder
        encoder_coeffs = 0.81 * ((N ** 4) * M) ** 0.0625, 0.87 * ((N ** 4) * M) ** -0.0625

        decoder_coeffs = (3 * M) ** 0.25, (12 * M) ** -0.25

        return encoder_coeffs, decoder_coeffs


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class DeepNorm(nn.Module):
    def __init__(self, dim, fn, w, a, b):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class PatchEmbedding(nn.Module):
    def __init__(self, *, dim, dim_out, patch_size, group):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.proj = nn.Conv1d(patch_size * dim, dim_out, 1, groups=group)

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (l p) -> b (c p) l', p=p)
        return self.proj(fmap)


class PEG(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.proj = Residual(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, stride=1))

    def forward(self, x):
        return self.proj(x)


class LocalAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=30, dropout=0., local_patch_size=7, group=1):
        super().__init__()
        inner_dim = dim_head * heads
        self.local_patch_size = local_patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv1d(dim, inner_dim, 1, bias=False, groups=group)
        self.to_kv = nn.Conv1d(dim, inner_dim * 2, 1, bias=False, groups=group)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        shape, p = fmap.shape, self.local_patch_size
        b, n, l, h = *shape, self.heads
        l = l // p

        fmap = rearrange(fmap, 'b c (l p1) -> (b l) c p1', p1=p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 -> (b h) (p1) d', h=h), (q, k, v))

        dots = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim=- 1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b l h) (p) d -> b (h d) (l p)', h=h, l=l, p=p)
        return self.to_out(out)


class GlobalAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=30, dropout=0., k=7, group=1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv1d(dim, inner_dim, 1, bias=False, groups=group)
        self.to_kv = nn.Conv1d(dim, inner_dim * 2, k, stride=k, bias=False, groups=group)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, l, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) l -> (b h) (l) d', h=h), (q, k, v))

        dots = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) l d -> b (h d) l', h=h, l=l)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FrameAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(dim, inner_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, l, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) l -> (b h) (l) d', h=h), (q, k, v))

        dots = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) l d -> b (h d) l', h=h, l=l)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads=6, dim_head=30, mlp_mult=4, local_patch_size=7, global_k=7, dropout=0.,
                 has_local=True, Post_Norm=False, group=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if not Post_Norm:
                # PreNorm
                self.layers.append(nn.ModuleList([
                    # Residual(PreNorm(dim, FeatureLocalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                    #                                      local_patch_size=local_patch_size))) if has_local else nn.Identity(),
                    # Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))) if has_local else nn.Identity(),
                    # Residual(PreNorm(dim, FeatureGlobalAttention(dim, heads=heads, dim_head=dim_head,
                    #                                              dropout=dropout, k=global_k))),
                    # Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))),
                    # Residual(PreNorm(dim, FrameAttention(dim, heads=heads, dim_head=dim_head,
                    #                                              dropout=dropout))),
                    # Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout)))
                    # ]))
                    Residual(PreNorm(dim, LocalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                                         local_patch_size=local_patch_size, group=group))) if has_local else nn.Identity(),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))) if has_local else nn.Identity(),
                    Residual(PreNorm(dim, GlobalAttention(dim, heads=heads, dim_head=dim_head,
                                                          dropout=dropout, k=global_k, group=group))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))),
                    ]))
            else:
                # Post Norm
                # self.layers.append(nn.ModuleList([
                #     PostNorm(dim, Residual(LocalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                #                                           local_patch_size=local_patch_size))) if has_local else nn.Identity(),
                #     PostNorm(dim, Residual(FeedForward(dim, mlp_mult, dropout=dropout))) if has_local else nn.Identity(),
                #     PostNorm(dim, Residual(GlobalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, k=global_k))),
                #     PostNorm(dim, Residual(FeedForward(dim, mlp_mult, dropout=dropout))),
                #     PostNorm(dim, Residual(FrameAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                #     PostNorm(dim, Residual(FeedForward(dim, mlp_mult, dropout=dropout)))
                #]))
                pass

    def forward(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x


class TwinsSVT_1d_group(nn.Module):
    """
        num_classes: number of class
        frame_size: frames of feature inputed
        next_dim: next dimension
        patch_size: size of patch (ex. length / Patch size -> channel * Patch size)
        local_patch_size

        PatchEmbedding:
            input: batch channel length
            process: batch channel*patch_size length/patch_size
            process: Conv(channel*patch_size -> next_dim, 1x1)
            output: batch next_dim length/patch_size
        End

        Transformer:
            FeatureLocalAttention:
                input: batch next_dim length/patch_size
                process: batch next_dim length/patch_size -> batch*(length/patch_size/local_patch_size) next_dim patch
                        (spilt feature by local_patch_size and concat at shape[0]
                process: get q k v by Conv(next_dim -> dim_head*head, 1x1); size: batch*double_patched_length inner_dim local_patch_size)
                        1x1 conv alone with patched feature (ex. length 4)
                process: rearrage q k v: batch dim_head*head patch -> batch*head patch dim_head
                process: q k dot product
                process: softmax after qk dot product
                process: qk v dot product
                process: batch*double_patched_length patch dim_head -> batch head*dim_head length*local_patch_size
                process: Conv(dim_head*head -> next_dim, 1x1)
                process: dropout
                output: batch next_dim length/patch_size

            FeedForward:
                input: batch next_dim length/patch_size
                process: Conv(next_dim -> next_dim*mult, 1x1)
                process: GELU
                process: dropout
                process: Conv(next_dim*mult -> next_dim, 1x1)
                process: dropout
                output: batch next_dim length/patch_size)

            FeatureGlobalAttention:
                input: batch next_dim length/patch_size
                process: get q by Conv(next_dim -> dim_head*head, 1x1)
                process: get k v by Conv(next_dim -> dim_head*head, global_k, stride k)
                process: rearrage q k v: batch dim_head*head patch -> batch*head patch dim_head
                process: q k dot product
                process: softmax after qk dot product
                process: dropout
                process: qk v dot product
                process: batch*double_patched_length patch dim_head -> batch head*dim_head length*local_patch_size
                process: Conv(dim_head*head -> next_dim, 1x1)
                process: dropout
                output: batch next_dim length/patch_size

            FeedForward: ...

            FrameAttention:

            FeedForward: ...
            *All the above process will PreNorm and have residual process.
    """

    def __init__(
            self,
            *,
            num_classes,
            frames_size,
            Post_norm=False,
            # num_layer=2,
            # next_dim=[60, 720],
            # patch_size=[8, 4],
            # local_patch_size=[4, 2],
            # global_k=[20, 20],
            # depth=[1, 2],

            s1_next_dim=60,
            s1_patch_size=8,
            s1_local_patch_size=4,
            s1_global_k=20,
            s1_depth=1,
            s2_next_dim=720,
            s2_patch_size=4,
            s2_local_patch_size=2,
            s2_global_k=20,
            s2_depth=2,

            # s3_next_dim=516,
            # s3_patch_size=2,
            # s3_local_patch_size=1,
            # s3_global_k=8,
            # s3_depth=1,
            # s4_next_dim=512,
            # s4_patch_size=2,
            # s4_local_patch_size=4,
            # s4_global_k=7,
            # s4_depth=4,
            peg_kernel_size=9,
            dropout=0.
    ):
        super().__init__()

        kwargs = dict(locals())

        time_dim = frames_size  # Number of frames to input
        layers = []

        for prefix in ('s1', 's2'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's2'

            next_dim = config['next_dim']  # channel output of next dimension (ex. 7 window_size -> 64 next_sim)
            patch_size = config['patch_size']  # patch feature (ex. 8576 length / 4 patch_size)
            local_patch_size = config['local_patch_size']
            global_patch_size = config['global_k']
            depth = config['depth']

            layers.append(nn.Sequential(  # dim = 7 dim out = 64
                PatchEmbedding(dim=time_dim, dim_out=next_dim, patch_size=patch_size, group=1),
                Transformer(dim=next_dim, depth=1, local_patch_size=local_patch_size,
                            global_k=global_patch_size, dropout=dropout, has_local=not is_last,
                            group=time_dim),   # Convolution frames separately
                            # Transformer 1, group = 3; Transformer 2, group = 64
                PEG(dim=next_dim, kernel_size=peg_kernel_size),
                Transformer(dim=next_dim, depth=depth, local_patch_size=local_patch_size,
                            global_k=global_patch_size, dropout=dropout, has_local=not is_last,
                            group=1)     # Convolution frames together
            ))

            time_dim = next_dim

        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool1d(1),
            Rearrange('... () -> ...'),
            nn.Linear(time_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    input = torch.randn(1, 3, 8576)
    model = TwinsSVT_1d_group(num_classes=18, frames_size=3)
    model.eval()
    output = model(input)
    print(output.shape)
