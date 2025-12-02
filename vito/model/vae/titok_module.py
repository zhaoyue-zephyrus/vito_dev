# Titok for Video
# Code adapted from https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/blocks.py

from collections import OrderedDict
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    def __init__(
        self,
        video_size=256,
        video_length=16,
        patch_size=8,
        patch_length=4,
        in_chans=3,
        z_chans=4,
        double_z=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cls_token=True,
        norm_code=False,
        ln_in_attn=False,
        conv_last_layer=False,
        use_rope=False,
        use_final_proj=False,
        num_latent_tokens=64,
        is_legacy=False,
    ):
        super().__init__()

        self.grid_size = video_size // patch_size
        self.video_length = video_length // patch_length
        self.num_patches = self.video_length * self.grid_size ** 2
        self.num_latent_tokens = num_latent_tokens

        self.width = embed_dim
        self.num_layers = depth
        self.num_heads = num_heads
        self.token_size = z_chans * (2 if double_z else 1)
        self.is_legacy = is_legacy

        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=(patch_length, patch_size, patch_size),
            stride=(patch_length, patch_size, patch_size),
            bias=True)

        scale = self.width ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(num_latent_tokens, self.width))
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_patches + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=mlp_ratio
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        

        latent_tokens = _expand_token(self.latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.num_patches:]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens

class TiTokDecoder(nn.Module):
    def __init__(
        self,
        video_size=256,
        video_length=16,
        patch_size=8,
        patch_length=4,
        in_chans=3,
        z_chans=4,
        double_z=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cls_token=True,
        norm_code=False,
        ln_in_attn=False,
        conv_last_layer=False,
        use_rope=False,
        use_final_proj=False,
        num_latent_tokens=64,
        is_legacy=False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.grid_size = video_size // patch_size
        self.video_length = video_length // patch_length
        self.num_patches = self.video_length * self.grid_size ** 2
        self.num_latent_tokens = num_latent_tokens

        self.width = embed_dim
        self.num_layers = depth
        self.num_heads = num_heads
        self.token_size = z_chans
        self.is_legacy = is_legacy

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, self.width))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=mlp_ratio
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv3d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv3d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv3d(self.width, patch_length * self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (s p1 p2 c) t h w -> b c (t s) (h p1) (w p2)',
                    s=patch_length, p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv3d(3, 3, 3, padding=1, bias=True)

    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.num_patches, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.num_patches] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D T H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.video_length, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x
