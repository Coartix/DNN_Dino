import torch
import torchvision
import torch.nn as nn
import math
import warnings
from utils import trunc_normal_

""" 
    Regularization technique introduced in the paper "Deep Networks with Stochastic Depth"
    Stochastic Depth involves randomly dropping (setting to zero) entire layers during training.
        The idea is to reduce overfitting and improve generalization by introducing a form
        of random depth in the network.
"""
class DropLayer(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropLayer, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # Determine the shape of the random tensor
        # Generate a random tensor with values between keep_prob and 1.0
        rd_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        # Binarize the random tensor
        rd_tensor.floor_()
        # Apply drop by scaling the input tensor and element-wise multiplication with the random tensor
        # the scaling compensates to maintain the overall magnitude of the tensor
        output = x.div(keep_prob) * rd_tensor
        return output

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0., proj_drop=0.):
        super(SelfAttention, self).__init__()
        
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attention


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, norm_layer=nn.LayerNorm,
                 mlp_ratio = 4., layer_drop=0., attn_drop=0., proj_drop=0.):
        super(Block, self).__init__()
        
        self.norm1 = norm_layer(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, proj_drop=proj_drop, attn_drop=attn_drop)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=proj_drop)

        self.drop_layer = DropLayer(layer_drop) if layer_drop > 0 else nn.Identity()
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=proj_drop)

    def forward(self, x, return_attention=False):
        _x, attn = self.attn(self.norm1(x))
        if return_attention:
            # To visualize the attention map
            return attn
        
        x = x + self.drop_layer(_x)
        x = x + self.drop_layer(self.mlp(self.norm2(x)))
        return x

""" 
    Effectue l'encodage des patches d'une image
    La classe PatchEmbed projette les patches d'une image en utilisant une couche de convolution
    Exemple: Input(batch_size, num_channels, width_image, height_image) -> Output(batch_size, num_patches, embed_dim)
"""
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, in_chans=3):
        super(PatchEmbed, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViT_small(nn.Module):
    def __init__(self, img_size=[32], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_layer_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super(ViT_small, self).__init__()

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dlr = [x.item() for x in torch.linspace(drop_layer_rate, 0., depth)]
        self.blocks = nn.ModuleList([
            Block(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop_rate,
                layer_drop=dlr[i], norm_layer=norm_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # To change

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        
        '''
        # If the number of patches is different from the number of positional embeddings
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w_patch = w // self.patch_embed.patch_size
        h_patch = h // self.patch_embed.patch_size
        # Add small increments to avoid floating point errors
        w_patch, h_patch = w_patch + 0.1, h_patch + 0.1

        # Interpolate the positional encoding to the size of the patches
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w_patch / math.sqrt(N), h_patch / math.sqrt(N)),
            mode='bicubic'
        )
        assert int(w_patch) == patch_pos_embed.shape[-2] and int(h_patch) == patch_pos_embed.shape[-1]

        # Reshape the positional encoding back to the original shape
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        '''

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to the embed patch tokens without changing dimensionality
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])
        
        
def vit_tiny(patch_size=8, **kwargs):
    model = ViT_small(patch_size=patch_size, **kwargs)
    return model