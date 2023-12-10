import torch
import torchvision
import torch.nn as nn

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
    
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        
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

        attention = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)

        x = torch.bmm(attention, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attention


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
    

class ViT_small(nn.Module):
    def __init__(self, embed_dim, nb_blocks, patch_size, hidden_dim, depth, num_heads, 
        super(ViT_small, self).__init__()
        
        self.patch_size = patch_size

        self.head = nn.Identity()
        
        
    