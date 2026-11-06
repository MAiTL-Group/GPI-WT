# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.0 (default, Nov  6 2019, 21:49:08) 
# [GCC 7.3.0]
# Embedded file name: /data2/luochen/Mypaper4/WKT320/unrolled.py
# Compiled at: 2024-02-10 17:14:44
# Size of source mod 2**32: 40519 bytes
import torch
import torch.nn as nn
from torch.nn import init
import os
import torch.nn.functional as F
from utils import ifft2c, fft2c, r2c, c2r
#from utils_spirit import  crop, spirit_calibrate, spirit, adjspirit
from spirit import crop, spirit_calibrate, spirit, adjspirit
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange
#from spectral_norm import SpectralNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

conv_size = 64
filter_size = 3

class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.05):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight'):
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    if init_type == 'normal':
                        nn.init.normal_(m.weight.data, 0.0, gain)
                    else:
                        if init_type == 'xavier':
                            nn.init.xavier_normal_((m.weight.data), gain=gain)
                        else:
                            if init_type == 'kaiming':
                                nn.init.kaiming_normal_((m.weight.data), a=0, mode='fan_in')
                            else:
                                if init_type == 'orthogonal':
                                    nn.init.orthogonal_((m.weight.data), gain=gain)
                                elif hasattr(m, 'bias') and m.bias is not None:
                                    nn.init.constant_(m.bias.data, 0.0)
            if classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def pair(t):
    if isinstance(t, tuple):
        return t
    return (t, t)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return (self.fn)((self.norm(x)), **kwargs)



#class Attention(nn.Module):
#
#    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
#        super().__init__()
#        inner_dim = dim_head * heads
#        self.heads = heads
#        self.scale = dim_head ** (-0.5)
#        self.attend = nn.Softmax(dim=(-1))
#        self.dropout = nn.Dropout(dropout)
#        self.qkv = nn.Linear(dim, inner_dim, bias=False)
#        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))
#
#    def forward(self, x):
#        w = rearrange((self.qkv(x)), 'b n (h d) -> b h n d', h=(self.heads))
#        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
#        attn = self.attend(dots)
#        attn = self.dropout(attn)
#        out = torch.matmul(attn, w)
#        out = rearrange(out, 'b h n d -> b n (h d)')
#        return self.to_out(out)

class LAttention(BaseNetwork):
    def __init__(self, dim, num_heads, window_size=[300, 1], qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #trunc_normal_((self.relative_position_bias_table), std=0.02)
        self.softmax = nn.Softmax(dim=(-1))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[0], qkv[0]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
#        if mask is not None:
#            nW = mask.shape[0]
#            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#            attn = attn.view(-1, self.num_heads, N, N)
#            attn = self.softmax(attn)
#        else:
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Mlp(BaseNetwork):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(BaseNetwork):
    __doc__ = ' Window based multi-head self attention (W-MSA) module with relative position bias.\n    It supports both of shifted and non-shifted window.\n\n    Args:\n        dim (int): Number of input channels.\n        window_size (tuple[int]): The height and width of the window.\n        num_heads (int): Number of attention heads.\n        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True\n        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set\n        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0\n        proj_drop (float, optional): Dropout ratio of output. Default: 0.0\n    '

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_((self.relative_position_bias_table), std=0.02)
        self.softmax = nn.Softmax(dim=(-1))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[0], qkv[0]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            print(mask)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class WTransformerBlock(BaseNetwork):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        #self.mlp_ratio = mlp_ratio

        self.norm = norm_layer(dim)
        self.attn = WindowAttention(dim,
              window_size=(to_2tuple(self.window_size)), num_heads=num_heads, qkv_bias=qkv_bias,
              qk_scale=qk_scale,
              attn_drop=attn_drop,
              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        #self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

#    def calculate_mask(self, x_size):
#        H, W = x_size
#        img_mask = torch.zeros((1, H, W, 1))
#        h_slices = (slice(0, -self.window_size),
#         slice(-self.window_size, -self.shift_size),
#         slice(-self.shift_size, None))
#        w_slices = (slice(0, -self.window_size),
#         slice(-self.window_size, -self.shift_size),
#         slice(-self.shift_size, None))
#        cnt = 0
#        for h in h_slices:
#            for w in w_slices:
#                img_mask[:, h, w, :] = cnt
#                cnt += 1
#            else:
#                mask_windows = window_partition(img_mask, self.window_size)
#                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=(self.attn_mask))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        return x



class TransKW(BaseNetwork):
    def __init__(self, img_size=384, in_chans=30, embed_dim=96, window_size=8, num_heads=6, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1.0, **kwargs):
        super(TransKW, self).__init__()

        num_feat = 64
        #self.img_range = img_range
        #self.upscale = upscale
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1, bias=True)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)

#        self.line_first = nn.Sequential(
#            nn.Linear(in_chans, embed_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(embed_dim, embed_dim),
#            )
#        self.line_last = nn.Sequential(
#            nn.Linear(embed_dim, embed_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(embed_dim, in_chans),
#            )
#        self.line_after_body = nn.Sequential(
#            nn.Linear(embed_dim, embed_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(embed_dim, embed_dim),
#            )
    
        self.embed_dim = embed_dim
        #self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        #self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layer = WTransformerBlock(dim=embed_dim, 
              num_heads=num_heads,
              window_size=window_size,
              #mlp_ratio=(self.mlp_ratio),
              qkv_bias=qkv_bias,
              qk_scale=qk_scale,
              drop=drop_rate,
              attn_drop=attn_drop_rate,
              norm_layer=norm_layer, 
              ) 
              
               
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_((m.weight), std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_window(self, x): 
        x_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        #x = self.pos_drop(x)
        x = self.layer(x, x_size)
        ##x = self.norm(x)
        #x = x+self.pos_drop(self.layer(x, x_size))
        x = x+self.pos_drop(x)
#        x = self.layers[1](x, x_size)
#        x = self.norm(x)        
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


    def forward(self, x):
##        x = c2r(r2c(x).transpose(0, 1))
#        H, W = x.shape[2:]
#        x = x.permute(0, 2, 3, 1)        
#        x_first = self.line_first(x)    
#        x_first1 = x_first.permute(0, 3, 1, 2)  
#        res = self.forward_window(x_first1)
#        res = res.permute(0, 2, 3, 1)    
#        res = x_first + self.line_after_body(res) 
#        x = x + self.line_last(res)
#        x = x.permute(0, 3, 1, 2)  
##        x = c2r(r2c(x).transpose(0, 1))  

        H, W = x.shape[2:]       
        x_first = self.conv_first(x)     
        res = self.forward_window(x_first)    
        res = x_first + self.conv_after_body(res) 
        x = x + self.conv_last(res)         
        return x



class TransKL(BaseNetwork):
    def __init__(self, img_size=384, in_chans=30, embed_dim=96, window_size=8, num_heads=6, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1.0, **kwargs):
        super(TransKL, self).__init__()
        self.device = 'cuda:1'
        #print(in_chans)
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1, bias=True)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)

#        self.line_first = nn.Sequential(
#            nn.Linear(in_chans, embed_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(embed_dim, embed_dim),
#            )
#        self.line_last = nn.Sequential(
#            nn.Linear(embed_dim, embed_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(embed_dim, in_chans),
#            )
#        self.line_after_body = nn.Sequential(
#            nn.Linear(embed_dim, embed_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(embed_dim, embed_dim),
#            )                  
#    
        self.embed_dim = embed_dim
        self.num_features = embed_dim


        self.pos_drop = nn.Dropout(p=drop_rate)
 
              
        self.layer =  LAttention(dim=embed_dim, 
              num_heads=num_heads,
              window_size = [self.window_size,1],
              qkv_bias=qkv_bias,
              qk_scale=qk_scale,
              attn_drop=attn_drop_rate,
              )
               
        self.norm = norm_layer(self.num_features)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_((m.weight), std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_line(self, x): #[1, embed_dim, 320, 320]]
        B, C, H, W = x.shape #15,32,320,300
        x = x.permute(0, 3, 2, 1) #x.permute(0, 2, 3, 1) #BHWC
        x = rearrange(x, 'b h (t w) c -> (b h t) w c', t=1)
        #x = x.flatten(0,1)
        x = self.norm(x)
        x = self.layer(x) 
        x = x+self.pos_drop(x) 
        #x = x+self.pos_drop(self.layer(x) )  
        #x = self.pos_drop(x) 
        #x = self.layer2(x)      
        #x = self.norm(x)
        x = rearrange(x, '(b h t) w c -> b h (t w) c', b=B, t=1)
        #x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1) #x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
##        x = c2r(r2c(x).transpose(0, 1))
#        H, W = x.shape[2:]
#        #print('cn', next(self.conv_first.parameters()).device)   
#        x = x.permute(0, 2, 3, 1)         
#        x_first = self.line_first(x)    
#        res = self.forward_line(x_first)
#        res = x_first + self.line_after_body(res) 
#        x = x + self.line_last(res) 
#        x = x.permute(0, 3, 1, 2) 
##        x = c2r(r2c(x).transpose(0, 1))            

        H, W = x.shape[2:]
        #print('cn', next(self.conv_first.parameters()).device)          
        x_first = self.conv_first(x)    
        res = self.forward_line(x_first)
        res = x_first + self.conv_after_body(res) 
        x = x + self.conv_last(res) 

        return x



    
class WBTBlocknW(BaseNetwork):

    def __init__(self, image_size=368, nch=30):
        super(WBTBlocknW, self).__init__()
        self.nch = nch

        self.Trans_KW = TransKW(img_size=image_size, in_chans=30, embed_dim=300, window_size=4, num_heads=6,
          mlp_ratio=2.0,
          qkv_bias=True,
          qk_scale=None,
          drop_rate=0.1,
          attn_drop_rate=0.0,
          drop_path_rate=0.1,
          norm_layer=(nn.LayerNorm),
          patch_norm=True,
          upscale=1,
          img_range=1.0 ) 

#        self.Trans_IW = TransIW(img_size=image_size, in_chans=2, embed_dim=126, window_size=4, num_heads=6,
#          mlp_ratio=2.0,
#          qkv_bias=True,
#          qk_scale=None,
#          drop_rate=0.1,
#          attn_drop_rate=0.0,
#          drop_path_rate=0.1,
#          norm_layer=(nn.LayerNorm),
#          patch_norm=True,
#          upscale=1,
#          img_range=1.0 )                  
          
        self.lam = nn.Parameter(torch.Tensor([0.8]))
        self.mu = nn.Parameter(torch.Tensor([0.05]))
        self.miu1 = nn.Parameter(torch.Tensor([0.05])) #0.05
#        self.miu2 = nn.Parameter(torch.Tensor([0.05]))

    def forward(self, xK, mask, atb, kernel, W):
        
    
        y = xK - self.miu1 * self.Trans_KW(xK*W)/W - self.mu * c2r(adjspirit(kernel, spirit(kernel, r2c(xK)))) 
# - self.miu2 * c2r(fft2c(r2c( self.Trans_IW(c2r(ifft2c(r2c(xK)))) )))  
        xK = y - self.lam * mask * xK + self.lam * atb
        
        return xK
        
class WBTBlocknL(BaseNetwork):

    def __init__(self, image_size=368, nch=30):
        super(WBTBlocknL, self).__init__()
        self.nch = nch
        device = 'cuda:1'

        self.Trans_KL = TransKL(img_size=image_size, in_chans=30, embed_dim=300, window_size=320, num_heads=6,
          mlp_ratio=2.0,
          qkv_bias=True,
          qk_scale=None,
          drop_rate=0.1,
          attn_drop_rate=0.0,
          drop_path_rate=0.1,
          norm_layer=(nn.LayerNorm),
          patch_norm=True,
          upscale=1,
          img_range=1.0 )

#        self.Trans_IL = TransIL(img_size=image_size, in_chans=2, embed_dim=138, window_size=5, num_heads=6,
#          mlp_ratio=2.0,
#          qkv_bias=True,
#          qk_scale=None,
#          drop_rate=0.1,
#          attn_drop_rate=0.0,
#          drop_path_rate=0.1,
#          norm_layer=(nn.LayerNorm),
#          patch_norm=True,
#          upscale=1,
#          img_range=1.0 ).to('cuda:1')                   
          
        self.lam = nn.Parameter(torch.Tensor([0.8]))
        self.mu = nn.Parameter(torch.Tensor([0.05]))
        self.miu1 = nn.Parameter(torch.Tensor([0.05])) #0.05
#        self.miu2 = nn.Parameter(torch.Tensor([0.05])).to(device)
        

    def forward(self, xK, mask, atb, kernel, W):
#        print('xK device:', xK.device)
#        print('Trans_KL device:', next(self.Trans_KL.parameters()).device)
#        print('Trans_IL device:', next(self.Trans_IL.parameters()).device) 
        y = xK - self.miu1 * self.Trans_KL(xK*W)/W  - self.mu * c2r(adjspirit(kernel, spirit(kernel, r2c(xK)))) 
 #- self.miu2 * c2r(fft2c(r2c( self.Trans_IL(c2r(ifft2c(r2c(xK)))) )))
        xK = y - self.lam * mask * xK + self.lam * atb
        
        return xK        


class WBT(BaseNetwork):

    def __init__(self, LayerNo=10, image_size=(256, 232), channels=24):
        super(WBT, self).__init__()
        self.LayerNo = LayerNo
        self.nch = 30 #30#2
        
        Layers = []
        for i in range(LayerNo):
            if i%2 == 1:
                Layers.append(WBTBlocknW(image_size, self.nch))
            elif i%2 == 0:
                Layers.append(WBTBlocknL(image_size, self.nch))                             
        
        self.fcs = nn.ModuleList(Layers)

    def forward(self, x, mask, atb, kernel, W=None): # 
#    def forward(self, x, mask, kernel, W=None): #    
        #print('k', kernel.shape)
        #atb = x
        for i in range(self.LayerNo):
#            if i%2 == 0:
#                x = self.fcs[i]( x, mask, atb, kernel, W)
#            elif i%2 == 1:
            x = self.fcs[i]( x, mask, atb, kernel, W)
        
        return x

