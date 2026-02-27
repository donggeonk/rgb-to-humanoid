"""
HMR2 Feature Extractor - Self-contained inference-only implementation.
Extracts 1024-dim features from cropped person images.
Zero imports from hmr4d. Only standard PyTorch + timm + einops + cv2 + numpy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import os
from inspect import isfunction
from functools import partial
from typing import Optional
from einops import rearrange, repeat
from timm.layers import drop_path, trunc_normal_

# ============ Image Preprocessing Constants ============ #

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# ============ Geometry Utils ============ #


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


# ============ Helper Utilities ============ #


def _to_2tuple(x):
    """Convert to 2-tuple (replaces timm's to_2tuple)."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ============ ViT Backbone ============ #


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = (
            F.interpolate(
                abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(B, -1, C)
        )
    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTAttention(nn.Module):
    """Attention module for the ViT backbone (distinct from Transformer decoder Attention)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = ViTAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )

        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        )
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        use_checkpoint=False,
        frozen_stages=-1,
        ratio=1,
        last_norm=True,
        patch_padding="pad",
        freeze_attn=False,
        freeze_ffn=False,
    ):
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ratio=ratio,
        )
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()


# ============ Transformer Decoder Components ============ #


class AdaptiveLayerNorm1D(nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = nn.LayerNorm(data_dim)
        self.linear = nn.Linear(norm_cond_dim, 2 * data_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])
        return x * (1 + alpha) + beta


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return nn.BatchNorm1d(dim)
    elif norm == "layer":
        return nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Self-attention for Transformer decoder (distinct from ViTAttention)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v]
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(
                f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})"
            )

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = nn.Dropout(
                emb_dropout
            )  # ZeroTokenDropout equivalent for inference
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x


# ============ SMPL Head ============ #


class SMPLTransformerDecoderHead(nn.Module):
    """Cross-attention based SMPL Transformer decoder.
    In feat_mode, only returns the 1024-dim token features."""

    def __init__(
        self,
        joint_rep_type="6d",
        num_body_joints=23,
        transformer_input="zero",
        ief_iters=1,
        init_decoder_xavier=False,
        mean_params_path=None,
        transformer_decoder_kwargs=None,
    ):
        super().__init__()
        self.joint_rep_type = joint_rep_type
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (num_body_joints + 1)
        self.npose = npose
        self.num_body_joints = num_body_joints
        self.ief_iters = ief_iters
        self.input_is_mean_shape = transformer_input == "mean_shape"

        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        if transformer_decoder_kwargs is not None:
            transformer_args.update(transformer_decoder_kwargs)
        self.transformer = TransformerDecoder(**transformer_args)

        dim = transformer_args["dim"]
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if init_decoder_xavier:
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if mean_params_path is not None:
            mean_params = np.load(mean_params_path)
            init_body_pose = torch.from_numpy(
                mean_params["pose"].astype(np.float32)
            ).unsqueeze(0)
            init_betas = torch.from_numpy(
                mean_params["shape"].astype("float32")
            ).unsqueeze(0)
            init_cam = torch.from_numpy(
                mean_params["cam"].astype(np.float32)
            ).unsqueeze(0)
        else:
            # Default zeros if no mean params provided
            init_body_pose = torch.zeros(1, npose)
            init_betas = torch.zeros(1, 10)
            init_cam = torch.zeros(1, 3)

        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, only_return_token_out=False):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = rearrange(x, "b c h w -> b (h w) c")

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        if self.joint_rep_type == "aa":
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.ief_iters):
            assert i == 0, "Only support 1 iteration for now"

            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[
                    :, None, :
                ]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

            if only_return_token_out:
                return token_out
            else:
                # Readout from token_out
                pred_body_pose = self.decpose(token_out) + pred_body_pose
                pred_betas = self.decshape(token_out) + pred_betas
                pred_cam = self.deccam(token_out) + pred_cam
                pred_body_pose_list.append(pred_body_pose)
                pred_betas_list.append(pred_betas)
                pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]

        pred_smpl_params_list = {}
        pred_smpl_params_list["body_pose"] = torch.cat(
            [
                joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :]
                for pbp in pred_body_pose_list
            ],
            dim=0,
        )
        pred_smpl_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.num_body_joints + 1, 3, 3
        )

        pred_smpl_params = {
            "global_orient": pred_body_pose[:, [0]],
            "body_pose": pred_body_pose[:, 1:],
            "betas": pred_betas,
        }
        return pred_smpl_params, pred_cam, pred_smpl_params_list, token_out


# ============ HMR2 Model ============ #


class HMR2(nn.Module):
    """HMR2 model for feature extraction.
    In feat_mode=True (default), returns (B, 1024) feature tokens.
    """

    def __init__(self, mean_params_path=None):
        super().__init__()
        # ViT backbone: ViTPose-H config
        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.55,
        )

        # SMPL head with transformer decoder
        # Config from model_config.yaml
        transformer_decoder_kwargs = dict(
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=1280,
        )

        self.smpl_head = SMPLTransformerDecoderHead(
            joint_rep_type="6d",
            num_body_joints=23,
            transformer_input="zero",
            ief_iters=1,
            init_decoder_xavier=False,
            mean_params_path=mean_params_path,
            transformer_decoder_kwargs=transformer_decoder_kwargs,
        )

    def forward(self, batch, feat_mode=True):
        """
        Args:
            batch: dict with key "img" of shape (B, 3, 256, 256)
            feat_mode: if True (default), returns (B, 1024) feature tokens;
                       if False, returns full SMPL prediction dict.
        """
        # Backbone: crop 256x256 -> 256x192
        x = batch["img"][:, :, :, 32:-32]
        vit_feats = self.backbone(x)

        # Output head
        if feat_mode:
            token_out = self.smpl_head(
                vit_feats, only_return_token_out=True
            )  # (B, 1024)
            return token_out

        # Full process
        pred_smpl_params, pred_cam, _, token_out = self.smpl_head(
            vit_feats, only_return_token_out=False
        )
        output = {}
        output["token_out"] = token_out
        output["pred_smpl_params"] = pred_smpl_params
        output["pred_cam"] = pred_cam
        return output


# ============ Image Preprocessing ============ #


def crop_and_resize(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Crop and resize image based on bounding box center and size.

    Args:
        img: (H, W, 3) numpy array, BGR or RGB image
        bbx_xy: (2,) numpy array, bounding box center [x, y]
        bbx_s: scalar, bounding box size
        dst_size: output image size (square)
        enlarge_ratio: enlargement ratio for the bounding box

    Returns:
        img_crop: (dst_size, dst_size, 3) cropped and resized image
        bbx_xys_final: (3,) numpy array [cx, cy, size_after_enlarge]
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack(
        [
            bbx_xy - hs,  # left-up corner
            bbx_xy + np.array([hs, -hs]),  # right-up corner
            bbx_xy,  # center
        ]
    ).astype(np.float32)
    dst = np.array(
        [[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]],
        dtype=np.float32,
    )
    A = cv2.getAffineTransform(src, dst)

    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return img_crop, bbx_xys_final


def preprocess_image(img_crop, device="cuda"):
    """
    Preprocess a cropped image (256x256 BGR uint8) to a normalized tensor for HMR2.

    Args:
        img_crop: (256, 256, 3) numpy array, BGR uint8 image from crop_and_resize
        device: torch device

    Returns:
        img_tensor: (1, 3, 256, 256) normalized float tensor
    """
    # BGR to RGB, uint8 to float, HWC to CHW
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)

    # Normalize with ImageNet mean/std
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0).to(device)


# ============ Factory ============ #


def load_hmr2(ckpt_path=None, device="cuda"):
    """
    Load HMR2 model for feature extraction.

    Args:
        ckpt_path: path to checkpoint file. If None, uses default path
                   relative to this file's directory.
        device: torch device string

    Returns:
        model: HMR2 model in eval mode on specified device
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "checkpoints",
            "hmr2",
            "epoch=10-step=25000-001.ckpt",
        )

    # Find mean params: look alongside this file first, then in GVHMR source
    mean_params_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "smpl_mean_params.npz"
    )
    if not os.path.exists(mean_params_path):
        # Try the GVHMR source location
        alt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "GVHMR",
            "hmr4d",
            "network",
            "hmr2",
            "configs",
            "smpl_mean_params.npz",
        )
        if os.path.exists(alt_path):
            mean_params_path = alt_path
        else:
            # The model can still work without mean params (uses zeros),
            # but weights from checkpoint will override anyway
            mean_params_path = None

    # Build model
    # Hardcoded config from model_config.yaml:
    # ViT: embed_dim=1280, depth=32, num_heads=16
    # SMPL Head: transformer_decoder depth=6, dim=1024
    # Image size: 256x192 (but input is 256x256, cropped to 256x192 in forward)
    model = HMR2(mean_params_path=mean_params_path)

    # Load checkpoint weights
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # Filter to only backbone and smpl_head keys
    keys = [
        k for k in state_dict.keys() if k.split(".")[0] in ["backbone", "smpl_head"]
    ]
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model
