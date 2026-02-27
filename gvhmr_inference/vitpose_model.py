"""
ViTPose Model - Self-contained inference-only implementation.
Produces 2D keypoint heatmaps from cropped person images.

Zero imports from hmr4d. All architecture code is inlined here.
Supports ViTPose-S/B/H for COCO 256x192.
"""

import os
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
from timm.layers import drop_path, trunc_normal_


# ============ Utility ============ #


def to_2tuple(x):
    """Convert a scalar to a 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


# ============ ViT Backbone ============ #


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks)."""

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


class Attention(nn.Module):
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
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )

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
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
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
    """Vision Transformer backbone for ViTPose."""

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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

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

    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero,
            # it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp


# ============ Heatmap Head ============ #


class TopdownHeatmapSimpleHead(nn.Module):
    """Top-down heatmap simple head with deconv layers.

    Simplified for inference only - no loss computation, no training utilities.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=3,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        extra=None,
        in_index=0,
        input_transform=None,
        align_corners=False,
        upsample=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.upsample = upsample
        self.in_index = in_index
        self.align_corners = align_corners

        # Initialize input transform settings
        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

        if extra is not None and not isinstance(extra, dict):
            raise TypeError("extra should be dict or None.")

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers, num_deconv_filters, num_deconv_kernels
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f"num_deconv_layers ({num_deconv_layers}) should >= 0.")

        identity_final_layer = False
        if extra is not None and "final_conv_kernel" in extra:
            assert extra["final_conv_kernel"] in [0, 1, 3]
            if extra["final_conv_kernel"] == 3:
                padding = 1
            elif extra["final_conv_kernel"] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra["final_conv_kernel"]
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = (
                num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
            )

            layers = []
            if extra is not None:
                num_conv_layers = extra.get("num_conv_layers", 0)
                num_conv_kernels = extra.get("num_conv_kernels", [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2,
                        )
                    )
                    layers.append(nn.BatchNorm2d(conv_channels))
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = (
                f"num_layers({num_layers}) != length of num_filters({len(num_filters)})"
            )
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = (
                f"num_layers({num_layers}) != length of num_kernels({len(num_kernels)})"
            )
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder."""
        if not isinstance(inputs, list):
            if self.upsample > 0:
                inputs = F.interpolate(
                    F.relu(inputs),
                    scale_factor=self.upsample,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            return inputs

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


# ============ ViTPose Model ============ #


class ViTPoseModel(nn.Module):
    """Combined ViT backbone + heatmap head for pose estimation.

    Weight keys match the original checkpoint structure:
      backbone.*  -> ViT backbone
      keypoint_head.*  -> TopdownHeatmapSimpleHead
    """

    def __init__(self, backbone, keypoint_head):
        super(ViTPoseModel, self).__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 192) cropped person images
        Returns:
            heatmaps: (B, 17, 64, 48) keypoint heatmaps
        """
        x = self.backbone(x)
        x = self.keypoint_head(x)
        return x


# ============ Factory ============ #


_VITPOSE_CONFIGS = {
    "s": dict(embed_dim=384, depth=12, num_heads=12, drop_path_rate=0.1),
    "b": dict(embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.3),
    "h": dict(embed_dim=1280, depth=32, num_heads=16, drop_path_rate=0.55),
}


def build_vitpose(size_or_ckpt: str, ckpt_path=None, device="cuda"):
    """Build ViTPose for COCO17 256x192 and load checkpoint.

    Args:
        size_or_ckpt: Model size ("s", "b", "h") or a checkpoint path.
            When a path is given, size is inferred and ckpt_path is ignored.
        ckpt_path: Explicit checkpoint path (used only when size_or_ckpt is a size letter).
        device: Device string.

    Input:  (B, 3, 256, 192)
    Output: (B, 17, 64, 48)
    """
    if size_or_ckpt in _VITPOSE_CONFIGS:
        size = size_or_ckpt
    else:
        # Treat as checkpoint path, infer size from filename
        ckpt_path = size_or_ckpt
        fname = os.path.basename(ckpt_path).lower()
        size = next((s for s in ("s", "b", "h") if f"-{s}" in fname or f"_{s}" in fname), None)
        if size is None:
            raise ValueError(f"Cannot infer ViTPose size from filename: {fname}. "
                             f"Pass size explicitly as first arg ('s', 'b', 'h').")

    cfg = _VITPOSE_CONFIGS[size]
    backbone = ViT(
        img_size=(256, 192), patch_size=16, ratio=1, use_checkpoint=False,
        mlp_ratio=4, qkv_bias=True, **cfg,
    )
    head = TopdownHeatmapSimpleHead(
        in_channels=cfg["embed_dim"], out_channels=17,
        num_deconv_layers=2, num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4), extra=dict(final_conv_kernel=1),
    )
    model = ViTPoseModel(backbone, head)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Filter out keys not in the model (e.g. cls_token in ViTPose-S)
        model_keys = set(model.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model
