# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Scale-MAE models."""

from collections import OrderedDict
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum
import pytorch_lightning as pl



def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int, grid_size: int, res: Tensor, cls_token: bool = False
) -> Tensor:
    """Generate spatial resolution specific 2D positional embeddings.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid_size: Height (ph) and width (pw) of the image patches.
        res: Spatial resolution tensor of shape (N,) of the image.
        cls_token: Increase positional embedding size by 1 for class token.

    Returns:
        pos_embed: Spatial resolution aware positional embeddings (Ph * Pw, D).
    """
    device, dtype = res.device, res.dtype
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid: Tensor = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    grid = torch.einsum('chw,n->cnhw', grid, res)
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], dtype=dtype, device=device), pos_embed],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim: int, grid: Tensor) -> Tensor:
    """Generate 2D sin-cos positional embedding from grid.

    Args:
        embed_dim: Dimension of the positional embeddings.
        grid: Tensor representing the image patch grid (C, N, Ph, Pw)

    Returns:
        emb: 2D sin-cos positional embeddings (Ph * Pw, D).
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Generate 1D sin-cos positional embedding from grid dimension.

    Args:
        embed_dim: Dimension of the positional embeddings.
        pos: Tensor of positions to be encoded (M,).

    Returns:
        emb: 1D sin-cos positional embeddings (M, D).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


class ScaleMAE(VisionTransformer):
    """Custom Vision Transformer for Scale-MAE with GSD positional embeddings.

    This is a ViT encoder only model of the Scale-MAE architecture with GSD positional embeddings.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2212.14532
    """

    def __init__(self, res: float = 1.0, *args: Any, **kwargs: Any) -> None:
        """Initialize a new ScaleMAE model.

        Args:
            res: Spatial resolution of the image in meters.
            *args: Additional arguments to
                pass to :class:`timm.models.vision_transformer.VisionTransformer`.
            **kwargs: Additional keyword arguments to
                pass to :class:`timm.models.vision_transformer.VisionTransformer`.
        """
        super().__init__(*args, **kwargs)

        self.res = res

        # Scale MAE uses resolution specific positional embeddings
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x) 

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def _pos_embed(self, x: Tensor) -> Tensor:
        """Apply GSD positional embeddings to the input tensor."""
        res = torch.tensor(self.res, dtype=x.dtype, device=x.device)
        res = res.repeat(x.shape[0])
        pos_embed = (
            get_2d_sincos_pos_embed_with_resolution(
                self.embed_dim,
                int(self.patch_embed.num_patches**0.5),
                res,
                cls_token=True,
            )
            .to(x.dtype)
            .to(x.device)
        )
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)
        return x



def interpolate_pos_embed(
    model: ScaleMAE, state_dict: OrderedDict[str, Tensor]
) -> OrderedDict[str, Tensor]:
    """Interpolate the positional embeddings if image size is different than pretrained image size.

    Args:
        model: ScaleMAE model.
        state_dict: Pretrained model state dict.

    Returns:
        state_dict: State dict with interpolated positional embeddings.
    """
    pos_embed_checkpoint = state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = 0
    if model.pos_embed is not None:
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed

    return state_dict



class CustomScaleMAE(pl.LightningModule):
    def __init__(self, num_classes: int = 40,transform=None):
        super().__init__()
        self.transform=transform
        self.encoder = ScaleMAE(
            img_size=12, patch_size=4, in_chans=6,
            embed_dim=768, depth=12, num_heads=12,
            num_classes=num_classes  # Configure the built-in head
        )
        # No need for self.to_logits since we're using the built-in head


    def process_data(self,batch):

        #L_tokens=[]
        #L_masks=[]
        
        img_s2,img_l7,img_mo,img_s1,img_al,date_s2,date_l7,date_mo,date_s1,date_al,mask_s2,mask_l7,mask_mo,mask_s1,mask_al = batch

        tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_s2, mask_s2)
        data_s2=(tmp_img,tmp_mask)

        tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_l7, mask_l7)
        data_l7=(tmp_img,tmp_mask)

        data_modis=(img_mo,mask_mo)
            
        return data_s2,data_l7,data_modis

    def forward(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        feats = self.encoder.forward_features(x)
        cls_feat = feats[:, 0]
        return self.encoder.head(cls_feat)