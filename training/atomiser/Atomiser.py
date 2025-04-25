from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

import time



def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache=True, key=None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key is None:
            raise ValueError("When caching, you must provide a `key`")
        if key in cache:
            return cache[key]
        result = f(*args, key=key, **kwargs)  # key forwarded here
        cache[key] = result
        return result
    return cached_fn


def pruning(tokens, attention_mask, percent):
    N = tokens.size(1)
    n_mask = int(N * percent/100.)
    perm = torch.randperm(N, device=tokens.device)
    keep_idx = perm[n_mask:]              
    return tokens[:, keep_idx], attention_mask[:, keep_idx], keep_idx


class Atomiser(pl.LightningModule):
    def __init__(
        self,
        *,
        config,
        transform,
        depth: int,
        input_axis: int = 2,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 1000,
        latent_attn_depth: int = 0,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        masking: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        self.input_axis = input_axis
        self.masking = masking
        self.config = config
        self.transform = transform

        shape_input_year = self.get_shape_attributes_config("year")
        shape_input_day = self.get_shape_attributes_config("day")
        # Compute input dim from encodings
        dx = self.get_shape_attributes_config("pos")
        dy = self.get_shape_attributes_config("pos")
        dw = self.get_shape_attributes_config("wavelength")
        db = self.get_shape_attributes_config("bandvalue")
        input_dim = dx + dy + dw + db +shape_input_day + shape_input_year

        # Initialize spectral params
        self.VV = nn.Parameter(torch.empty(dw))
        self.VH = nn.Parameter(torch.empty(dw))
        nn.init.trunc_normal_(self.VV, std=0.02, a=-2., b=2.)
        nn.init.trunc_normal_(self.VH, std=0.02, a=-2., b=2.)

        # Latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)

        get_cross_attn = cache_fn(lambda key=None: PreNorm(
            latent_dim,
            CrossAttention(
                query_dim   = latent_dim,
                context_dim = input_dim,
                heads       = cross_heads,
                dim_head    = cross_dim_head,
                dropout     = attn_dropout,
                use_flash   = True
            ),
            context_dim = input_dim
        ) if key is not None else None)  

        get_cross_ff = cache_fn(lambda key=None: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout)
        ) if key is not None else None)  

        get_latent_attn = cache_fn(lambda key=None: PreNorm(
            latent_dim,
            SelfAttention(
                dim        = latent_dim,
                heads      = latent_heads,
                dim_head   = latent_dim_head,
                dropout    = attn_dropout,
                use_flash  = True
            )
        ) if key is not None else None)  

        get_latent_ff = cache_fn(lambda key=None: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout)
        ) if key is not None else None)  

        # Build cross/self-attn layers
        self.layers = nn.ModuleList()
        for i in range(depth):
            cache_args = {'_cache': (i>0 and weight_tie_layers), 'key': i}
            # cross
            cross_attn = get_cross_attn(**cache_args)
            cross_ff   = get_cross_ff(**cache_args)
            # self
            self_attns = nn.ModuleList()
            for j in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**{'_cache':(j>0 and weight_tie_layers),'key':j}),
                    get_latent_ff(**{'_cache':(j>0 and weight_tie_layers),'key':j})
                ]))
            self.layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

        # Additional latent-only layers
        self.latent_attn_layers = nn.ModuleList()
        for i in range(latent_attn_depth):
            cache_args = {'_cache':(i>0 and weight_tie_layers), 'key':i}
            self.latent_attn_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # Classifier
        if final_classifier_head:
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        else:
            self.to_logits = nn.Identity()


    def get_shape_attributes_config(self,attribute):
        if self.config["Atomiser"][attribute+"_encoding"]=="NOPE":
            return 0
        if self.config["Atomiser"][attribute+"_encoding"]=="NATURAL":
            return 1
        if self.config["Atomiser"][attribute+"_encoding"]=="FF":
            if self.config["Atomiser"][attribute+"_num_freq_bands"]==-1:
                return int(self.config["Atomiser"][attribute+"_max_freq"])*2+1
            else:
                return int(self.config["Atomiser"][attribute+"_num_freq_bands"])*2+1
        
        if self.config["Atomiser"][attribute+"_encoding"]=="GAUSSIANS":
            return int(len(self.config["wavelengths_encoding"].keys()))
        

    def get_tokens(self,img,date,mask,mode="optique",modality="s2",wave_encoding=None):
        
  

        if mode=="optique":
            return self.transform.apply_transformations_optique(img,date,mask,modality)
        if mode=="sar":
            return self.transform.apply_transformations_SAR(img,date,mask,modality,wave_encoding=wave_encoding)
                
    def process_data(self,batch):
        L_tokens=[]
        L_masks=[]
        
        img_s2,img_l7,img_mo,img_s1,img_al,date_s2,date_l7,date_mo,date_s1,date_al,mask_s2,mask_l7,mask_mo,mask_s1,mask_al = batch
        
        if self.config["dataset"]["S2"]:
            tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_s2, mask_s2)
            tokens_s2,tokens_mask_s2=self.get_tokens(tmp_img,date_s2,tmp_mask,mode="optique",modality="s2")
            L_masks.append(tokens_mask_s2)
            L_tokens.append(tokens_s2)





        
        if self.config["dataset"]["L7"]:
            tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_l7, mask_l7)
            tokens_l7,tokens_mask_l7=self.get_tokens(tmp_img,date_l7,tmp_mask,mode="optique",modality="l7")
            L_masks.append(tokens_mask_l7)
            L_tokens.append(tokens_l7)



        if self.config["dataset"]["MODIS"]:
            tokens_mo,tokens_mask_mo=self.get_tokens(img_mo,date_mo,mask_mo,mode="optique",modality="modis")
            L_masks.append(tokens_mask_mo)
            L_tokens.append(tokens_mo)


        if self.config["dataset"]["S1"]:
            tokens_s1,tokens_mask_s1=self.get_tokens(img_s1,date_s1,mask_s1,mode="sar",modality="s1",wave_encoding=(self.VV,self.VH))
            L_masks.append(tokens_mask_s1)
            L_tokens.append(tokens_s1)


        if self.config["dataset"]["ALOS"]:
            tmp_img,tmp_mask=self.transform.apply_temporal_spatial_transforms(img_al, mask_al)
            tokens_al,tokens_mask_al=self.get_tokens(tmp_img,date_al,tmp_mask,mode="sar",modality="alos",wave_encoding=(self.VV,self.VH))
            L_masks.append(tokens_mask_al)
            L_tokens.append(tokens_al)
      



        tokens=torch.cat(L_tokens,dim=1)
        tokens_mask=torch.cat(L_masks,dim=1)


            
        return tokens,tokens_mask





    def forward(self, data, training=False):
        # Preprocess tokens + mask
        tokens, tokens_mask = self.process_data(data)
        b = tokens.shape[0]
        # initialize latents
        x = repeat(self.latents, 'n d -> b n d', b=b)
        # apply mask to tokens
        tokens_mask = tokens_mask.to(torch.bool)
        tokens = tokens.masked_fill(~tokens_mask.unsqueeze(-1), 0.)

        # cross & self layers
        for (cross_attn, cross_ff, self_attns) in self.layers:
            # optionally prune
            t, m = tokens, tokens_mask
            if self.masking > 0:
                t, m, idx = pruning(t, m, self.masking)
            # cross-attn
            x = cross_attn(x, context=t, mask=m) + x
            x = cross_ff(x) + x
            # restore tokens if pruned
            if self.masking > 0:
                tokens[:, idx] = t
                tokens_mask[:, idx] = m
            # self-attn blocks
            for (sa, ff) in self_attns:
                x = sa(x) + x
                x = ff(x) + x

        # latent-only layers
        for (sa, ff) in self.latent_attn_layers:
            x = sa(x) + x
            x = ff(x) + x

        # classifier
        return self.to_logits(x)