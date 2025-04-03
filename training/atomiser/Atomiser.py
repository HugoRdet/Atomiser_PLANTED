from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce




def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


def masking(tokens,attention_mask,percent):
    

    masked_tokens=int(tokens.shape[1]*(percent/100.))
    
    idx = torch.randperm(tokens.size(1))
    tokens=tokens[:,idx]
    attention_mask=attention_mask[:,idx]
    return tokens[:,masked_tokens:],attention_mask[:,masked_tokens:]



class Atomiser(nn.Module):
    def __init__(
        self,
        *,
        config,
        depth,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        lattent_attn_depth = 0,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        self_per_cross_attn = 1,
        final_classifier_head = True,
        masking=0,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.masking=masking
        self.config=config

        
        #fourier_channels = (input_axis * ((self.num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        shape_input_dim_x = self.get_shape_attributes_config("pos")
        shape_input_dim_y = self.get_shape_attributes_config("pos")

        query_num_freq_bands = config["Atomiser"]["query_num_freq_bands"]
        query_max_freq = config["Atomiser"]["query_max_freq"]

        
        shape_input_year = self.get_shape_attributes_config("year")
        shape_input_day = self.get_shape_attributes_config("day")
        shape_input_wavelength=self.get_shape_attributes_config("wavelength")
        shape_input_bandvalue=self.get_shape_attributes_config("bandvalue")
        self.nb_classes=config["dataset"]["classes"]

        input_dim=shape_input_dim_x+shape_input_dim_y+shape_input_year+shape_input_day+shape_input_wavelength+shape_input_bandvalue
        self.coordinates_start_wl=shape_input_bandvalue
        self.coordinates_end_wl=shape_input_bandvalue+shape_input_wavelength


        
        
        
        # Initialize your parameter
        self.encode_bands_no_wavelength = nn.Parameter(torch.empty(shape_input_wavelength))

        # Apply truncated normal initialization with specified parameters
        nn.init.trunc_normal_(self.encode_bands_no_wavelength, mean=0.0, std=0.02, a=-2.0, b=2.0)
        queries_tensor=get_positional_processing(512, None, query_max_freq, query_num_freq_bands)
        plt.figure()
        plt.imshow(queries_tensor[:100])

  

        self.register_buffer("queries", queries_tensor)




        #https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        #weights are shared except for the first blocs of cross attention / latent attention
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.lattent_attn_layers = nn.ModuleList([])

        for i in range(lattent_attn_depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.lattent_attn_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

     

        queries_dim=self.get_shape_attributes_config("query")*2
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim))
        
        self.tmp_Relu=nn.ReLU()
        self.to_logits=nn.Linear(queries_dim, 256)
        self.to_logits_1=nn.Linear(256, self.nb_classes)

       


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
        
                
        




    def forward(self,data,tokens_mask = None,attention_mask=None,return_embeddings = False):
        
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype

        
        

        x = repeat(self.latents, 'n d -> b n d', b = b)

        
        #for idx_channel in range(data.shape[0]):
        #    data[idx_channel,tokens_mask[idx_channel]==1,self.coordinates_start_wl:self.coordinates_end_wl]=self.encode_bands_no_wavelength
        
     


        queries=self.queries.clone()
        
        queries=repeat(queries,"l f -> b l f",b=data.shape[0])
        
        #latents = self.decoder_cross_attn(queries, context = x)
        latents =self.decoder_ff(queries)
        
    
        logits=self.to_logits(latents).relu()
        logits=self.to_logits_1(logits)
        logits=rearrange(logits,"b (h w) l-> b l h w ",h=512,w=512)


        return logits