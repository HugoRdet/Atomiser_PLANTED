import torch
from .utils_dataset import read_yaml,save_yaml
from .image_utils import *
from .files_utils import*
from math import pi
import einops 
import datetime
import numpy as np
import datetime
from torchvision.transforms.functional import rotate, hflip, vflip
import random
import torch.nn as nn
import time

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    if num_bands==-1:
        scales = torch.linspace(1., max_freq , max_freq, device = device, dtype = dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)

    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]

    x = x * scales * pi
        
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class transformations_config_flair(nn.Module):

    

    def __init__(self,bands_yaml,config):
        super().__init__()
      
        self.bands_yaml=read_yaml(bands_yaml)
        self.bands_sen2_infos=self.bands_yaml["bands_sen2_info"]
        self.s2_waves=self.get_wavelengths_infos(self.bands_sen2_infos)
        self.s2_res=self.get_resolutions_infos(self.bands_sen2_infos)
        self.register_buffer("positional_encoding_s2", None)
        self.register_buffer('wavelength_encoding_s2', None)
        

        self.bands_l7_infos=self.bands_yaml["bands_l7_info"]
        self.l7_waves=self.get_wavelengths_infos(self.bands_l7_infos)
        self.l7_res=self.get_resolutions_infos(self.bands_l7_infos)
        self.register_buffer("positional_encoding_l7", None)
        self.register_buffer('wavelength_encoding_l7', None)

        self.bands_modis_infos=self.bands_yaml["bands_modis_info"]
        self.mo_waves=self.get_wavelengths_infos(self.bands_modis_infos)
        self.mo_res=self.get_resolutions_infos(self.bands_modis_infos)
        self.register_buffer("positional_encoding_modis", None)
        self.register_buffer('wavelength_encoding_modis', None)

        self.bands_sen1_infos=self.bands_yaml["bands_sen1_info"]
        self.register_buffer("positional_encoding_s1", None)

        self.bands_alos_infos=self.bands_yaml["bands_alos_info"]
        self.register_buffer("positional_encoding_alos", None)

  

        self.config=config
  
        self.nb_tokens_limit=config["trainer"]["max_tokens"]

        self.gaussian_means=[]
        self.gaussian_stds=[]

        if "wavelengths_encoding" in self.config:
            for gaussian_idx in self.config["wavelengths_encoding"]:
                self.gaussian_means.append(self.config["wavelengths_encoding"][gaussian_idx]["mean"])
                self.gaussian_stds.append(self.config["wavelengths_encoding"][gaussian_idx]["std"])
        
        self.gaussian_means=torch.Tensor(np.array(self.gaussian_means)).to(torch.float32).view(1, -1)
        self.gaussian_stds=torch.Tensor(np.array(self.gaussian_stds)).to(torch.float32).view(1, -1)
        
        
        
        
        

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
        

    def get_wavelengths_infos(self,bands_info):
        bandwidth=[]
        central_w=[]
        for band_name in bands_info:
            band=bands_info[band_name]
            bandwidth.append(band["bandwidth"])
            central_w.append(band["central_wavelength"])

        return np.array(bandwidth),np.array(central_w)
    
    def get_resolutions_infos(self,bands_info):
        res=[]
        for band_name in bands_info:
            band=bands_info[band_name]
            res.append(band["resolution"])


        return np.array(res)
   


        

        




    def get_band_identifier(self,bands,channel_idx):
        for band_key in bands:
            band=bands[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None

    
    def get_band_infos(self,bands,band_identifier):
        return bands[band_identifier]



    
    def pos_encoding(self,img_shape,size,positional_scaling_l=None,max_freq=4,num_bands = 4,device="cpu"):
        L_pos=[]
        if positional_scaling_l is None:
            positional_scaling_l=torch.ones(img_shape[-1])*2.0

        for idx in range(positional_scaling_l.shape[0]):
            positional_scaling=positional_scaling_l[idx]

            axis_pos = list(map(lambda size: torch.linspace(-positional_scaling/2.0, positional_scaling/2.0, steps=size,device=device), (size,size)))

            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            
            pos=fourier_encode(pos,max_freq=max_freq,num_bands = num_bands)

            pos=einops.rearrange(pos,"h w c f -> h w  (c f) ").unsqueeze(-2)
            L_pos.append(pos)

        


        return torch.cat(L_pos,dim=-2)
    
    
    def get_positional_processing(self,img_shape,resolution,T_size,B_size,modality,device):
    
        id_cache=f"positional_encoding_{modality}"
        encoded = getattr(self, id_cache)
        if  encoded is not None:
            
     
            encoded = encoded.expand(
                B_size,               # expand the batch‐axis
                encoded.size(1),      # T (time) stays the same
                encoded.size(2),      # H
                encoded.size(3),      # W
                encoded.size(4),      # C
                encoded.size(5)       # D
            )
            return encoded

        #b t h w c
        size=img_shape[-2]
        

        positional_scaling=None
        if not (resolution is None):
            positional_scaling=(size * resolution)/400.0
        
         
        max_freq=self.config["Atomiser"]["pos_max_freq"]
        num_bands=self.config["Atomiser"]["pos_num_freq_bands"]
        

        encoding=self.pos_encoding(img_shape,size,positional_scaling_l=positional_scaling,max_freq=max_freq,num_bands = num_bands,device=device).unsqueeze(0).unsqueeze(0)

        encoding = encoding.to(device)
        setattr(self,id_cache, encoding)

        encoding=einops.repeat(encoding,"b t h w c d -> (B b) (T t) h w c d",B=B_size,T=T_size)
        return encoding


    def apply_temporal_spatial_transforms(self, img, mask):
        """
        Apply random 90-degree rotation and flip to each item in a batch of 
        [B, T, H, W, C] images and masks. Rotation and flip are applied consistently 
        across all channels and time steps for each sample.
        
        Args:
            img (torch.Tensor): Input image tensor of shape [B, T, H, W, C]
            mask (torch.Tensor): Input mask tensor of shape [B, T, H, W, C]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and mask tensors
        """

        B, T, H, W, C = img.shape
        img = img.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        mask = mask.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

        img_out = img.clone()
        mask_out = mask.clone()

        for b in range(B):
            # --- Choose random 90-degree rotation ---
            k = random.randint(0, 3)  # rotate 0, 90, 180, or 270 degrees
            if k > 0:
                img_out[b] = torch.rot90(img_out[b], k=k, dims=(-2, -1))  # rotate over H-W
                mask_out[b] = torch.rot90(mask_out[b], k=k, dims=(-2, -1))

            # --- Random horizontal flip ---
            if random.random() > 0.5:
                img_out[b] = img_out[b].flip(-1)  # flip W
                mask_out[b] = mask_out[b].flip(-1)

            # --- Random vertical flip ---
            if random.random() > 0.5:
                img_out[b] = img_out[b].flip(-2)  # flip H
                mask_out[b] = mask_out[b].flip(-2)

        # Back to [B, T, H, W, C]
        img_out = img_out.permute(0, 2, 3, 4, 1)
        mask_out = mask_out.permute(0, 2, 3, 4, 1)

        return img_out, mask_out






    def fourier_encode_scalar(self,scalar,size,max_freq,num_bands):
        
        tmp_encoding=fourier_encode(torch.Tensor(scalar), max_freq=max_freq, num_bands = num_bands) # B T C
        
        tmp_encoding = tmp_encoding.unsqueeze(2).unsqueeze(2)  # [B, T, 1, 1, C]
        tmp_encoding = tmp_encoding.expand(-1,    # keep B
                                   -1,    # keep T
                                    size, # H = size
                                    size, # W = size
                                   -1)    # keep C
  
        return tmp_encoding
    
    def scaling_frequencies(self,x):
        return (1000/x)-1.5
    

    import torch

    def compute_gaussian_band_max_encoding(self, lambda_centers, bandwidths, num_points=50,modality="S2"):

        
        # Ensure inputs are PyTorch tensors and on the same device as self.gaussian_means.
        device = self.gaussian_means.device

        
        lambda_centers = torch.as_tensor(lambda_centers, dtype=torch.float32, device=device)
        bandwidths = torch.as_tensor(bandwidths, dtype=torch.float32, device=device)

        
        lambda_min = lambda_centers - bandwidths / 2 
        lambda_max = lambda_centers + bandwidths / 2 
        
        
        t = torch.linspace(0, 1, num_points, device=device)  
        
        # Compute the sampled wavelengths for each spectral band using broadcasting.
        # Each spectral band gets its own set of sample points.
        # sampled_lambdas shape: [s, num_points]

        sampled_lambdas = lambda_min.unsqueeze(1) + (lambda_max - lambda_min).unsqueeze(1) * t.unsqueeze(0)

        gaussians = torch.exp(
            -0.5 * (
                ((sampled_lambdas.unsqueeze(2) - self.gaussian_means.unsqueeze(0).unsqueeze(0)) / 
                self.gaussian_stds.unsqueeze(0).unsqueeze(0)) ** 2
            )
        )
        
        # For each spectral band and each Gaussian, find the maximum activation across the sampled points.
        # The max is taken along dim=1 (the num_points axis), returning a tensor of shape [s, num_gaussians].
      
        encoding = gaussians.max(dim=-2).values
       

        
        return encoding


    
    def get_bvalue_processing(self,img):
        if self.config["Atomiser"]["bandvalue_encoding"]=="NATURAL":
                return img.unsqueeze(-1)
        
        elif self.config["Atomiser"]["bandvalue_encoding"]=="FF":
            num_bands=self.config["Atomiser"]["bandvalue_num_freq_bands"]
            max_freq=self.config["Atomiser"]["bandvalue_max_freq"]
            fourier_encoded_values=fourier_encode(img, max_freq, num_bands)
            
            return fourier_encoded_values

    def wavelength_processing(self,device,wavelength,bandwidth,img_size,B_size,T_size,modality="s2"):

        id_cache=f"wavelength_encoding_{modality}"
        encoded = getattr(self, id_cache)
        if  encoded is not None :

            #encoded=einops.repeat(encoded,'b t h w c d  -> (B b) t h w c d ',B=B_size)

            encoded = encoded.expand(
                B_size,               # expand the batch‐axis
                encoded.size(1),      # T (time) stays the same
                encoded.size(2),      # H
                encoded.size(3),      # W
                encoded.size(4),      # C
                encoded.size(5)       # D
            )

            return encoded.to(device)
        
   


        
        if self.config["Atomiser"]["wavelength_encoding"]=="GAUSSIANS":
            encoded=self.compute_gaussian_band_max_encoding(wavelength, bandwidth, num_points=50).unsqueeze(0).unsqueeze(0).unsqueeze(0)
  
            encoded=einops.repeat(encoded,'b t h w c d  -> b (T t) (h h1) (w w1) c d ',T=T_size,h1=img_size,w1=img_size)

            encoded=encoded.to(device)
            setattr(self,id_cache, encoded)
       
            encoded=einops.repeat(encoded,'b t h w c d  -> (B b) t h w c d ',B=B_size)

            return encoded
    

    def time_processing(self,time_stamp,img_size=-1):
        dt = time_stamp.astype('datetime64[s]')

        years = dt.astype('datetime64[Y]').astype(int) + 1970
        norm_year = (years - 1999) / 27.0

        day_of_year = (dt - dt.astype('datetime64[Y]')) / np.timedelta64(1, 'D')
        norm_day = (day_of_year - 1) / 366.0
       
        return (norm_day.astype(np.float32),norm_year.astype(np.float32))
    
    
    def time_encoding(self,time_stamp,img_size=-1):
 
        norm_year = time_stamp[1]

       
        norm_day = time_stamp[0]
       
        


        y_max_freq=self.config["Atomiser"]["year_max_freq"]
        y_num_bands=self.config["Atomiser"]["year_num_freq_bands"]
        d_max_freq=self.config["Atomiser"]["day_max_freq"]
        d_num_bands=self.config["Atomiser"]["day_num_freq_bands"]

      

            

        if self.config["Atomiser"]["day_encoding"]=="FF":
            y_max_freq=self.config["Atomiser"]["year_max_freq"]
            y_num_bands=self.config["Atomiser"]["year_num_freq_bands"]
            d_max_freq=self.config["Atomiser"]["day_max_freq"]
            d_num_bands=self.config["Atomiser"]["day_num_freq_bands"]

            year_encoding=self.fourier_encode_scalar(norm_year,img_size,y_max_freq,y_num_bands)
            day_encoding=self.fourier_encode_scalar(norm_day,img_size,d_max_freq,d_num_bands)

            time_encoding=torch.cat([year_encoding,day_encoding],dim=-1)

            
            
            return  time_encoding
        return None
         


    def apply_transformations_optique(self, im_sen, dates_sen, mask_sen, mode):
        # --- select band info based on mode ---
        if mode=="s2":
            tmp_infos = self.bands_sen2_infos
            res = self.s2_res
            tmp_bandwidth, tmp_central_wavelength = self.s2_waves
        elif mode=="l7":
            tmp_infos = self.bands_l7_infos
            res = self.l7_res
            tmp_bandwidth, tmp_central_wavelength = self.l7_waves
        elif mode=="modis":
            tmp_infos = self.bands_modis_infos
            res = self.mo_res
            tmp_bandwidth, tmp_central_wavelength = self.mo_waves

        # handle singleton dims
        if im_sen.ndim == 4:
            im_sen = im_sen.unsqueeze(0).unsqueeze(0)
            mask_sen = mask_sen.unsqueeze(0).unsqueeze(0)

        B_size, T_size, H, W, C = im_sen.shape
        img_size = H

        # =========== TOTAL TIMER ===========
        torch.cuda.synchronize()
        t_total_start = time.time()

        # 1) Time encoding
        torch.cuda.synchronize()
        t0 = time.time()
        time_encoding = self.time_encoding(dates_sen, img_size).unsqueeze(-2)
        c1 = tmp_bandwidth.shape[0]
        time_encoding = time_encoding.expand(
            B_size, T_size, H, W, c1, -1
        )
        torch.cuda.synchronize()
        t1 = time.time()

        # 2) Wavelength encoding
        torch.cuda.synchronize()
        t2 = time.time()
        central_wavelength_processing = self.wavelength_processing(
            im_sen.device,
            tmp_central_wavelength,
            tmp_bandwidth,
            img_size,
            B_size,
            T_size,
            modality=mode
        )
        torch.cuda.synchronize()
        t3 = time.time()

        # 3) Band‑value encoding
        torch.cuda.synchronize()
        t4 = time.time()
        value_processed = self.get_bvalue_processing(im_sen)
        torch.cuda.synchronize()
        t5 = time.time()

        # 4) Positional encoding
        torch.cuda.synchronize()
        t6 = time.time()
        band_post_proc = self.get_positional_processing(
            im_sen.shape, res, T_size, B_size, mode, im_sen.device
        )
        torch.cuda.synchronize()
        t7 = time.time()

        # 5) Concat + rearrange
        torch.cuda.synchronize()
        t8 = time.time()

        print("cw ",central_wavelength_processing.shape)
        print("vp ",value_processed.shape)
        print("bp ",band_post_proc.shape)
        print("te ",time_encoding.shape)
        tokens = torch.cat([
            central_wavelength_processing.to(im_sen.device),
            value_processed.to(im_sen.device),
            band_post_proc.to(im_sen.device),
            time_encoding.to(im_sen.device)
        ], dim=5)
        tokens = einops.rearrange(tokens, "b t h w c f -> b (t h w c) f")
        token_masks = einops.rearrange(mask_sen, "b t h w c -> b (t h w c)")
        torch.cuda.synchronize()
        t9 = time.time()

        # =========== PRINT BREAKDOWN ===========
        print(f"[apply_optique] total {(t9-t_total_start):.3f}s | "
            f"time_enc {(t1-t0):.3f}s | "
            f"wl_enc {(t3-t2):.3f}s | "
            f"bv_enc {(t5-t4):.3f}s | "
            f"pos_enc {(t7-t6):.3f}s | "
            f"concat {(t9-t8):.3f}s")

        return tokens, token_masks

    

    def apply_transformations_SAR(self,im_sen,dates_sen,mask_sen,mode,wave_encoding=None):
        if mode=="s1":
            tmp_infos=self.bands_sen2_infos
            res=None
            tmp_bandwidth=None
        elif mode=="alos":
            tmp_infos=self.bands_l7_infos
            res=None
            tmp_bandwidth=None
        
        im_sen=im_sen[:,:,:,:,:-1]
        mask_sen=mask_sen[:,:,:,:,:-1]


     
        
        img_size=im_sen.shape[3]
        
        time_encoding=self.time_encoding(dates_sen,img_size).unsqueeze(-2)
        #time_encoding=einops.repeat(time_encoding,"b t h w c e -> b t h w (c1 c) e ",c1=im_sen.shape[-1])

        c1 = im_sen.shape[-1]
        time_encoding = time_encoding.expand(
            -1,   # B stays the same
            -1,   # T stays the same
            -1,   # H stays the same
            -1,   # W stays the same
            c1,  # expand the singleton c‐dimension
            -1    # E stays the same
        )  # now [B, T, H, W, c1, E]

        T_size=im_sen.shape[1]
        B_size=im_sen.shape[0]

            
        shape_input_wavelength=self.get_shape_attributes_config("wavelength")
        target_shape_w=(im_sen.shape[0],im_sen.shape[1],im_sen.shape[2],im_sen.shape[3],im_sen.shape[4],shape_input_wavelength)
        central_wavelength_processing=torch.empty(target_shape_w)
        
        
        
        if wave_encoding!=None:
            VV,VH=wave_encoding
            central_wavelength_processing[:,:,:,:,0].copy_(VV)
            central_wavelength_processing[:,:,:,:,1].copy_(VH)
               
        value_processed=self.get_bvalue_processing(im_sen)
        



        #positional encoding

        #get_positional_processing(self,img_shape,resolution,T_size,B_size,modality,device):
        band_post_proc = self.get_positional_processing(im_sen.shape,res,T_size,B_size,mode,im_sen.device )

   


        tokens=torch.cat([central_wavelength_processing.to(im_sen.device),
                          value_processed.to(im_sen.device),
                          band_post_proc.to(im_sen.device),
                          time_encoding.to(im_sen.device)         
                ],dim=5)
        
        

        tokens=einops.rearrange(tokens,"b t h w c f ->b  (t h w c) f")
        token_masks=mask_sen
        token_masks=einops.rearrange(mask_sen,"b t h w c -> b (t h w c)")



        

        

        
        

   
    

        return tokens,token_masks

  