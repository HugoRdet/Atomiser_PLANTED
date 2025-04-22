import torch
import torch.nn as nn
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

        self.bands_l7_infos=self.bands_yaml["bands_l7_info"]
        self.l7_waves=self.get_wavelengths_infos(self.bands_l7_infos)
        self.l7_res=self.get_resolutions_infos(self.bands_l7_infos)

        self.bands_modis_infos=self.bands_yaml["bands_modis_info"]
        self.mo_waves=self.get_wavelengths_infos(self.bands_modis_infos)
        self.mo_res=self.get_resolutions_infos(self.bands_modis_infos)

        self.bands_sen1_infos=self.bands_yaml["bands_sen1_info"]
        self.bands_alos_infos=self.bands_yaml["bands_alos_info"]
        self.wavelengths_encoding_cache=dict()
        self.positional_encoding_cache=dict()

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
        
        
        self.register_buffer('pos_enc', None)
        for m in ['s2', 'l7', 'modis', 's1', 'alos']:
            self.register_buffer(f'wavelength_cache_{m}', None)
        
        

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
    
    def get_resolutions_infos(self, bands_info):
        return np.array([band["resolution"] for band in bands_info.values()])
   


        

        




    def get_band_identifier(self,bands,channel_idx):
        for band_key in bands:
            band=bands[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None

    
    def get_band_infos(self,bands,band_identifier):
        return bands[band_identifier]



    
    def pos_encoding(self, img_shape, size, positional_scaling_l=None, max_freq=4, num_bands=4):
        L_pos = []
        if positional_scaling_l is None:
            positional_scaling_l = torch.ones(img_shape[-1]) * 2.0
        for scale in positional_scaling_l:
            axis_pos = [torch.linspace(-scale/2, scale/2, steps=size, device='cpu') 
                        for _ in range(2)]
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            pos = fourier_encode(pos, max_freq=max_freq, num_bands=num_bands)
            pos = einops.rearrange(pos, "h w c f -> h w (c f)").unsqueeze(-2)
            L_pos.append(pos)
        return torch.cat(L_pos, dim=-2)
    
    
    def get_positional_processing(self, img_shape, resolution, T_size, B_size, modality, device):
        size = img_shape[-2]
        buffer_name = f"pos_enc_{size}"
        if hasattr(self, buffer_name) and getattr(self, buffer_name) is not None:
            encoding = getattr(self, buffer_name)
        else:
            pos_scale = None
            if resolution is not None:
                pos_scale = (size * resolution) / 400.0
            max_f = self.config["Atomiser"]["pos_max_freq"]
            n_b = self.config["Atomiser"]["pos_num_freq_bands"]
            enc = self.pos_encoding(img_shape, size, positional_scaling_l=pos_scale, max_freq=max_f, num_bands=n_b)
            # add batch/time dims
            encoding = enc.unsqueeze(0).unsqueeze(0).to(device)
            self.register_buffer(buffer_name, encoding)
        # expand to full batch
        full = einops.repeat(encoding, "b t h w c -> (B b) (T t) h w c", B=B_size, T=T_size)
        return full


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
        
        tmp_encoding=tmp_encoding.unsqueeze(2).unsqueeze(2)
        tmp_encoding=einops.repeat(tmp_encoding," B T h w C ->B T (s1 h) (s2 w) C",s1=size,s2=size)
  
        return tmp_encoding
    
    def scaling_frequencies(self,x):
        return (1000/x)-1.5
    

    import torch

    def compute_gaussian_band_max_encoding(self, lambda_centers, bandwidths, num_points=50, modality="S2"):
        device = self.gaussian_means.device
        lc = torch.as_tensor(lambda_centers, dtype=torch.float32, device=device)
        bw = torch.as_tensor(bandwidths, dtype=torch.float32, device=device)
        lam_min, lam_max = lc - bw/2, lc + bw/2
        t = torch.linspace(0, 1, num_points, device=device)
        sampled = lam_min.unsqueeze(1) + (lam_max - lam_min).unsqueeze(1) * t.unsqueeze(0)
        gauss = torch.exp(-0.5 * (((sampled.unsqueeze(2) - self.gaussian_means.unsqueeze(0).unsqueeze(0)) /
                                   self.gaussian_stds.unsqueeze(0).unsqueeze(0)) ** 2))
        return gauss.max(dim=1).values


    
    def get_bvalue_processing(self, img):
        mode = self.config["Atomiser"]["bandvalue_encoding"]
        if mode == "NATURAL":
            return img.unsqueeze(-1)
        if mode == "FF":
            nf = self.config["Atomiser"]["bandvalue_num_freq_bands"]
            mf = self.config["Atomiser"]["bandvalue_max_freq"]
            return fourier_encode(img, mf, nf)
        return img.unsqueeze(-1)

    def wavelength_processing(self, device, wavelength, bandwidth, img_size, B_size, T_size, modality="s2"):
        cache_attr = f"wavelength_cache_{modality}"
        cached = getattr(self, cache_attr, None)
        if isinstance(cached, torch.Tensor):
            enc = cached
        else:
            enc0 = self.compute_gaussian_band_max_encoding(wavelength, bandwidth, num_points=50)
            enc0 = enc0.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            enc = einops.repeat(enc0, 'b t h w c d -> b (T t) (h h1) (w w1) c d',
                                 T=T_size, h1=img_size, w1=img_size)
            enc = enc.to(device)
            self.register_buffer(cache_attr, enc)
        # expand over batch
        full = einops.repeat(enc, 'b t h w c d -> (B b) t h w c d', B=B_size)
        return full
    

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
         


    def apply_transformations_optique(self,im_sen,dates_sen,mask_sen,mode):
        if mode=="s2":
            tmp_infos=self.bands_sen2_infos
            res=self.s2_res
            tmp_bandwidth,tmp_central_wavelength=self.s2_waves
        elif mode=="l7":
            tmp_infos=self.bands_l7_infos
            res=self.l7_res
            tmp_bandwidth,tmp_central_wavelength=self.l7_waves
        elif mode=="modis":
            tmp_infos=self.bands_modis_infos
            res=self.mo_res
            tmp_bandwidth,tmp_central_wavelength=self.mo_waves

        if len(im_sen.shape)==4:
            im_sen = im_sen.unsqueeze(dim=0).unsqueeze(dim=0) # 1;1;8;12;12;3
            token_masks=mask_sen.unsqueeze(dim=0).unsqueeze(dim=0)# 1;1;8;12;12;3
     
        
        img_size=im_sen.shape[3]
        tokens_list = []
        #token_masks=mask_sen #T H W C
        #token_masks=einops.rearrange(mask_sen,"t h w c -> (t h w c)")

        
        

        

        time_encoding=self.time_encoding(dates_sen,img_size).unsqueeze(-2)
        time_encoding=einops.repeat(time_encoding,"b t h w c e -> b t h w (c1 c) e ",c1=tmp_bandwidth.shape[0])

        T_size=im_sen.shape[1]
        B_size=im_sen.shape[0]

        central_wavelength_processing=self.wavelength_processing(im_sen.device,tmp_central_wavelength,tmp_bandwidth,img_size,B_size,T_size,modality=mode)        
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
        token_masks=einops.rearrange(mask_sen,"b t h w c -> b (t h w c)")

   
    

        return tokens,token_masks
    

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
        time_encoding=einops.repeat(time_encoding,"b t h w c e -> b t h w (c1 c) e ",c1=im_sen.shape[-1])

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

  