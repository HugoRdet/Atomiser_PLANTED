import torch
from .utils_dataset import read_yaml,save_yaml
from .image_utils import *
from .files_utils import*
from math import pi
import einops 
import datetime
import numpy as np

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


class transformations_config_flair:

    

    def __init__(self,bands_yaml,config):
      
        self.bands_sen_infos=read_yaml(bands_yaml)["bands_sen_info"]
        self.bands_aer_infos=read_yaml(bands_yaml)["bands_aer_info"]
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
        
        
        self.encoded_fourier_cc=dict()
        self.encoded_fourier_scalar=dict()
        self.encoded_fourier_wavength=dict()
        self.dico_group_channels=dict()
        
        

    def get_shape_attributes_config(self,attribute):
        if self.config["Atomiser"][attribute+"_encoding"]=="NOPE":
            return 0
        if self.config["Atomiser"][attribute+"_encoding"]=="NATURAL":
            return 1
        if self.config["Atomiser"][attribute+"_encoding"]=="FF":
            if self.config["Atomiser"][attribute+"_num_freq_bands"]==-1:
                return int(self.config["Atomiser"][attribute+"_max_freq"])*2+1
            return int(self.config["Atomiser"][attribute+"_num_freq_bands"])*2+1

        

        




    def get_band_identifier(self,bands,channel_idx):
        for band_key in bands:
            band=bands[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None

    
    def get_band_infos(self,bands,band_identifier):
        return bands[band_identifier]



    
    def pos_encoding(self,size,positional_scaling=None,max_freq=4,num_bands = 4):
        if positional_scaling==None:
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size), (size,size)))
        else:
            axis_pos = list(map(lambda size: torch.linspace(-positional_scaling/2.0, positional_scaling/2.0, steps=size), (size,size)))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
        
        pos=fourier_encode(pos,max_freq=max_freq,num_bands = num_bands)
    
        pos=einops.rearrange(pos,"h w c f -> (c f) h w")
        return pos
    
    
    def get_positional_processing(self,img,resolution):
    

        
        size=img.shape[2]
        

        positional_scaling=None
        if resolution!=None:
            positional_scaling=(size * resolution)/400.0

        
        if size in self.encoded_fourier_cc:
            return self.encoded_fourier_cc[size]
        else:
         
            max_freq=self.config["Atomiser"]["pos_max_freq"]
            num_bands=self.config["Atomiser"]["pos_num_freq_bands"]
            

            encoding=self.pos_encoding(size,positional_scaling=positional_scaling,max_freq=max_freq,num_bands = num_bands)
            #tensor size : h w c with c the number of fourier features
            #low pass filter  

            #if ((encoding_x_s.shape[-1]-1)/2.0)>size:
            #    
            #    dim_max=int(size/2.0)
            #    features_dim=int((encoding_x_s.shape[-1]-1)/2.0)

            #    encoding_x_s[:,:,dim_max:features_dim]=0.0
            #    encoding_x_s[:,:,features_dim+dim_max:-1]=0.0
            #    encoding_y_s[:,:,dim_max:features_dim]=0.0
            #    encoding_y_s[:,:,features_dim+dim_max:-1]=0.0
                
               

            

            #res=torch.stack((encoding_x_s,encoding_y_s),dim=-1)
            #res=einops.rearrange(res,"h w c s -> (c s) h w")
            
            self.encoded_fourier_cc[size]=encoding
            return encoding





    def fourier_encode_scalar(self,scalar,size,max_freq,num_bands):
        
        tmp_encoding=fourier_encode(torch.Tensor([scalar]), max_freq=max_freq, num_bands = num_bands)
        
        tmp_encoding=tmp_encoding.unsqueeze(0)
        
        tmp_encoding=tmp_encoding.repeat(size,size,1)
        tmp_encoding=einops.rearrange(tmp_encoding,"h w c -> c h w")
        return tmp_encoding
    
    def scaling_frequencies(self,x):
        return (1000/x)-1.5
    

    def compute_gaussian_band_max_encoding(self,lambda_center, bandwidth, num_points=50):
        """
        Computes the max activation of Gaussians over a given spectral band using PyTorch.

        Parameters:
        - lambda_center: Central wavelength of the spectral band.
        - bandwidth: Width of the spectral band.
        - centers: List or tensor of Gaussian centers.
        - stds: List or tensor of Gaussian standard deviations.
        - num_points: Number of points sampled within the bandwidth.
        - device: 'cpu' or 'cuda' for GPU acceleration.

        Returns:
        - PyTorch tensor with the max Gaussian activation for each Gaussian center.
        """
        # Convert to PyTorch tensors and move to the specified device
        lambda_min = lambda_center - bandwidth / 2
        lambda_max = lambda_center + bandwidth / 2
        sampled_lambdas = torch.linspace(lambda_min, lambda_max, num_points).view(-1, 1)


        # Compute Gaussian activations (vectorized)
        gaussians = torch.exp(-0.5 * ((sampled_lambdas - self.gaussian_means) / self.gaussian_stds) ** 2)

        # Max activation across the sampled wavelengths
        encoding = gaussians.max(dim=0).values  

        return encoding

    
    def get_bvalue_processing(self,img):
        if self.config["Atomiser"]["bandvalue_encoding"]=="NATURAL":
                return img
        
        elif self.config["Atomiser"]["bandvalue_encoding"]=="FF":
            num_bands=self.config["Atomiser"]["bandvalue_num_freq_bands"]
            max_freq=self.config["Atomiser"]["bandvalue_max_freq"]

            fourier_encoded_values=fourier_encode(img, max_freq, num_bands)
            fourier_encoded_values=einops.rearrange(fourier_encoded_values,"b h w c -> (b c) h w")
            return fourier_encoded_values

    def wavelength_processing(self,wavelength,bandwidth,img_size):

        if (int(wavelength),int(bandwidth),int(img_size)) in self.encoded_fourier_wavength:
            return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]
        
        
        if wavelength==-1:

            if self.config["Atomiser"]["wavelength_encoding"]=="NATURAL":
                self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]=(torch.zeros((1,img_size,img_size)),False)
                return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]
            elif self.config["Atomiser"]["wavelength_encoding"]=="FF":
                num_bands=self.config["Atomiser"]["wavelength_num_freq_bands"]
                self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]=(torch.zeros((num_bands*2+1,img_size,img_size)),False)
                return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]
            elif self.config["Atomiser"]["wavelength_encoding"]=="GAUSSIANS":
                num_bands=self.gaussian_means.shape[1]
                self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]= (torch.zeros((num_bands,img_size,img_size)),False)
                return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]
                
            
            
        
        if self.config["Atomiser"]["wavelength_encoding"]=="NATURAL":
            self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]= (torch.full((1, img_size, img_size), self.scaling_frequencies(wavelength)),True)
            return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]

        if self.config["Atomiser"]["wavelength_encoding"]=="FF":
            max_freq=self.config["Atomiser"]["wavelength_max_freq"]
            num_bands=self.config["Atomiser"]["wavelength_num_freq_bands"]
            self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]= (self.fourier_encode_scalar(self.scaling_frequencies(wavelength),img_size,max_freq,num_bands),True)
            return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]
        
        if self.config["Atomiser"]["wavelength_encoding"]=="GAUSSIANS":
            encoded=self.compute_gaussian_band_max_encoding(wavelength, bandwidth, num_points=50).unsqueeze(-1).unsqueeze(-1)
            encoded=einops.repeat(encoded,'d h w -> d (h h1) (w w1)',h1=img_size,w1=img_size)
            self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]= (encoded,True)
            return self.encoded_fourier_wavength[(int(wavelength),int(bandwidth),int(img_size))]
    

    def time_processing(self,date,img_size):
        year=date.year
        day_of_year = date.timetuple().tm_yday
        
        y_r=(2021-year)/4.0
        d_m=day_of_year/366.0

        if self.config["Atomiser"]["day_encoding"]=="NATURAL":
            y_r_encoding= torch.full((1, img_size, img_size), y_r)
            d_m_encoding= torch.full((1, img_size, img_size), d_m)
            raise("not implemented")

            

        if self.config["Atomiser"]["day_encoding"]=="FF":
            y_max_freq=self.config["Atomiser"]["year_max_freq"]
            y_num_bands=self.config["Atomiser"]["year_num_freq_bands"]
            d_max_freq=self.config["Atomiser"]["day_max_freq"]
            d_num_bands=self.config["Atomiser"]["day_num_freq_bands"]

            year_encoding=self.fourier_encode_scalar(y_r,img_size,y_max_freq,y_num_bands)
            day_encoding=self.fourier_encode_scalar(d_m,img_size,d_max_freq,d_num_bands)
            return  torch.cat([year_encoding,day_encoding],dim=0)
         


    def apply_transformations_aer(self,im_aerial,date_aerial):
        im_aerial = im_aerial.unsqueeze(dim=0)
        img_size=im_aerial.shape[2]

   
        token_masks=torch.ones((int(im_aerial.shape[1]*im_aerial.shape[2]*im_aerial.shape[3])))*-1
        
        token_masks_id=0
        tokens_list=[]

        #aerial images handling
        for index_channel in range(5):

            channel_id = self.get_band_identifier(self.bands_aer_infos,index_channel)
            
            infos = self.get_band_infos(self.bands_aer_infos,channel_id)
            date_obj = datetime.datetime.strptime(date_aerial, '%Y-%m-%d')
            time_processing=self.time_processing(date_obj,img_size)

            # Extract the channel and process it
            img_channel = im_aerial[:, index_channel]  # shape: (1, H, W)
            value_processed=self.get_bvalue_processing(img_channel)
       

            #positional encoding
            resolution_value=float(infos["resolution"])
            band_post_proc = self.get_positional_processing(img_channel,resolution_value )
   
            
            #wavelength encoding
            tmp_central_wavelength=float(infos["central_wavelength"])
            tmp_bandwidth=float(infos["bandwidth"])
            central_wavelength_processing,bool_verif=self.wavelength_processing(tmp_central_wavelength,tmp_bandwidth,img_size)
            
            if not bool_verif:
                token_masks[token_masks_id:token_masks_id+img_size*img_size:]=index_channel
        
            token = torch.cat(
                [value_processed,
                central_wavelength_processing,
                band_post_proc,
                time_processing],
                dim=0
            ).unsqueeze(0)
            tokens_list.append(token)
            token_masks_id+=img_size*img_size

        # Concatenate tokens from all channels at once
        tokens = torch.cat(tokens_list, dim=0)
        tokens = einops.rearrange(tokens, "b c h w -> (b h w) c")

        return tokens,token_masks
    
    

    
    
    def apply_transformations_sen(self,im_sen,date_sen):
        im_sen = im_sen.unsqueeze(dim=0)
        img_size=im_sen.shape[2]

        tokens_list = []


        token_masks=torch.ones((im_sen.shape[0]*im_sen.shape[1]*im_sen.shape[2]*im_sen.shape[3]))*-1
        
        token_masks_id=0
        

        #aerial images handling
        for index_channel in range(len(self.bands_sen_infos.keys())):

            channel_id = self.get_band_identifier(self.bands_sen_infos,index_channel)
            infos = self.get_band_infos(self.bands_sen_infos,channel_id)

            time_processing=self.time_processing(date_sen,img_size)
        
            # Extract the channel and process it
            img_channel = im_sen[:, index_channel]  # shape: (1, H, W)
            value_processed=self.get_bvalue_processing(img_channel)
       

            #positional encoding
            resolution_value=float(infos["resolution"])
            band_post_proc = self.get_positional_processing(img_channel,resolution_value )
         
            #wavelength encoding
            tmp_central_wavelength=float(infos["central_wavelength"])
            tmp_bandwidth=float(infos["bandwidth"])
            central_wavelength_processing,bool_verif=self.wavelength_processing(tmp_central_wavelength,tmp_bandwidth,img_size)
            
            if not bool_verif:
                token_masks[token_masks_id:token_masks_id+img_size*img_size:]=index_channel

            # Concatenate all components for this channel in one go
            token = torch.cat(
                [value_processed,
                central_wavelength_processing,
                band_post_proc,
                time_processing],
                dim=0
            ).unsqueeze(0)
            tokens_list.append(token)
            token_masks_id+=im_sen.shape[1]*im_sen.shape[2]*im_sen.shape[3]

        # Concatenate tokens from all channels at once
        tokens = torch.cat(tokens_list, dim=0)
        tokens = einops.rearrange(tokens, "b c h w -> (b h w) c")
        

        return tokens,token_masks
    

    
        
        

        
        
    
    def apply_transformations_atomiser(self, data):

  

        im_aerial,mask,im_sen,date_sen,date_aerial=data
        tokens_masks=[]
        tokens,tmp_tokens_masks=self.apply_transformations_aer(im_aerial,date_aerial)
        tokens_masks.append(tmp_tokens_masks)
        L_tokens=[tokens]

        if im_sen.shape[0]>12:
            im_sen=im_sen[:12]

        if False:
            for t in range(im_sen.shape[0]):
                tmp_y=int(date_sen[0][t].item())
                tmp_m=int(date_sen[1][t].item())
                tmp_d=int(date_sen[2][t].item())
    
    
    
                date_obj = datetime.date(tmp_y,tmp_m,tmp_d)
    
    
                tokens,tmp_tokens_masks=self.apply_transformations_sen(im_sen[t],date_obj)
                
                tokens_masks.append(tmp_tokens_masks)
                L_tokens.append(tokens)
            


        tokens = torch.cat(L_tokens, dim=0)
        
        tokens_masks = torch.cat(tokens_masks,dim=0)

        

        token_dim=tokens.shape[-1]

        nb_padding_tokens=self.nb_tokens_limit-tokens.shape[0]
        padding_mask=torch.ones(nb_padding_tokens)*-2
        padding_tokens=torch.zeros((nb_padding_tokens,token_dim))

        tokens = torch.cat([tokens,padding_tokens], dim=0)
        tokens_masks = torch.cat([tokens_masks,padding_mask],dim=0)

        

       
        attention_mask=torch.ones(tokens.shape[0],dtype=torch.bool)
        attention_mask[tokens_masks==-2]=0



        
        return tokens,tokens_masks,attention_mask
