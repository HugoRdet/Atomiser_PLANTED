import h5py
import os
import torch
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
import einops as einops
from torch.utils.data import Dataset,DataLoader,Sampler
import h5py
from tqdm import tqdm
from .image_utils import*
from .utils_dataset import*
import random
from .FLAIR_2 import*

def del_file(path):
    if os.path.exists(path):
        os.remove(path)


def filter_dates(mask, clouds:bool=2, area_threshold:float=0.5, proba_threshold:int=60):
    """ Mask : array T*2*H*W
        Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
        Area_threshold : threshold on the surface covered by the clouds / snow 
        Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)
        Return array of indexes to keep
    """
    dates_to_keep = []
    
    for t in range(mask.shape[0]):
        if clouds != 2:
            cover = np.count_nonzero(mask[t, clouds, :,:]>=proba_threshold)
        else:
            cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
        cover /= mask.shape[2]*mask.shape[3]
        if cover < area_threshold:
            dates_to_keep.append(t)

    return dates_to_keep


def monthly_image(sp_patch, sp_raw_dates):
    average_patch, average_dates = [], []
    month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),
                                  end=sp_raw_dates[-1].strftime('%Y-%m-%d'),
                                  freq='M')

    for m in month_range:
        month_dates = [i for i, date in enumerate(sp_raw_dates)
                       if date.month == m.month and date.year == m.year]

        if month_dates:
            average_patch.append(np.mean(sp_patch[month_dates], axis=0))
            average_dates.append(datetime.datetime(m.year, m.month, 1))

    return np.array(average_patch), average_dates


def create_dataset(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtd, name="tiny", mode="train", stats=None):
    """
    Creates an HDF5 dataset using the given sample indices (dico_idxs) from ds.
    If stats (per-channel mean/std) is None, computes it on-the-fly in a streaming fashion.
    Then applies normalization to each image: (image - mean) / std

    Args:
        dico_idxs (dict): Mapping from some key to a list of sample indices
        ds: A dataset that supports ds[idx] -> (image, label)
            where image is shape (12, 120, 120).
        name (str): HDF5 file prefix
        mode (str): e.g. "train" or "test"
        stats (torch.Tensor or None): shape (12,2) with [:,0] as mean, [:,1] as std

    Returns:
        stats (torch.Tensor): The per-channel mean/std used for normalization
    """

    # 1) Clean up any existing file
    h5_path = f'./data/custom_flair/{name}_{mode}.h5'
    del_file(h5_path)
  
    # 2) If stats is not given, compute it in a streaming fashion
    if stats is None:
        stats =compute_channel_mean_std(images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtd)

        



    # 3) Create a new HDF5 file
    db = h5py.File(h5_path, 'w')


    # 4) Iterate through your dictionary of IDs, fetch images, and store them
    for idx_img in range(len(images)):
        if idx_img>=1:
            continue
        
        im_aer,mask,sen_spatch,img_dates,sen_mask,aerial_date=get_sample(idx_img,images, labels, sentinel_images, centroids,sentinel_products,sentinel_masks,aerial_mtd, palette=lut_colors)
        
        

        # Convert to float (if needed) before normalization
        im_aer = im_aer.astype(float)
        sen_spatch=sen_spatch.astype(float)
        mask=mask.astype(int)


        # Apply per-channel normalization
        # normalized_value = (value - mean[channel]) / std[channel]
        
        #im_aer = (im_aer - stats["im_mean"][:, None, None]) / stats["im_std"][:, None, None]
        #sen_spatch = (sen_spatch - stats["sen_mean"][:, None, None]) / stats["sen_std"][:, None, None]


        to_keep = filter_dates(sen_mask, clouds=2, area_threshold=0.5, proba_threshold=60)
        sen_spatch = sen_spatch[to_keep]
        img_dates=img_dates[to_keep]


        sen_spatch, img_dates =monthly_image(sen_spatch, img_dates)

        days=[]
        months=[]
        years=[]
        for tmp_date in img_dates:
            tmp_day=tmp_date.day 
            tmp_month=tmp_date.month
            tmp_year=tmp_date.year
            days.append(tmp_day)
            months.append(tmp_month)
            years.append(tmp_year)


        


        im_aer = im_aer.astype(np.float32)
        sen_spatch = sen_spatch.astype(np.float32)
        days = np.array(days, dtype=np.float32)
        months = np.array(months, dtype=np.float32)
        years = np.array(years, dtype=np.float32)
        mask = mask.astype(np.float32)
        mask[mask>13]=13
        
        sen_mask = sen_mask.astype(np.float32)


 


        
        
        # Convert back to numpy to store in HDF5
        db.create_dataset(f'img_aerial_{idx_img}', data=im_aer)
        db.create_dataset(f'img_sen_{idx_img}', data=sen_spatch)
        db.create_dataset(f'days_{idx_img}', data=days)
        db.create_dataset(f'months_{idx_img}', data=months)
        db.create_dataset(f'years_{idx_img}', data=years)
        db.create_dataset(f'mask_{idx_img}',data=mask)
        db.create_dataset(f'sen_mask_{idx_img}',data=sen_mask)
        db.create_dataset(f'aerial_mtd_{idx_img}',data=aerial_date)
  


    db.close()


    
            








import random
import torch
from torchvision.transforms.functional import rotate, hflip, vflip

class CustomFLAIR(Dataset):
    def __init__(self, file_path, config=None, trans_config=None, augment=True):
        self.file_path = file_path
        self.num_samples = None
        self.config = config
        self.trans_config = trans_config
        self.augment = augment

        self._initialize_file()

    def _initialize_file(self):
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 8

    def __len__(self):
        return self.num_samples

    def set_modality_mode(self, mode):
        self.modality_mode = mode

    def reset_modality_mode(self):
        self.modality_mode = self.original_mode

    def process_mask(self, mask):
        mask = mask.float()
        mask[mask > 13] = 13
        mask = mask - 1
        return mask

    def apply_transforms(self, im_aerial, mask, im_sen):
        # Random rotation angle
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)

        # Apply rotation consistently
        im_aerial = rotate(im_aerial, angle)
        mask = rotate(mask.unsqueeze(0), angle).squeeze(0)  # mask is [H,W]

        # im_sen shape [T, C, H, W] -> rotate each channel identically
       
        for t in range(im_sen.shape[0]):  # 12 time points
      
            for c in range(im_sen.shape[1]):  # 10 channels
                
                rotated_channel = rotate(im_sen[t, c].unsqueeze(0), angle).squeeze(0)
                im_sen[t, c]=rotated_channel
            

        # Random horizontal flip
        if random.random() > 0.5:
            im_aerial = hflip(im_aerial)
            mask = hflip(mask.unsqueeze(0)).squeeze(0)
            im_sen = im_sen.flip(-1)

        # Random vertical flip
        if random.random() > 0.5:
            im_aerial = vflip(im_aerial)
            mask = vflip(mask.unsqueeze(0)).squeeze(0)
            im_sen = im_sen.flip(-2)

        return im_aerial, mask, im_sen

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            im_aerial = torch.tensor(f[f'img_aerial_{idx}'][:], dtype=torch.float32)  # [5,512,512]
            im_sen = torch.tensor(f[f'img_sen_{idx}'][:], dtype=torch.float32)  # [12,10,40,40]
            days = torch.tensor(f[f'days_{idx}'][:], dtype=torch.float32)
            months = torch.tensor(f[f'months_{idx}'][:], dtype=torch.float32)
            years = torch.tensor(f[f'years_{idx}'][:], dtype=torch.float32)
            mask = torch.tensor(f[f'mask_{idx}'][:], dtype=torch.float32)  # [512,512]
            sen_mask = f[f'sen_mask_{idx}'][:]
            aerial_date = f[f'aerial_mtd_{idx}'].asstr()[()]

            #if self.augment:
            #    im_aerial, mask, im_sen = self.apply_transforms(im_aerial, mask, im_sen)

            data = (im_aerial, mask, im_sen, (years, months, days), aerial_date)

            tokens, tokens_mask, attention_mask = self.trans_config.apply_transformations_atomiser(data)

            tokens = tokens.float()
            tokens_mask = tokens_mask.float()
            mask = self.process_mask(mask)

        return tokens, tokens_mask, attention_mask, mask[0]

  
  
import torch
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import torch.distributed as dist

class DistributedShapeBasedBatchSampler(Sampler):
    """
    A distributed batch sampler that groups samples by shape and partitions the batches
    across GPUs. Each process only sees a subset of the batches based on its rank.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, rank=None, world_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Set up distributed parameters.
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank=0
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
        self.rank = rank
        self.world_size = world_size

        # Group indices by image shape.
        self.shape_to_indices = {}
        # Use a temporary DataLoader to iterate over the dataset (batch_size=1).
        loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
        for idx, data in tqdm(enumerate(loader), desc="Sampler initialization"):
            # Assuming each sample returns (image, label, ...); adjust as needed.
            image = data[0]
            # Convert image.shape (a torch.Size) to a tuple so it can be used as a key.
            shape_key = tuple(image.shape)
            self.shape_to_indices.setdefault(shape_key, []).append(idx)
        
        # Create batches from the groups.
        self.batches = []
        for indices in tqdm(self.shape_to_indices.values(), desc="Batch creation"):
            random.shuffle(indices)
            # Create batches for this shape group.
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size:
                    self.batches.append(batch)
        
        if self.shuffle:
            random.shuffle(self.batches)
        
        # Make sure total number of batches is divisible by the number of processes.
        total_batches = len(self.batches)
        remainder = total_batches % self.world_size
        if remainder != 0:
            if not self.drop_last:
                # Pad with extra batches (repeating from the beginning) so each process has equal work.
                pad_size = self.world_size - remainder
                self.batches.extend(self.batches[:pad_size])
                total_batches = len(self.batches)
            else:
                # If dropping last incomplete batches, remove the excess.
                total_batches = total_batches - remainder
                self.batches = self.batches[:total_batches]
        self.total_batches = total_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for i in range(self.rank, self.total_batches, self.world_size):
            batch = self.batches[i]
            yield batch

    def __len__(self):
        # Number of batches that this process will iterate over.
        return self.total_batches // self.world_size




class CustomFlairDataModule(pl.LightningDataModule):
    def __init__(self, path,config,trans_config, batch_size=8, num_workers=4):
        super().__init__()
        self.train_file = path + "_train.h5"
        self.val_file = path + "_train.h5"
        self.test_file = path + "_train.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans_config = trans_config
        self.config=config

    def setup(self,stage=None):
        # Define transformations for the training phase.
        #self.train_transform = T.Compose([
        #    T.RandomHorizontalFlip(),
        #    T.RandomVerticalFlip(),
        #])

        # Initialize the datasets with their transformations.
        # Note: This setup() method is called on each process so the dataset and sampler
        # will be created in the proper distributed context.

        self.train_dataset = CustomFLAIR(
            self.train_file,
            config=self.config,
            trans_config=self.trans_config,
        )

        self.val_dataset = CustomFLAIR(
            self.val_file,
            config=self.config,
            trans_config=self.trans_config,
        )
   
        self.test_dataset = CustomFLAIR(
            self.test_file,
            config=self.config,
            trans_config=self.trans_config,
        )


    def train_dataloader(self):
        # Create the custom distributed sampler inside the DataLoader call.

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Train DataLoader created on rank: {rank}")
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Validation DataLoader created on rank: {rank}")
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers
        )

    def test_dataloader(self):

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Test DataLoader created on rank: {rank}")
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers
        )