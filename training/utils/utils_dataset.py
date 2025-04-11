import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from .files_utils import*
import re

def get_ids_in_folder(base_path="./data/public/1.0.1/train/"):
    """
    Get a list of IDs from .npz files present in the specified directory.

    Parameters:
    - base_path (str): Path to the directory containing the .npz files.

    Returns:
    - list of ints: IDs extracted from filenames.
    """
    ids = []
    pattern = re.compile(r"^(\d{4})_d5000\.npz$")

    for filename in os.listdir(base_path):
        match = pattern.match(filename)
        if match:
            ids.append(int(match.group(1)))

    return sorted(ids)

def load_npz_by_id(id, base_path="./data/public/1.0.1/train/"):
    """
    Load an .npz file by its ID.

    Parameters:
    - id (int): The identifier number XXXX in the filename.
    - base_path (str): Path to the directory containing the .npz files.

    Returns:
    - dict-like object: The content of the loaded .npz file.
    """
    filename = f"{id:04d}_d5000.npz"
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")

    data = np.load(filepath)
    return data
def compute_channel_mean_std(mode="train"):
    # Initialize accumulators for the 'sen_spatch' images (10 channels)
    type_convert=np.float64
    sen1_sum    = np.zeros(3, dtype=type_convert)
    sen1_sum_sq = np.zeros(3, dtype=type_convert)
    sen1_count  = 0.0

    sen2_sum    = np.zeros(10, dtype=type_convert)
    sen2_sum_sq = np.zeros(10, dtype=type_convert)
    sen2_count  = 0.0

    l7_sum    = np.zeros(6, dtype=type_convert)
    l7_sum_sq = np.zeros(6, dtype=type_convert)
    l7_count  = 0.0

    modis_sum    = np.zeros(7, dtype=type_convert)
    modis_sum_sq = np.zeros(7, dtype=type_convert)
    modis_count  = 0.0

    alos_sum    = np.zeros(3, dtype=type_convert)
    alos_sum_sq = np.zeros(3, dtype=type_convert)
    alos_count  = 0.0


    ids=get_ids_in_folder(base_path=f"./data/public/1.0.1/{mode}/")

    for id in ids:
        data=load_npz_by_id(id,base_path=f"./data/public/1.0.1/{mode}/")

        s1=data["s1"] #shape (8,12,12,3) 
        s2=data["s2"] #shape (8,12,12,10) 
        l7=data["l7"] #shape (20,4,4,6) 
        modis=data["modis"] #shape (60,1,1,7) 
        alos=data["alos"] #shape (4,4,4,3)
        alos[:,:,:,0]=10*np.log(alos[:,:,:,0]**2)-83.0
        alos[:,:,:,1]=10*np.log(alos[:,:,:,1]**2)-83.0

        for img_idx in range(s2.shape[0]):
            
            sen1_sum+=np.sum(s1[img_idx],axis=(0,1,2))
            sen2_sum+=np.sum(s2[img_idx],axis=(0,1,2))
            l7_sum+=np.sum(l7[img_idx] ,axis=(0,1,2))
            modis_sum+=np.sum(modis[img_idx],axis=(0,1,2))
            alos_sum+=np.sum(alos[img_idx],axis=(0,1,2))

       
            sen1_sum_sq+=np.sum(np.square(s1[img_idx]),axis=(0,1,2))
            sen2_sum_sq+=np.sum(np.square(s2[img_idx]),axis=(0,1,2))
            l7_sum_sq+=np.sum(np.square(l7[img_idx] ),axis=(0,1,2))
            modis_sum_sq+=np.sum(np.square(modis[img_idx]),axis=(0,1,2))
            alos_sum_sq+=np.sum(np.square(alos[img_idx]),axis=(0,1,2))

            sen1_count += np.prod(s1[img_idx].shape[:3])
            sen2_count += np.prod(s2[img_idx].shape[:3])
            l7_count   += np.prod(l7[img_idx].shape[:3])
            modis_count+= np.prod(modis[img_idx].shape[:3])
            alos_count += np.prod(alos[img_idx].shape[:3])

    
        
        s1_mean=sen1_sum / sen1_count 
        s1_var = sen1_sum_sq / sen1_count - np.square(s1_mean)
        s1_var = np.maximum(s1_var, 0)
        s1_std = np.sqrt(s1_var)

        s2_mean= sen2_sum / sen2_count
        s2_var = sen2_sum_sq / sen2_count - np.square(s2_mean)
        s2_var = np.maximum(s2_var, 0)
        s2_std = np.sqrt(s2_var)

        l7_mean= l7_sum / l7_count
        l7_var = l7_sum_sq / l7_count - np.square(l7_mean)
        l7_var = np.maximum(l7_var, 0)
        l7_std = np.sqrt(l7_var)
        
        modis_mean= modis_sum / modis_count
        modis_var = modis_sum_sq / modis_count - np.square(modis_mean)
        modis_var = np.maximum(modis_var, 0)
        modis_std = np.sqrt(modis_var)

        alos_mean= alos_sum / alos_count
        alos_var = alos_sum_sq / alos_count - np.square(alos_mean)
        alos_var = np.maximum(alos_var, 0)
        alos_std = np.sqrt(alos_var)

    print(s1_mean,s1_std)
    print(s2_mean,s2_std)

    torch.save(torch.from_numpy(s1_mean),"./data/normalisation/s1_mean.pt")
    torch.save(torch.from_numpy(s1_std),"./data/normalisation/s1_std.pt")

    torch.save(torch.from_numpy(s2_mean),"./data/normalisation/s2_mean.pt")
    torch.save(torch.from_numpy(s2_std),"./data/normalisation/s2_std.pt")

    torch.save(torch.from_numpy(l7_mean),"./data/normalisation/l7_mean.pt")
    torch.save(torch.from_numpy(l7_std),"./data/normalisation/l7_std.pt")

    torch.save(torch.from_numpy(modis_mean),"./data/normalisation/modis_mean.pt")
    torch.save(torch.from_numpy(modis_std),"./data/normalisation/modis_std.pt")

    torch.save(torch.from_numpy(alos_mean),"./data/normalisation/alos_mean.pt")
    torch.save(torch.from_numpy(alos_std),"./data/normalisation/alos_std.pt")
        

    

    


def plot_histo_dico(dico):
    # Assuming 'dico_labels' is your dictionary
    plt.figure(figsize=(20, 10))  # Make the figure wider
    plt.bar(dico.keys(), dico.values())
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate labels if needed
    plt.tight_layout()       # Adjust layout to fit labels

    plt.savefig('histogram.pdf', format='pdf')  # Saves the figure as a PDF file
    plt.close()  # Close the plot to avoid displaying it

def get_labels(ds,id):

    row=ds.iloc[id]

    # Access the "labels" column
    labels = row['labels']  # Replace 'labels' with the actual column name if different

    return labels

def get_split(ds,id):

    row=ds.iloc[id]

    # Access the "labels" column
    split = row['split']  # Replace 'labels' with the actual column name if different

    return split

def get_one_hot_indices(tensor):
    """
    Returns a list of indices of labels encoded in a one-hot encoded tensor.
    
    Args:
        tensor (torch.Tensor): A 2D one-hot encoded tensor (batch_size x num_classes)
        
    Returns:
        List[int]: Indices of the labels for each row.
    """
    # Use argmax to find the indices of the maximum value (1) in each row
    indices = torch.nonzero(tensor == 1, as_tuple=True)[0]
    return indices.tolist()


def get_tiny_dataset(ds,df,MAX_IDs=200,mode="train"):

    idxs=dict()
    dico_stats=dict()
    dico_tmp_counts=dict()


    for i in range(len(ds)):
        split=get_split(df,i)
        if split!=mode:
            continue
        _, tmp_lbl = ds[i]
        tmp_id=get_one_hot_indices(tmp_lbl)

    

        if len(tmp_id)>1:
            continue

        

        if not tmp_id[0] in dico_stats:
            dico_stats[tmp_id[0]]=1
        else:

            if dico_stats[tmp_id[0]]>=MAX_IDs:
                continue
            dico_stats[tmp_id[0]]+=1
        if not tmp_id[0] in idxs:
            idxs[tmp_id[0]]=[]

        idxs[tmp_id[0]].append(i)

    


    for i in range(len(ds)):
        split=get_split(df,i)
        if split!=mode:
            continue

        if i in idxs:
            continue

        _, tmp_lbl = ds[i]
        L_id=get_one_hot_indices(tmp_lbl)

        min_rep=-1
        min_rep_id=-1
        for tmp_idx in L_id:
            if not tmp_idx in dico_stats:
                dico_stats[tmp_idx]=0
                
            if dico_stats[tmp_idx]>=MAX_IDs:
                continue
            if min_rep==-1 or dico_stats[tmp_idx]<min_rep:
                min_rep=dico_stats[tmp_idx]
                min_rep_id=tmp_idx



        if min_rep_id==-1:
            continue

    
        
        
        dico_stats[min_rep_id]+=1

        if not min_rep_id in idxs:
            idxs[min_rep_id]=[]
        
        idxs[min_rep_id].append(i)

    print("summary dico stats")
    print(dico_stats)
    

    return idxs,dico_stats
    

        
    