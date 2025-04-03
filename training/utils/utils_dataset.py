import torch
from tqdm import tqdm
from datetime import datetime
from configilm.extra.DataSets import BENv2_DataSet
import matplotlib.pyplot as plt

from .files_utils import*

def get_xp_name(encoder_name,opt=None):
    now = datetime.now()
    dt_string = now.strftime("- %H:%M:%S %d/%m")
    if opt!=None:
        encoder_name+" "+opt+dt_string
    return encoder_name+dt_string



def prepare_BENv2(mode="train"):
    dico_paths={"images_lmdb":"data/Encoded-BigEarthNet/",
            "metadata_parquet":"data/Encoded-BigEarthNet/metadata.parquet",
            "metadata_snow_cloud_parquet":"data/Encoded-BigEarthNet/metadata_for_patches_with_snow_cloud_or_shadow.parquet"}


    return BENv2_DataSet.BENv2DataSet(data_dirs=dico_paths, img_size=(14, 120, 120),split=mode)

def map_onehot_to_labels(L, ds, data):
    """
    L a list of ids
    ds lmdb dataset
    data a dataframe
    """
    # Dictionary to store results
    coordinate_to_label = {}

    # Iterate through each element in the list L
    for idx in L:
        # Get the one-hot vector from the dataset
        one_hot_vector = ds[idx][1]  # Shape [19], assumed to be a PyTorch tensor
        
        # Get the labels associated with this data point
        labels = get_labels(data, idx)
        
        # Process only elements with a single label
        if len(labels) != 1:
            continue

        # Find the index of the 1 in the one-hot vector
        for i, value in enumerate(one_hot_vector):
            if value == 1:  # Match active coordinate
                coordinate_to_label[i] = labels[0]

    return coordinate_to_label

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
    

        
    