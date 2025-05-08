from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ResNet import *
from training.scale_MAE import*
from collections import defaultdict
from training import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import matplotlib.pyplot as plt
import random
import torchmetrics
import warnings
import wandb
from torch.distributed import broadcast

#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")


def dynamic_weighted_average(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [T_total, B, C]
    returns: [B, C]
    """
    probs = F.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1).values
    weights = confidences / (confidences.sum(dim=0, keepdim=True) + 1e-8)
    return (logits * weights.unsqueeze(-1)).sum(dim=0)


class Model(pl.LightningModule):
    def __init__(self, config,labels, wand, name,transform=None):
        super().__init__()
        self.config = config
        self.labels=labels
        self.wand = wand
        self.transform=transform
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        weights_loss= torch.sqrt(1+self.get_label_weights().to(torch.float32))

        # Ensure same weights on all GPUs
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(weights_loss, src=0)

        self.weights_loss=weights_loss

        self.best_val_loss = float("inf")
        self.best_val_ap = float("-inf")
        self.best_train_loss = float("inf")
        self.best_train_ap = float("-inf")
        self.labels_idx = labels
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.name = name
        self.table=False

        self.metric_Acc_train = torchmetrics.Accuracy(task="multiclass", num_classes=40,average="micro")
        self.metric_Acc_validation = torchmetrics.Accuracy(task="multiclass", num_classes=40,average="micro")
        self.metric_Acc_test = torchmetrics.Accuracy(task="multiclass", num_classes=40,average="micro")

        self.metric_F1_train_per_class = torchmetrics.classification.MulticlassF1Score(num_classes=40, average="none")
        self.metric_F1_val_per_class = torchmetrics.classification.MulticlassF1Score(num_classes=40, average="none")
        self.metric_F1_test_per_class = torchmetrics.classification.MulticlassF1Score(num_classes=40, average="none")

        


        self.confmat_test = torchmetrics.ConfusionMatrix(
            task='multiclass',
            num_classes=self.num_classes
        )

        if config["encoder"] == "ResNet50":
            self.encoder = ResNet50(config["trainer"]["num_classes"], channels=6,transform=self.transform)
        if config["encoder"] == "ResNet101":
            self.encoder = ResNet101(config["trainer"]["num_classes"], channels=6,transform=self.transform)
        if config["encoder"] == "ResNet152":
            self.encoder = ResNet152(config["trainer"]["num_classes"], channels=6,transform=self.transform)
        if config["encoder"] == "ResNetSmall":
            self.encoder = ResNetSmall(config["trainer"]["num_classes"], channels=6,transform=self.transform)
        if config["encoder"] == "ResNetSuperSmall":
            self.encoder = ResNetSuperSmall(config["trainer"]["num_classes"], channels=6,transform=self.transform)


        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(
                config=self.config,
                transform=self.transform,
                depth=config["Atomiser"]["depth"],
                num_latents=config["Atomiser"]["num_latents"],
                latent_dim=config["Atomiser"]["latent_dim"],
                cross_heads=config["Atomiser"]["cross_heads"],
                latent_heads=config["Atomiser"]["latent_heads"],
                cross_dim_head=config["Atomiser"]["cross_dim_head"],
                latent_dim_head=config["Atomiser"]["latent_dim_head"],
                num_classes=config["trainer"]["num_classes"],
                attn_dropout=config["Atomiser"]["attn_dropout"],
                ff_dropout=config["Atomiser"]["ff_dropout"],
                weight_tie_layers=config["Atomiser"]["weight_tie_layers"],
                self_per_cross_attn=config["Atomiser"]["self_per_cross_attn"],
                final_classifier_head=config["Atomiser"]["final_classifier_head"],
                masking=config["Atomiser"]["masking"]
            )

        if config["encoder"] == "ViT":
            ViT_conf = config["ViT"]["config"]
            self.encoder = SimpleViT(
                image_size=config["ViT"]["image_size"],
                patch_size=config["ViT"]["patch_size"],
                num_classes=self.num_classes,
                dim=config["ViT"][ViT_conf]["dim"],
                depth=config["ViT"][ViT_conf]["depth"],
                heads=config["ViT"][ViT_conf]["heads"],
                mlp_dim=config["ViT"][ViT_conf]["mlp_dim"],
                channels=6,
                dim_head=config["ViT"][ViT_conf]["dim_head"],
                transform=self.transform
            )


        if config["encoder"] == "ScaleMAE":
            self.encoder=CustomScaleMAE(transform=self.transform)


        self.common_classes,self.frequent_classes,self.rare_classes=self.get_label_frequencies()

      
        


  
        self.loss = nn.CrossEntropyLoss(weight=self.weights_loss)
        self.lr = float(config["trainer"]["lr"])
        self.max_tokens=self.config["trainer"]["max_tokens"]

    def get_label_frequencies(self):
     

        ids_common=[]
        ids_rare=[]
        ids_frequent=[]

        for label in self.labels:
           
            if self.labels[label]["Frequency"]=="frequent":
                ids_frequent.append(int(self.labels[label]["id"]))
            
            if self.labels[label]["Frequency"]=="common":
                ids_common.append(int(self.labels[label]["id"]))

            if self.labels[label]["Frequency"]=="rare":
                ids_rare.append(int(self.labels[label]["id"]))

        ids_common=torch.from_numpy(np.array(ids_common))
        ids_rare=torch.from_numpy(np.array(ids_rare))
        ids_frequent=torch.from_numpy(np.array(ids_frequent))
        return ids_common,ids_frequent,ids_rare

    def get_label_weights(self):
        weights=dict()

    

        cpt_count=0
        for label in self.labels:
            weights[int(self.labels[label]["id"])]=int(self.labels[label]["count"])
            cpt_count+=int(self.labels[label]["count"])



        res=[]
        for w in range(len(self.labels.keys())):
            res.append(weights[w])
        
        res=torch.from_numpy(np.array(res)).to(float)
        res*=40
        res=cpt_count/res

        return res
        
    def forward(self, data,training=False):

        if self.config["encoder"] == "Atomiser":
            return self.encoder(data,training=training)


        data_s2,data_l7,data_modis=self.encoder.process_data(data)

        key_activ_mod= "training" if training else "test"

        bool_s2= self.config["dataset"][key_activ_mod]["S2"] 
        bool_l7= self.config["dataset"][key_activ_mod]["L7"] 
        bool_mo= self.config["dataset"][key_activ_mod]["MODIS"]

        res_array=[]

        if bool_s2:
            #B 8 12 12 6
            self.encoder.res=20
            
            imgs_s2,masks_s2=data_s2
            b_size=imgs_s2.shape[0]
            t_size=imgs_s2.shape[1]

            imgs_s2[masks_s2==1]=0.0
    
            tmp_img_s2=einops.rearrange(imgs_s2,"B T H W C -> (T B) C H W")
    
            
            res_s2=self.encoder.forward(tmp_img_s2)
            res_s2=einops.rearrange(res_s2,"(T B) L -> T B L",T=t_size,B=b_size)

            res_array.append(res_s2)
        
        if bool_l7:
            #B 20 4 4 6 
            self.encoder.res=30
            imgs_l7,masks_l7=data_l7

            b_size=imgs_l7.shape[0]
            t_size=imgs_l7.shape[1]

            imgs_l7[masks_l7==1]=0.0

            tmp_img_l7=einops.rearrange(imgs_l7,"B T H W C -> (T B) C H W")

            tmp_img_l7 = F.interpolate(tmp_img_l7, size=(12, 12), mode='bicubic', align_corners=False)
            #tmp_img_l7 = F.pad(tmp_img_l7, pad=(4, 4, 4, 4), mode='constant', value=0)
            
            tmp_img_l7=self.encoder.forward(tmp_img_l7)
            tmp_img_l7=einops.rearrange(tmp_img_l7,"(T B) L -> T B L",T=t_size,B=b_size)

            res_array.append(tmp_img_l7)
        
        if bool_mo:
            imgs_mo,masks_mo=data_modis
            imgs_mo[masks_mo==1]=0.0
            self.encoder.res=500

            b_size=imgs_mo.shape[0]
            t_size=imgs_mo.shape[1]

            tmp_img_mo=einops.rearrange(imgs_mo,"B T H W C -> (T B) C H W")
            tmp_img_mo = F.interpolate(tmp_img_mo, size=(12, 12), mode='bicubic', align_corners=False)
            #tmp_img_mo = F.pad(tmp_img_mo, pad=(5, 6, 5, 6), mode='constant', value=0)

                
            
            tmp_img_mo=self.encoder.forward(tmp_img_mo)
            tmp_img_mo=einops.rearrange(tmp_img_mo,"(T B) L -> T B L",T=t_size,B=b_size)

            res_array.append(tmp_img_mo)

        res_array=torch.cat(res_array,dim=0)

        if self.config["trainer"]["agregation"]=="confidence":
            
            res_array=dynamic_weighted_average(res_array)
            
        else:
            res_array=res_array.mean(dim=0)
            
            
        
        return res_array

       
     
      
        
    

            


    def training_step(self, batch, batch_idx):
        

        labels,frequency=batch[-2:]
        y_hat = self.forward(batch[:-2],training=False)
        
        labels=labels.to(torch.long)
        loss = self.loss(y_hat, labels)

        y_hat=torch.argmax(y_hat,dim=1)
        

        self.metric_Acc_train.update(y_hat,labels)
        self.metric_F1_train_per_class.update(y_hat, labels)


        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.log("log train_loss", torch.log(loss), on_step=True, on_epoch=False, logger=True, sync_dist=False)

        

        return loss 
    
    

        
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        loss = metrics.get("train_loss", float("inf"))

        

        Acc=self.metric_Acc_train.compute()
        f1s = self.metric_F1_train_per_class.compute()
        f1_freq = f1s[self.frequent_classes].mean()
        f1_com = f1s[self.common_classes].mean()
        f1_rare = f1s[self.rare_classes].mean()
        
        #self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        

        self.log("train_Acc", Acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1", f1s.mean(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1_freq", f1_freq, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1_com", f1_com, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1_rare", f1_rare, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.metric_Acc_train.reset()
        self.metric_F1_train_per_class.reset()

    
    



 


    
    
    def validation_step(self, batch, batch_idx):
        
        labels,frequency=batch[-2:]
        y_hat = self.forward(batch[:-2],training=False)


       
        
        labels=labels.to(torch.long)
        loss = self.loss(y_hat, labels)

      

        #if torch.isnan(loss) or torch.isinf(loss):
        #    print(f"[Rank {self.global_rank}] Loss is NaN!")
        #    print("y_hat:", y_hat)
        #    print("labels:", labels)
        #    print("weights_loss:", self.weights_loss)



        y_hat=torch.argmax(y_hat,dim=1) #

        
        

        self.metric_Acc_validation.update(y_hat,labels)
        self.metric_F1_val_per_class.update(y_hat, labels)


        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        return loss

    def on_validation_epoch_end(self):
        
        metrics = self.trainer.callback_metrics
        loss = metrics.get("val_loss", float("inf"))


        Acc=self.metric_Acc_validation.compute()
        f1s = self.metric_F1_val_per_class.compute()
        f1_freq = f1s[self.frequent_classes].mean()
        f1_com = f1s[self.common_classes].mean()
        f1_rare = f1s[self.rare_classes].mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("log val_loss", torch.log(loss), on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log("val_Acc", Acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1", f1s.mean(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1_freq", f1_freq, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1_com", f1_com, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1_rare", f1_rare, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.metric_Acc_validation.reset()
        self.metric_F1_val_per_class.reset()
        
        
            
        
    
        
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        
    def test_step(self, batch, batch_idx):
        labels,frequency=batch[-2:]
        y_hat = self.forward(batch[:-2],training=False)
        
        labels=labels.to(torch.long)
        

        y_hat=torch.argmax(y_hat,dim=1)
        

        self.metric_Acc_test.update(y_hat,labels)
        self.metric_F1_test_per_class.update(y_hat, labels)


        

        return None


    def on_test_epoch_end(self):
  

        Acc=self.metric_Acc_test.compute()
        f1s = self.metric_F1_test_per_class.compute()
        f1_freq = f1s[self.frequent_classes].mean()
        f1_com = f1s[self.common_classes].mean()
        f1_rare = f1s[self.rare_classes].mean()
        
        

        self.log("test_Acc", Acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1", f1s.mean(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1_freq", f1_freq, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1_com", f1_com, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1_rare", f1_rare, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.metric_Acc_test.reset()
        self.metric_F1_test_per_class.reset()


        
        
        
        
    def save_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        torch.save(self.encoder.state_dict(), file_path)
        
    def load_model(self, name=None):
        if name is not None:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}_{name}.pth"
        else:
            file_path = f"./pth_files/{self.config['encoder']}_{self.name}.pth"
        self.encoder.load_state_dict(torch.load(file_path, weights_only=True))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        cosine_anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["trainer"]["epochs"]*2, eta_min=0.0)#
        #cosine_anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30000, T_mult=1, eta_min=0.0)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': cosine_anneal_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}}

