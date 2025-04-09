from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ResNet import *
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


#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

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

        self.metric_Acc_train = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.metric_Acc_validation = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.metric_Acc_test = torchmetrics.Accuracy(task="multiclass", num_classes=40)

        self.metric_F1_train = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_validation = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_test = torchmetrics.F1Score(task="multiclass", num_classes=40)

        self.metric_F1_train_freq = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_validation_freq = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_test_freq = torchmetrics.F1Score(task="multiclass", num_classes=40)

        self.metric_F1_train_com = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_validation_com = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_test_com = torchmetrics.F1Score(task="multiclass", num_classes=40)

        self.metric_F1_train_rare = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_validation_rare = torchmetrics.F1Score(task="multiclass", num_classes=40)
        self.metric_F1_test_rare = torchmetrics.F1Score(task="multiclass", num_classes=40)

        

        self.metric_IoU_val = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average="macro")
        self.metric_IoU_test = torchmetrics.classification.MulticlassJaccardIndex(self.num_classes, average=None)

        self.confmat_test = torchmetrics.ConfusionMatrix(
            task='multiclass',
            num_classes=self.num_classes
        )


        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(
                config=self.config,
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


        self.loss = nn.CrossEntropyLoss()
        self.lr = float(config["trainer"]["lr"])
        self.max_tokens=self.config["trainer"]["max_tokens"]
        
    def forward(self, tokens,tokens_masks,training=False):
        model_output=self.encoder(tokens,tokens_masks,training=training)
       
     
      
        return model_output
            


    def training_step(self, batch, batch_idx):
        

        tokens,tokens_mask,labels,frequency=self.process_data(batch)
        
        

        y_hat = self.forward(tokens,tokens_masks=tokens_mask,training=True)
        
        labels=labels.to(torch.long)
        loss = self.loss(y_hat, labels)

        y_hat=torch.argmax(y_hat,dim=1)
        

        self.metric_Acc_train.update(y_hat,labels)
        self.metric_F1_train.update(y_hat,labels)

        if y_hat[frequency==0].shape[0]!=0:
            self.metric_F1_train_freq.update(y_hat[frequency==0],labels[frequency==0])
        if y_hat[frequency==1].shape[0]!=0:
            self.metric_F1_train_com.update(y_hat[frequency==1],labels[frequency==1])
        if y_hat[frequency==2].shape[0]!=0:
            self.metric_F1_train_rare.update(y_hat[frequency==2],labels[frequency==2])


        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        

        return loss 

    

        
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        loss = metrics.get("train_loss", float("inf"))

        

        Acc=self.metric_Acc_train.compute()
        F1_score=self.metric_F1_train.compute()
        F1_freq=self.metric_F1_train_freq.compute()
        F1_com=self.metric_F1_train_com.compute()
        F1_rare=self.metric_F1_train_rare.compute()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("log train_loss", torch.log(loss), on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log("train_Acc", Acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1", F1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1_freq", F1_freq, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1_com", F1_com, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_F1_rare", F1_rare, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.metric_Acc_train.reset()
        self.metric_F1_train.reset()
        self.metric_F1_train_freq.reset()
        self.metric_F1_train_com.reset()
        self.metric_F1_train_rare.reset()

    
    def get_tokens(self,img,date,mask,mode="optique",modality="s2"):
        
  

        if mode=="optique":
            return self.transform.apply_transformations_optique(img,date,mask,modality)
        if mode=="sar":
            return self.transform.apply_transformations_SAR(img,date,mask,modality)



 


    def process_data(self,batch):
        L_tokens=[]
        L_masks=[]
        img_s2,img_l7,img_mo,img_s1,img_al,date_s2,date_l7,date_mo,date_s1,date_al,mask_s2,mask_l7,mask_mo,mask_s1,mask_al,labels,frequency = batch
        
        if self.config["dataset"]["S2"]:
            tokens_s2,tokens_mask_s2=self.get_tokens(img_s2,date_s2,mask_s2,mode="optique",modality="s2")
            L_masks.append(tokens_mask_s2)
            L_tokens.append(tokens_s2)
        
        if self.config["dataset"]["L7"]:
            tokens_l7,tokens_mask_l7=self.get_tokens(img_l7,date_l7,mask_l7,mode="optique",modality="l7")
            L_masks.append(tokens_mask_l7)
            L_tokens.append(tokens_l7)

        if self.config["dataset"]["MODIS"]:
            tokens_mo,tokens_mask_mo=self.get_tokens(img_mo,date_mo,mask_mo,mode="optique",modality="modis")
            L_masks.append(tokens_mask_mo)
            L_tokens.append(tokens_mo)

        if self.config["dataset"]["S1"]:
            tokens_s1,tokens_mask_s1=self.get_tokens(img_s1,date_s1,mask_s1,mode="sar",modality="s1")
            L_masks.append(tokens_mask_s1)
            L_tokens.append(tokens_s1)

        if self.config["dataset"]["ALOS"]:
            tokens_al,tokens_mask_al=self.get_tokens(img_al,date_al,mask_al,mode="sar",modality="alos")
            L_masks.append(tokens_mask_al)
            L_tokens.append(tokens_al)


        tokens=torch.cat(L_tokens,dim=1)
        tokens_mask=torch.cat(L_masks,dim=1)


            
        return tokens,tokens_mask,labels,frequency
    
    def validation_step(self, batch, batch_idx):
        tokens,tokens_mask,labels,frequency=self.process_data(batch)
        
        y_hat = self.forward(tokens,tokens_masks=tokens_mask,training=False)
        
        labels=labels.to(torch.long)
        loss = self.loss(y_hat, labels)

        y_hat=torch.argmax(y_hat,dim=1)
        

        self.metric_Acc_validation.update(y_hat,labels)
        self.metric_F1_validation.update(y_hat,labels)
        if y_hat[frequency==0].shape[0]!=0:
            self.metric_F1_validation_freq.update(y_hat[frequency==0],labels[frequency==0])
        if y_hat[frequency==1].shape[0]!=0:
            self.metric_F1_validation_com.update(y_hat[frequency==1],labels[frequency==1])
        if y_hat[frequency==2].shape[0]!=0:
            self.metric_F1_validation_rare.update(y_hat[frequency==2],labels[frequency==2])


        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        return loss

    def on_validation_epoch_end(self):
        
        metrics = self.trainer.callback_metrics
        loss = metrics.get("val_loss", float("inf"))

        IoU=self.metric_IoU_val.compute()

        Acc=self.metric_Acc_validation.compute()
        F1_score=self.metric_F1_validation.compute()
        F1_freq=self.metric_F1_validation_freq.compute()
        F1_com=self.metric_F1_validation_com.compute()
        F1_rare=self.metric_F1_validation_rare.compute()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("log val_loss", torch.log(loss), on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log("val_Acc", Acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1", F1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1_freq", F1_freq, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1_com", F1_com, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_F1_rare", F1_rare, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.metric_Acc_validation.reset()
        self.metric_F1_validation.reset()
        self.metric_F1_validation_freq.reset()
        self.metric_F1_validation_com.reset()
        self.metric_F1_validation_rare.reset()
        
        
            
        
    
        
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        
    def test_step(self, batch, batch_idx):
        tokens,tokens_mask,labels,frequency=self.process_data(batch)
        

        y_hat = self.forward(tokens,tokens_masks=tokens_mask,training=False)
        
        labels=labels.to(torch.long)
        

        y_hat=torch.argmax(y_hat,dim=1)
        

        self.metric_Acc_test.update(y_hat,labels)
        self.metric_F1_test.update(y_hat,labels)

        if y_hat[frequency==0].shape[0]!=0:
            self.metric_F1_test_freq.update(y_hat[frequency==0],labels[frequency==0])
        if y_hat[frequency==1].shape[0]!=0:
            self.metric_F1_test_com.update(y_hat[frequency==1],labels[frequency==1])
        if y_hat[frequency==2].shape[0]!=0:
            self.metric_F1_test_rare.update(y_hat[frequency==2],labels[frequency==2])


        

        return None


    def on_test_epoch_end(self):
  

        Acc=self.metric_Acc_test.compute()
        F1_score=self.metric_F1_test.compute()
        F1_freq=self.metric_F1_test_freq.compute()
        F1_com=self.metric_F1_test_com.compute()
        F1_rare=self.metric_F1_test_rare.compute()
        
        

        self.log("test_Acc", Acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1", F1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1_freq", F1_freq, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1_com", F1_com, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("test_F1_rare", F1_rare, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.metric_Acc_test.reset()
        self.metric_F1_test.reset()
        self.metric_F1_test_freq.reset()
        self.metric_F1_test_com.reset()
        self.metric_F1_test_rare.reset()


        
        
        
        
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
        cosine_anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["trainer"]["epochs"]*2, eta_min=0.0)
        #cosine_anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, eta_min=0.0)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': cosine_anneal_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}}

