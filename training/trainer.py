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
from torch.distributed import broadcast

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
        weights_loss= torch.log(1+self.get_label_weights().to(torch.float32))

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

        self.metric_F1_train = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_validation = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_test = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")

        self.metric_F1_train_freq = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_validation_freq = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_test_freq = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")

        self.metric_F1_train_com = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_validation_com = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_test_com = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")

        self.metric_F1_train_rare = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_validation_rare = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")
        self.metric_F1_test_rare = torchmetrics.F1Score(task="multiclass", num_classes=40,average="macro")

        


        self.confmat_test = torchmetrics.ConfusionMatrix(
            task='multiclass',
            num_classes=self.num_classes
        )


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


        print(self.weights_loss)
        self.loss = nn.CrossEntropyLoss()#weight=self.weights_loss)
        self.lr = float(config["trainer"]["lr"])
        self.max_tokens=self.config["trainer"]["max_tokens"]

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
        
    def forward(self, tokens,training=False):
        model_output=self.encoder(tokens,training=training)
       
     
      
        return model_output
    

            


    def training_step(self, batch, batch_idx):
        

        labels,frequency=batch[-2:]
        y_hat = self.forward(batch[:-2],training=False)
        
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


        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        self.log("log train_loss", torch.log(loss), on_step=True, on_epoch=False, logger=True, sync_dist=False)

        

        return loss 

    

        
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        loss = metrics.get("train_loss", float("inf"))

        

        Acc=self.metric_Acc_train.compute()
        F1_score=self.metric_F1_train.compute()
        F1_freq=self.metric_F1_train_freq.compute()
        F1_com=self.metric_F1_train_com.compute()
        F1_rare=self.metric_F1_train_rare.compute()
        
        #self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        

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

    
    



 


    
    
    def validation_step(self, batch, batch_idx):
        
        labels,frequency=batch[-2:]
        y_hat = self.forward(batch[:-2],training=False)


       
        
        labels=labels.to(torch.long)
        loss = self.loss(y_hat, labels)

        print("val loss",loss)



        y_hat=torch.argmax(y_hat,dim=1) #

        
        

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
        labels,frequency=batch[-2:]
        y_hat = self.forward(batch[:-2],training=False)
        
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
        cosine_anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["trainer"]["epochs"]*2, eta_min=0.0)#
        #cosine_anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30000, T_mult=1, eta_min=0.0)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': cosine_anneal_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}}

