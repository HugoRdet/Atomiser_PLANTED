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
from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO
from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule
import random
import torchmetrics
import warnings
import wandb

#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

class Model(pl.LightningModule):
    def __init__(self, config,labels, wand, name,img_shape=None):
        super().__init__()
        self.config = config
        self.labels=labels
        self.wand = wand
        self.img_shape=img_shape
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

        self.metric_IoU_train = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro")
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
        
    def forward(self, tokens,tokens_masks,attention_mask):

        
        model_output=self.encoder(tokens,tokens_masks,attention_mask)
       
     
      
        return model_output
            


    def training_step(self, batch, batch_idx):
        tokens, tokens_mask, attention_mask, labels = batch
        y_hat = self.forward(tokens, tokens_mask, attention_mask)

        labels = labels.long()
        preds = torch.argmax(y_hat.clone(), dim=1)

        loss = self.loss(y_hat, labels)
        self.metric_IoU_train.update(preds, labels)

        # 🔥 CRITICAL DEBUGGING INFORMATION 🔥
        print("Unique predicted classes:", preds.unique())
        print("Unique true classes:", labels.unique())

        if batch_idx == 0:  # visualize once per epoch
            plt.figure(figsize=(10,4))

            plt.subplot(1,2,1)
            plt.title("Predicted")
            plt.imshow(preds[0].cpu().numpy(), cmap='tab20', vmin=0, vmax=self.num_classes-1)
            plt.colorbar()

            plt.subplot(1,2,2)
            plt.title("Ground Truth")
            plt.imshow(labels[0].cpu().numpy(), cmap='tab20', vmin=0, vmax=self.num_classes-1)
            plt.colorbar()

            plt.tight_layout()
            plt.show()

        return loss

    

        
    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        loss = metrics.get("train_loss", float("inf"))

        IoU=self.metric_IoU_train.compute()

        
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("log train_loss", torch.log(torch.tensor(loss)), on_step=False, on_epoch=True, logger=True, sync_dist=True)


        self.log("train_IoU", IoU, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        self.metric_IoU_train.reset()
        
    def validation_step(self, batch, batch_idx):
        tokens, tokens_mask,attention_mask, labels = batch
        

        y_hat = self.forward(tokens,tokens_mask,attention_mask)
        
        labels=labels.to(torch.long)
        loss = self.loss(y_hat, labels)

        y_hat=torch.argmax(y_hat,dim=1)
        self.metric_IoU_val.update(y_hat, labels)


        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=False, sync_dist=False)
        
        
        

       
        return loss        

    def on_validation_epoch_end(self):
        
        metrics = self.trainer.callback_metrics
        loss = metrics.get("val_loss", float("inf"))

        IoU=self.metric_IoU_val.compute()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("log val_loss", torch.log(loss), on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log("val_IoU", IoU, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        self.metric_IoU_val.reset()
        
        
            
        
    
        
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        
    def test_step(self, batch, batch_idx):
        tokens, tokens_mask,attention_mask, labels = batch
        

        y_hat = self.forward(tokens,tokens_mask,attention_mask)
        
        labels=labels.to(torch.long)
        

        y_hat=torch.argmax(y_hat,dim=1)
        self.metric_IoU_test.update(y_hat, labels)
        self.confmat_test.update(y_hat, labels)


    def on_test_epoch_end(self):
  
        IoU_per_class = self.metric_IoU_test.compute().cpu().numpy()
        mean_IoU = IoU_per_class.mean()
    
        labels = [self.labels[k] for k in self.labels.keys()] 
    
        if self.wand:
            table = wandb.Table(columns=["Class", "IoU"])
            for label_name, iou_score in zip(labels, IoU_per_class):
                table.add_data(label_name, float(iou_score))
            
            # Add the average IoU as a final row
            table.add_data("Average IoU", float(mean_IoU))
            
            wandb.log({"IoU_per_Class": table})
    
        # Reset IoU metric after logging
        self.metric_IoU_test.reset()
    
        confmat = self.confmat_test.compute().cpu().numpy()
    
        # Normalize confusion matrix by rows and round
        confmat_normalized = confmat.astype('float') / confmat.sum(axis=1, keepdims=True)
        confmat_normalized = np.nan_to_num(confmat_normalized)  # handle division by zero
        confmat_normalized = np.round(confmat_normalized, 2)
    
        # Log confusion matrix using WandB
        if self.wand:
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(confmat_normalized, cmap='Blues')
    
            axis = np.arange(len(labels))
    
            # Annotate confusion matrix
            ax.set_xticks(axis)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticks(axis)
            ax.set_yticklabels(labels)
    
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Normalized Confusion Matrix')
    
            # Ensure matrix occupies the entire plot area
            plt.tight_layout()
    
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{confmat_normalized[i, j]:.2f}", ha='center', va='center', color='black')
    
            #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            wandb.log({"Confusion Matrix": wandb.Image(fig)})
    
            plt.close(fig)
    
        self.confmat_test.reset()


        
        
        
        
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

