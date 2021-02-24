import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError  as MSE
from pytorch_lightning.metrics.classification import Accuracy

import numpy as np
import pandas as pd
from AESRC.model import Wav2VecClassifier
import math
import torch_optimizer
from AESRC.centerloss import CenterLoss

class Wav2VecModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        
        hidden_size = HPARAMS['model_hidden_size']

        self.model = Wav2VecClassifier(hidden_size=hidden_size)
        self.center_loss = CenterLoss(num_classes=8, feat_dim=hidden_size)

        self.classification_criterion = nn.NLLLoss()
        self.accuracy = Accuracy()

        self.lr = HPARAMS['training_lr']
        self.alpha = 1
        self.lr_cent = 0.5

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    # def on_after_backward(self):
    #     for param in self.center_loss.parameters():
    #         param.grad.data *= (self.lr_cent / (self.alpha * self.lr))

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer1 = torch_optimizer.DiffGrad(self.model.parameters(), lr=self.lr)
        # optimizer2 = torch_optimizer.DiffGrad(self.center_loss.parameters(), lr=self.lr_cent)
        return [optimizer1]

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     x, y = batch
    #     y_hat, attn_output = self(x)

    #     if optimizer_idx == 0:
    #         classification_loss = self.classification_criterion(y_hat.float(), y.long())
    #         preds = torch.argmax(y_hat, dim=1)
    #         acc = self.accuracy(preds.long(), y.long())
    #         return {'loss':classification_loss,'train_acc':acc,}

    #     if optimizer_idx == 1:
    #         center_loss = self.center_loss(attn_output, y)
    #         return {'loss':center_loss}

    #     # loss = classification_loss + self.alpha * center_loss

    #     # preds = torch.argmax(y_hat, dim=1)
    #     # acc = self.accuracy(preds.long(), y.long())

    #     # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
    #     # self.log('train_cls_loss', classification_loss, on_step=True, on_epoch=True, prog_bar=False)
    #     # self.log('train_center_loss', center_loss, on_step=True, on_epoch=True, prog_bar=False)

    #     return {'loss':loss, 
    #             'train_cls_loss':classification_loss,
    #             'train_center_loss':center_loss,
    #             'train_acc':acc,
    #             }
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, attn_output = self(x)

 
        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())
        center_loss = self.center_loss(attn_output, y)
        loss = classification_loss + self.alpha * center_loss


        self.log('train', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('cls', classification_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('center', center_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss':loss, 
                # 'train_cls_loss':classification_loss,
                # 'train_center_loss':center_loss,
                'train_acc':acc,
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        acc = torch.tensor([x['train_acc'] for x in outputs]).mean()
        # centerloss = torch.tensor([x['train_center_loss'] for x in outputs]).mean()
        # clsloss = torch.tensor([x['train_cls_loss'] for x in outputs]).mean()

        # self.log('epoch_loss' , loss, prog_bar=True)
        # self.log('center_loss' , centerloss, prog_bar=True)
        # self.log('cls_loss' , clsloss, prog_bar=True)
        # self.log('tr_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        loss = classification_loss

        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())

        return {'val_loss':loss, 
                'val_acc': acc}

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        
        self.log('v_loss', val_loss, prog_bar=True)
        self.log('v_acc', val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        loss = classification_loss
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())

        return {
                'loss': loss.item(),
                'acc': acc.item(),
                }

    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        acc = torch.tensor([x['acc'] for x in outputs]).mean()

        pbar = {
            'test_loss': loss.item(),
            'test_acc' : acc.item()
        }
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)