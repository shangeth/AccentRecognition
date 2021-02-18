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


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class Wav2VecModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        
        hidden_size = HPARAMS['model_hidden_size']

        self.model = Wav2VecClassifier(hidden_size=hidden_size)

        self.classification_criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.lr = HPARAMS['training_lr']

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        loss = classification_loss
        acc = self.accuracy(y_hat.max(dim = 1)[1].long(), y.long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return {'loss':loss, 
                'train_acc':acc,
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        acc = torch.tensor([x['train_acc'] for x in outputs]).mean()

        self.log('epoch_loss' , loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        loss = classification_loss
        acc = self.accuracy(y_hat.max(dim = 1)[1].long(), y.long())

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
        y_hat = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        loss = classification_loss
        acc = self.accuracy(y_hat.max(dim = 1)[1].long(), y.long())

        return {
                'acc': acc.item(),
                }

    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        acc = torch.tensor([x['acc'] for x in outputs]).mean()

        pbar = {'test_acc' : acc.item()}
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)