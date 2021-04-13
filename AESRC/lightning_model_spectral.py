import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy

import numpy as np
import pandas as pd
from AESRC.model import Wav2VecSpectralClassifier as Model
import math
import torch_optimizer
from AESRC.centerloss import CenterLoss

class LightningModel(pl.LightningModule):
    def __init__(self, hidden_size=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        hidden_size = hidden_size

        self.model = Model(hidden_size=hidden_size)
        self.center_loss = CenterLoss(num_classes=8, feat_dim=hidden_size)

        self.classification_criterion = nn.NLLLoss()
        self.accuracy = Accuracy()

        self.lr = lr
        self.alpha = 0.5
        self.lr_cent = 0.5
        self.finetune_lr = 1e-5

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        # encoder_params = self.model.encoder.parameters()
        model_params = self.model.parameters()
        other_params = self.model.parameters()
        # [para for para in list(model_params) if para not in list(encoder_params)]
        center_loss_params = self.center_loss.parameters()

        grouped_parameters = [
            # {"params": encoder_params, 'lr': self.finetune_lr},
            {"params": other_params , 'lr': self.lr},
            {"params": center_loss_params, 'lr': self.lr_cent},
        ]

        optimizer = torch_optimizer.DiffGrad(
            grouped_parameters
        )

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, attn_output = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())
        center_loss = self.center_loss(attn_output, y)
        loss = classification_loss + self.alpha * center_loss

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/cls', classification_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/center', center_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, attn_output = self(x)

        classification_loss = self.classification_criterion(y_hat.float(), y.long())
        loss = classification_loss
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, attn_output = self(x)

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
            'test/loss': loss.item(),
            'test/acc' : acc.item()
        }
        # self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
        return pbar