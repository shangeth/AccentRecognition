from argparse import ArgumentParser
from multiprocessing import Pool
import os

from AESRC.dataset import AESRCDataset
from AESRC.lightning_model import LightningModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
import torch
import torch.utils.data as data
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import Config
if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--dataset_path', type=str, default=Config.dataset_path)
    parser.add_argument('--data_csv_path', type=str, default=Config.data_csv_path)
    parser.add_argument('--wav_len', type=int, default=Config.wav_len)
    parser.add_argument('--model_checkpoint', type=str, default=Config.model_checkpoint)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020TrainData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
    )

    ## Validation Dataset
    valid_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020ValData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
        is_train=False
    )

    ## Testing Dataset
    test_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020TestData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
        is_train=False
    )

    testloader = data.DataLoader(
        valid_set, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4,
        drop_last=True
    )
    print('Dataset Split (Test)=', len(test_set))

    # Testing the Model
    if hparams.model_checkpoint:
        with torch.no_grad():
            model = LightningModel.load_from_checkpoint(hparams.model_checkpoint).cuda()
            model.eval()

            embs = []
            labels = []
            for x, ys in tqdm(testloader):
            # for x, ys in tqdm(test_set):
                accent, attn_output = model(x.cuda())
                z_a = attn_output.cpu().detach().numpy()
                embs.append(z_a)
                labels.append(ys.view(-1).numpy())

            embs = np.vstack(embs).reshape(-1, 512)
            # print(labels)
            labels = np.concatenate(labels, 0).reshape(-1)

            writer = SummaryWriter()
            writer.add_embedding(embs, metadata=labels, tag='accent')
            writer.close()

    else:
        print('Model check point for testing is not provided!!!')