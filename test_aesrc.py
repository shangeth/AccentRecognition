from argparse import ArgumentParser
from multiprocessing import Pool
import os

from AESRC.dataset import AESRCDataset
from AESRC.lightning_model import Wav2VecModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.utils.data as data

from utils import get_temp_train_val

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_csv_path', type=str, default='/home/shangeth/AccentRecognition/AESRC2020TestData.csv')
    parser.add_argument('--timit_wav_len', type=int, default=16000*1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_size', type=float, default=128)
    parser.add_argument('--gpu', type=int, default="1")
    parser.add_argument('--n_workers', type=int, default=int(int(Pool()._processes)*0.5))
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=None)


    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on AESRC2020 Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # hyperparameters and details about the model 
    HPARAMS = {
        'data_csv_path' : hparams.data_csv_path,
        'data_wav_len' : hparams.timit_wav_len,
        'data_batch_size' : hparams.batch_size,
        'data_wav_augmentation' : 'Random Crop, Additive Noise',

        'training_optimizer' : 'Adam',
        'training_lr' : 1e-3,
        'training_lr_scheduler' : '-',

        'model_hidden_size' : hparams.hidden_size,
        'model_architecture' : 'wav2vec + soft-attention',
    }

    ## Testing Dataset
    test_set = AESRCDataset(
        csv_file = '/home/shangeth/AccentRecognition/AESRC2020TestData.csv',
        wav_len = HPARAMS['data_wav_len'],
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.n_workers
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))

    # Testing the Model
    if hparams.model_checkpoint:
        model = Wav2VecModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=HPARAMS)
        trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                            gpus=hparams.gpu, 
                            )

        print('\nTesting on TIMIT Dataset:\n')
        trainer.test(model, test_dataloaders=testloader)
    else:
        print('Model check point for testing is not provided!!!')