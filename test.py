from argparse import ArgumentParser
from multiprocessing import Pool
import os

from AESRC.dataset import AESRCDataset
from AESRC.lightning_model import LightningModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.utils.data as data

from config import Config
if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--dataset_path', type=str, default=Config.dataset_path)
    parser.add_argument('--data_csv_path', type=str, default=Config.data_csv_path)
    parser.add_argument('--wav_len', type=int, default=Config.wav_len)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--hidden_size', type=float, default=Config.hidden_size)
    parser.add_argument('--gpu', type=int, default=Config.gpu)
    parser.add_argument('--n_workers', type=int, default=Config.n_workers)
    parser.add_argument('--dev', type=str, default=Config.dev)
    parser.add_argument('--model_checkpoint', type=str, default=Config.model_checkpoint)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on AESRC2020 Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020TrainData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.n_workers
    )
    ## Validation Dataset
    valid_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020ValData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers
    )
    ## Testing Dataset
    test_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020TestData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers
    )

    print('Dataset Split (Test)=', len(test_set))

    # Testing the Model
    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint)
        model.eval()
        trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                            gpus=hparams.gpu, 
                            )
        # print('\nTesting on AESRC2020 Train Dataset:\n')
        # trainer.test(model, test_dataloaders=trainloader)
        
        # print('\nTesting on AESRC2020 Val Dataset:\n')
        # trainer.test(model, test_dataloaders=valloader)

        print('\nTesting on AESRC2020 Test Dataset:\n')
        trainer.test(model, test_dataloaders=testloader)
    else:
        print('Model check point for testing is not provided!!!')