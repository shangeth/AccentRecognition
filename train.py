from argparse import ArgumentParser
from multiprocessing import Pool
import os

from AESRC.dataset import AESRCDataset
from AESRC.lightning_model import LightningModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
    parser.add_argument('--noise_dataset_path', type=str, default=Config.noise_dataset_path)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Training Model on AESRC2020 Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = AESRCDataset(
        csv_file = os.path.join(hparams.data_csv_path, 'AESRC2020TrainData.csv'),
        dataset_path = hparams.dataset_path,
        wav_len = hparams.wav_len,
        noise_dataset_path = hparams.noise_dataset_path
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

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))


    #Training the Model
    logger = TensorBoardLogger('logs', name=Config.run_name)

    model = LightningModel(hparams.hidden_size, Config.lr)

    checkpoint_callback = ModelCheckpoint(
        monitor='v_acc', 
        mode='max',
        verbose=1)

    trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                        gpus=hparams.gpu, 
                        max_epochs=hparams.epochs, 
                        checkpoint_callback=checkpoint_callback,
                        # callbacks=[
                        #     EarlyStopping(
                        #         monitor='v_loss',
                        #         min_delta=0.00,
                        #         patience=10,
                        #         verbose=True,
                        #         mode='min'
                        #         )
                        # ],
                        logger=logger,
                        resume_from_checkpoint=hparams.model_checkpoint,
                        distributed_backend='ddp'
                        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    print('\n\nTesting the model with checkpoint -', checkpoint_callback.best_model_path)
    model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=testloader)

    logger.log_hyperparams(
        params=dict(
            batch_size = hparams.batch_size,
            augmentation = 'Random Crop, Additive Noise, Clipping',
            optimizer = 'DiffGrad',
            lr = Config.lr,
            lr_Scheduler = '',
            architecture = 'wav2vecNoFinetune-lstm-attn-centerLoss',
            hidden_dim = 128
        ),
        metrics=test_result)
