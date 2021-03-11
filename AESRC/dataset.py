from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
import random
import warnings
warnings.simplefilter("ignore", UserWarning)

class AESRCDataset(Dataset):
    def __init__(self,
        csv_file,
        dataset_path,
        wav_len=48000,
        is_train=True,
        noise_dataset_path=None
        ):

        self.csv_file = csv_file
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train
        self.wav_len = wav_len
        self.noise_dataset_path = noise_dataset_path

        self.accent_dict = {
            'american': 0, 
            'british': 1, 
            'chinese': 2, 
            'indian': 3, 
            'japanese': 4, 
            'korean': 5, 
            'portuguese': 6, 
            'russian': 7
            }

        if self.noise_dataset_path:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random'),
                wavencoder.transforms.AdditiveNoise(self.noise_dataset_path, p=0.5),
                wavencoder.transforms.Clipping(p=0.2),
                ])
        else:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random'),
                wavencoder.transforms.Clipping(p=0.2),
                ])

        # Pad/Crop from the center
        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        wav, _ = torchaudio.load(os.path.join(self.dataset_path, row['wav_path']))
        label = self.accent_dict[row['label']]

        if self.is_train:
            wav = self.train_transform(wav)
        else:
            wav = self.test_transform(wav)
        

        if type(wav).__module__ == np.__name__:
            wav = torch.tensor(wav)
        return wav, label