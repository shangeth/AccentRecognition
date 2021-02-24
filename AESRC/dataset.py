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
        wav_len=48000,
        is_train=True,
        noise_dataset_path=None
        ):

        self.csv_file = csv_file
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

        self.train_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random')
            ])

        # Pad/Crop from the center
        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len)
            ])

        if self.noise_dataset_path:
            self.noise_transform = wavencoder.transforms.AdditiveNoise(self.noise_dataset_path)

        self.spectral_transform = torchaudio.transforms.MelSpectrogram()
        self.db_transform = torchaudio.transforms.AmplitudeToDB()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        wav, _ = torchaudio.load(row['wav_path'])
        label = self.accent_dict[row['label']]

        if self.is_train:
            wav = self.train_transform(wav)

            # apply noise to wav 50% of time
            if self.noise_dataset_path and random.random() < 0.5:
                wav = self.noise_transform(wav)
        else:
            wav = self.test_transform(wav)
        

        if type(wav).__module__ == np.__name__:
            wav = torch.tensor(wav)

        # wav = self.db_transform(self.spectral_transform(wav)).squeeze(0)
        # print(wav.max(), wav.min(), wav.mean())
        return wav, label