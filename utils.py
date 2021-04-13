import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os
from scipy import stats
from tqdm import tqdm
import glob 
import matplotlib.pyplot as plt
import torchaudio


def get_temp_train_val(csv_path, val_ratio=0.1, train_path='temp_train.csv', val_path='temp_val.csv'):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, stratify=df['label'], test_size=val_ratio, random_state=123)
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    print(f"\n\nData Split :\nTrain Stats :{Counter(train_df['label'])}")
    print(f"Val Stats : {Counter(val_df['label'])}\n")
    return train_path, val_path


def get_wav_len():
    PATH = '/home/tlntu/Tut/profiling/accent_classification/2020AESRC'
    for type in tqdm(os.listdir(PATH)):
        if type in accent_folders:
            for filename in glob.glob(os.path.join(PATH, type, '*/*.wav')):
                wav, _ = torchaudio.load(os.path.join(filename))
                lens.append(wav.shape[1])


    print(stats.describe(lens))
    plt.hist(lens)
    plt.savefig('train_wav.png')

    lens = []
    PATH = '/home/tlntu/Tut/profiling/accent_classification/2020AESRC/TESTSET/wav'
    for file in glob.glob(os.path.join(PATH, '*.wav')):
        wav, _ = torchaudio.load(os.path.join(PATH, file))
        lens.append(wav.shape[1])
    
    print(stats.describe(lens))
    plt.hist(lens)
    plt.savefig('test_wav.png')