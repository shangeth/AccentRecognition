import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def get_temp_train_val(csv_path, val_ratio=0.1, train_path='temp_train.csv', val_path='temp_val.csv'):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, stratify=df['label'])
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    print(f"\n\nData Split :\nTrain Stats :{Counter(train_df['label'])}")
    print(f"Val Stats : {Counter(val_df['label'])}\n")
    return train_path, val_path


def scp2csv(path = '/home/tlntu/Tut/profiling/accent_classification/data/testRmOOS/wav.scp'):
    df = pd.read_csv(path, delim_whitespace=True, names=['label', 'wav_path'])
    df.label = df.label.str.split("-").str.get(1).str.lower()
    df.wav_path = df.wav_path.str.split("/").str.get(-1)
    df['wav_path'] = '/home/tlntu/Tut/profiling/accent_classification/2020AESRC/TESTSET/wav/' + df['wav_path'].astype(str)

    df.to_csv('AESRC2020TestData.csv')