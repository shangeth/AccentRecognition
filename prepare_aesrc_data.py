import os
import shutil
import argparse
import glob
import pandas as pd
from tqdm import tqdm




def scp2csv_test(path = '/home/tlntu/Tut/profiling/accent_classification/data/testRmOOS/wav.scp'):
    df = pd.read_csv(path, delim_whitespace=True, names=['label', 'wav_path'])
    df.label = df.label.str.split("-").str.get(1).str.lower()
    df.wav_path = df.wav_path.str.split("/").str.get(-1)
    df['wav_path'] = '/home/tlntu/Tut/profiling/accent_classification/2020AESRC/TESTSET/wav/' + df['wav_path'].astype(str)
    df.to_csv('AESRC2020TestData.csv')
    print(f"Test data details are in AESRC2020TestData.csv with {len(df)} wav files")

def scp2csv_train_val(path='/home/tlntu/Tut/profiling/accent_classification/data/devv2/feats.scp'):
    df = pd.read_csv(path, delim_whitespace=True, names=['speaker', 'wav_path'])
    df_speaker = list(set(df.speaker.str.split("-").str.get(3).tolist()))
    return df_speaker

def get_val_train_csv():
    df = pd.read_csv('AESRC2020TrainData.csv')
    speakers_list = scp2csv_train_val()
    # print(speakers_list)
    val_df = df[df['speaker'].isin(speakers_list)].sample(frac=1).reset_index(drop=True)
    train_df = df[~df['speaker'].isin(speakers_list)].sample(frac=1).reset_index(drop=True)
    # print(train_df.head())
    # print(val_df.head())
    train_df.to_csv('AESRC2020TrainData.csv', index=False)
    n_speaker = len(list(set(train_df.speaker.tolist())))
    print(f"Training data details are in AESRC2020TrainData.csv with {len(train_df)} wav files and {n_speaker} Speakers")

    val_df.to_csv('AESRC2020ValData.csv', index=False)
    n_speaker = len(list(set(val_df.speaker.tolist())))
    print(f"Val data details are in AESRC2020ValData.csv with {len(val_df)} wav files and {n_speaker} Speakers")





if __name__ == "__main__":


    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--train_dataset_path', type=str, default='/home/tlntu/Tut/profiling/accent_classification/2020AESRC/')
    parser.add_argument('--test_dataset_path', type=str, default='/home/tlntu/Tut/profiling/accent_classification/2020AESRC/TESTSET/wav/')
    parser.add_argument('--train_csv_path', type=str, default='./AESRC2020TrainData.csv')
    parser.add_argument('--test_csv_path', type=str, default='./AESRC2020TestData.csv')
    hparams = parser.parse_args()

    TRAIN_DATA_PATH = hparams.train_dataset_path
    TEST_DATA_PATH = hparams.test_dataset_path



    accent_folders = ["Chinese_Speaking_English_Speech_Data", "Portuguese_Speaking_English_Speech_Data", "Indian_English_Speech_Data", "Korean_Speaking_English_Speech_Data",
    "American_English_Speech_Data", "British_English_Speech_Data", "Japanese_Speaking_English_Speech_Data", "Russian_Speaking_English_Speech_Data"]

    # Train Data ----------------------------------------------------------------------
    wav_paths = []
    labels = []
    speakers = []
    for type in tqdm(os.listdir(TRAIN_DATA_PATH)):
        if type in accent_folders:
            accent_class = type.split('_')[0].lower()
            for filename in glob.glob(os.path.join(TRAIN_DATA_PATH, type, '*/*.wav')):
                speaker = filename.split('/')[-2]
                wav_paths.append(filename)
                labels.append(accent_class)
                speakers.append(speaker)

    df_dict = {
        "wav_path" : wav_paths,
        "label" : labels,
        "speaker" : speakers
    }

    df = pd.DataFrame(df_dict)
    df.to_csv(hparams.train_csv_path, index=False) 
    n_speaker = len(list(set(df.speaker.tolist())))
    print(f"Training data details are in '{hparams.train_csv_path}' with {len(df)} wav files and {n_speaker} Speakers")


    # Test Data ----------------------------------------------------------------------


    scp2csv_test()
    get_val_train_csv()