import os
import shutil
import argparse
import glob
import pandas as pd
from tqdm import tqdm

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
for type in tqdm(os.listdir(TRAIN_DATA_PATH)):
    if type in accent_folders:
        accent_class = type.split('_')[0].lower()
        for filename in glob.glob(os.path.join(TRAIN_DATA_PATH, type, '*/*.wav')):
            wav_paths.append(filename)
            labels.append(accent_class)

df_dict = {
    "wav_path" : wav_paths,
    "label" : labels
}

df = pd.DataFrame(df_dict)
df.to_csv(hparams.train_csv_path) 

print(f"Training data details are in '{hparams.train_csv_path}' with {len(df)} wav files")

# Test Data ----------------------------------------------------------------------