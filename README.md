# AccentRecognition



This Repository contains the code for detecting the accent of a speaker with their speech signal. The repository experiments with AESRC2020 Dataset.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Usage

### Noise Dataset
```
# Noise Dataset is optional, but recommended
# 1. Download noise Dataset with wavencoder
# 2. Update config.py with the noise_dataset_path = 'folder-to-download-noise-data'

import wavencoder
wavencoder.utils.download_noise_dataset('folder-to-download-noise-data', sample_rate='16k', download_all=True)
```

### Config
```
Edit the config.py for configurations on data path, lr, batch size, ... etc.
Or you can optionally give as a command line parameters for train and test
```

### Download the dataset
```
# AESRC2020 Dataset
```

### Prepare the dataset for training and testing(Not Required)
```
# prepare csv with path to wav files and labels for training
# Train and Test csv files are available at Dataset folder

python prepare_aesrc_data.py --path='path to aesrc wav data folder'
```

### Training(Dev Model, to make sure everything is set as expected for training) 
```
python train.py --dev=True 
```

### Training(also check for other arguments in the train.py file)
```
python train.py 
```

### Test the Model(also check for other arguments in the test.py file)
```
python test.py  
# Edit config.py for trained model checkpoint
# or add --model_checkpoint='path to saved model checkpoint'
```

## Results

|                           Model                          	| Experiment Run                                                                   	| Test wav length 	| Test Accuracy 	|
|:--------------------------------------------------------:	|----------------------------------------------------------------------------------	|-----------------	|:-------------:	|
|                   MFCC_1DCNN_LSTM_Attn                   	| [Wandb Run](https://wandb.ai/shangeth/AccentRecognition?workspace=user-shangeth) 	| 3s              	|    0.34078    	|
| Mel_Spectrogram_1DCNN_LSTM_Attn                          	| [Wandb Run](https://wandb.ai/shangeth/AccentRecognition?workspace=user-shangeth) 	| 3s              	|     0.3751    	|
|     wav2vec_LSTM_Attn_CenterLoss (center after attn)     	| [Wandb Run](https://wandb.ai/shangeth/AccentRecognition?workspace=user-shangeth) 	| 3s              	|     0.6123    	|
|                                                          	|                                                                                  	| 4s              	|    0.62008    	|
|                                                          	|                                                                                  	| -1              	|     0.6279    	|
| wav2vec_LSTM_Attn_CenterLoss (center before final dense) 	| [Wandb Run](https://wandb.ai/shangeth/AccentRecognition?workspace=user-shangeth) 	| -1              	|     0.6161    	|
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
