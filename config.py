class Config(object):
    # Dataset path containing folders of different Accent wav files
    # American_English_Speech_Data  Chinese_Speaking_English_Speech_Data  Japanese_Speaking_English_Speech_Data  Portuguese_Speaking_English_Speech_Data  TESTSET
    # British_English_Speech_Data   Indian_English_Speech_Data            Korean_Speaking_English_Speech_Data    Russian_Speaking_English_Speech_Data
    dataset_path = '/home/shangeth/DATASET/2020AESRC' 

    # Dataset folder of this repository
    data_csv_path = '/home/shangeth/AccentRecognition2/src/Dataset' 
    
    wav_len = 16000*3
    batch_size = 200
    epochs = 100
    hidden_size = 256
    gpu = '-1'
    n_workers = 10
    dev = False
    model_checkpoint = None
    # '/home/shangeth/AccentRecognition2/AccentRecognition/3kls7j0p/checkpoints/epoch=84.ckpt' - 67%
    noise_dataset_path = '/home/shangeth/AccentRecognition/noise_dataset'
    lr = 1e-3
    run_name = 'wav2vec-lstm-attn_beforeFinalDenseNoBias'
