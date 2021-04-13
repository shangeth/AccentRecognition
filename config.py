class Config(object):
    dataset_path = '/home/shangeth/DATASET/2020AESRC'
    data_csv_path = '/home/shangeth/AccentRecognition2/src/Dataset'
    wav_len = 16000*3
    batch_size = 200
    epochs = 100
    hidden_size = 128
    gpu = '-1'
    n_workers = 10
    dev = False
    model_checkpoint = None
    noise_dataset_path = '/home/shangeth/AccentRecognition/noise_dataset'
    lr = 1e-3
    run_name = 'wav2vec-lstm-attn_beforeFinalDenseNoBias'
