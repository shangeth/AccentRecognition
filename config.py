class Config(object):
    dataset_path = '/home/n1900235d/DATASETS/AESRC2020/2020AESRC'
    data_csv_path = '/home/n1900235d/AccentRecognition/src/Dataset'
    wav_len = 16000*4
    batch_size = 150
    epochs = 200
    hidden_size = 512
    gpu = '-1'
    n_workers = 4
    dev = False
    model_checkpoint = 'logs/lstm_center_loss/version_0/checkpoints/epoch=149-step=31199.ckpt'
    noise_dataset_path = '/home/n1900235d/INTERSPEECH/NoiseDataset'
    lr = 1e-3
    run_name = 'lstm_center_loss'