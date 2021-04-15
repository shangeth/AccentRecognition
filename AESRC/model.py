import torch
import torch.nn as nn
import wavencoder
import numpy as np

class Wav2VecClassifier(nn.Module):
    def __init__(self, hidden_size=128):
        super(Wav2VecClassifier, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
        #     param.requires_grad = True

        lstm_inp = 512
        self.lstm = nn.LSTM(lstm_inp, hidden_size, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 8)
        )
        self.log_softmax = nn.LogSoftmax(1)


    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(x)
        accent = self.classifier(attn_output)
        accent = self.log_softmax(accent.view(batch_size, -1)).view(batch_size, -1)
        return accent, attn_output

# class Wav2VecClassifier(nn.Module):
#     def __init__(self, hidden_size=128):
#         super(Wav2VecClassifier, self).__init__()
#         self.encoder = wavencoder.models.Wav2Vec(pretrained=True)

#         for param in self.encoder.parameters():
#             param.requires_grad = False

#         # for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
#         #     param.requires_grad = True

#         lstm_inp = 512
#         self.lstm = nn.LSTM(lstm_inp, hidden_size, batch_first=True)
#         self.accent_classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#         )
#         self.classifier = nn.Linear(hidden_size, 8)

#         self.log_softmax = nn.LogSoftmax(1)

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.encoder(x)
#         out, (hidden, _) = self.lstm(x.transpose(1,2))
#         accent = self.accent_classifier(out)
#         attn_output = accent.mean(1)
#         accent = self.classifier(accent)
#         accent = self.log_softmax(accent.mean(1))
#         return accent, attn_output

# class Wav2VecSpectralClassifier(nn.Module):
#     def __init__(self, hidden_size=128):
#         super(Wav2VecSpectralClassifier, self).__init__()
#         self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         # for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
#         #     param.requires_grad = False

#         self.spectral_encoder = nn.Sequential(
#             nn.Conv1d(128, 128, 5),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 128, 4),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 64, 4),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#         )
        
#         self.comb_encoder = nn.Sequential(
#             nn.Conv1d(512+64, hidden_size, 3),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(inplace=True),
#             # nn.Conv1d(hidden_size, hidden_size, 3),
#             # nn.BatchNorm1d(hidden_size),
#             # nn.ReLU(inplace=True),
#         )
#         self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

#         self.accent_classifier = nn.Sequential(
#             nn.Linear(hidden_size, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 8),
#             nn.LogSoftmax(1)
#         )

#     def forward(self, x, xs):
#         w2v_features = self.encoder(x)
#         spec_features = self.spectral_encoder(xs.squeeze(1))
#         comb_features = torch.cat((spec_features, w2v_features), 1)
#         comb_features = self.comb_encoder(comb_features)

#         comb_features, _ = self.lstm(comb_features.transpose(1,2))
#         # attn_output = comb_features.mean(2)
#         attn_output = comb_features[:, -1, :]

#         accent = self.accent_classifier(attn_output)
#         return accent, attn_output


class Wav2VecSpectralClassifier(nn.Module):
    def __init__(self, hidden_size=128):
        super(Wav2VecSpectralClassifier, self).__init__()

        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(128, 128, 5, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden_size, 4),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.accent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        spec_features = self.spectral_encoder(x.squeeze(1))
        comb_features, _ = self.lstm(spec_features.transpose(1,2))
        attn_output = comb_features[:, -1, :]
        accent = self.accent_classifier(attn_output)
        return accent, attn_output
