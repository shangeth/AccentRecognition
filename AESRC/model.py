import torch
import torch.nn as nn
import wavencoder
import numpy as np


# class SelfAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass
        
    
#     def forward(self, x, attn=False):
#         w = torch.bmm(x, x.transpose(1,2)).sum(2) - (x*x).sum(2)
#         weights = torch.softmax(w, 1)
#         representations = torch.mul(x.transpose(1, 2), weights.unsqueeze(1).expand_as(x.transpose(1, 2))).sum(2).squeeze()
#         if attn:
#             return representations, attentions
#         else:
#             return representations

class Wav2VecClassifier(nn.Module):
    def __init__(self, hidden_size=128):
        super(Wav2VecClassifier, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True

        lstm_inp = 512
        self.lstm = nn.LSTM(lstm_inp, hidden_size, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(hidden_size, hidden_size)
        self.accent_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 8),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(x)
        accent = self.accent_classifier(attn_output)
        return accent, attn_output

class Wav2VecTransformer(nn.Module):
    def __init__(self, hidden_size=128):
        super(Wav2VecTransformer, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
        #     param.requires_grad = True
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.attention = nn.Sequential(
            wavencoder.layers.SoftAttention(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.accent_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 8, bias=False),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x =  self.transformer(x.transpose(1,2))
        attn_output = self.attention(x)
        accent = self.accent_classifier(attn_output)
        return accent, attn_output