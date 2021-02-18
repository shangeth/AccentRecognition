import torch
import torch.nn as nn
import wavencoder
import numpy as np

class Attention(nn.Module):
    def __init__(self, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.W = nn.Parameter(torch.Tensor(self.attn_dim, self.attn_dim), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(1, self.attn_dim), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.attn_dim)
        for weight in self.W:
            nn.init.uniform_(weight, -stdv, stdv)
        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, inputs, attn=False):
        inputs = inputs.transpose(1,2)
        batch_size = inputs.size(0)
        weights = torch.bmm(self.W.unsqueeze(0).repeat(batch_size, 1, 1), inputs)
        e = torch.tanh(weights.squeeze())

        e = torch.bmm(self.v.unsqueeze(0).repeat(batch_size, 1, 1), e)
        attentions = torch.softmax(e.squeeze(1), dim=-1)
        weighted = torch.mul(inputs, attentions.unsqueeze(1).expand_as(inputs))
        representations = weighted.sum(2).squeeze()
        if attn:
            return representations, attentions
        else:
            return representations


class Wav2VecClassifier(nn.Module):
    def __init__(self, hidden_size=128):
        super(Wav2VecClassifier, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=False)
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True

        lstm_inp = 512
        self.lstm = nn.LSTM(lstm_inp, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.accent_classifier = nn.Linear(hidden_size, 8)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)

        accent = self.accent_classifier(attn_output)
        return accent