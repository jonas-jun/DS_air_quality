import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AiR_predictor(nn.Module):
    def __init__(self, input_dim, embed_dim, rnn_dim, fc_dim, num_classes, bidirectional, opt):
        super(AiR_predictor, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.embedding = nn.Linear(self.input_dim, self.embed_dim)
        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.rnn_dim, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.rnn_dim*(self.bidirectional+1), self.num_classes)
        self.device = opt.device

    def forward(self, input_ids):
        embedded = self.embedding(input_ids) # float type!!!, dropout
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = outputs[:, -1, :]
        output = self.fc(outputs)
        return output


class AiR_predictor_att(nn.Module):
    def __init__(self, input_dim, embed_dim, rnn_dim, fc_dim, num_classes, bidirectional, opt):
        super(AiR_predictor_att, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.embedding = nn.Linear(self.input_dim, self.embed_dim)
        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.rnn_dim, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.fc_dim, self.num_classes)
        self.device = opt.device

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.rnn_dim))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.rnn_dim, self.fc_dim))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)


    def forward(self, input_ids):
        embedded = self.embedding(input_ids) # float type!!!, dropout
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = self.attention(outputs)
        output = self.fc(outputs)
        return output
    
    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

# for insert mode