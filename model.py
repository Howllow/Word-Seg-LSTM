"""
@author:jinqingzhe
@file: model.py
@time: 2019/12/25
@contact: 1600012896@pku.edu.cn

"""

import torch
import torch.nn as nn
import numpy as np


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, crf):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.hidden_dim = hidden_dim
        self.tag_num = len(tag_to_ix)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tag_num).cuda()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True)
        self.crf = crf

    def init_h(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim).cuda(),
                torch.randn(2, batch_size, self.hidden_dim).cuda())

    def lstm_out(self, embeds_packed, batch_size=1):
        self.hidden = self.init_h(batch_size)
        if batch_size != 1:
            lstm_out, self.hidden = self.lstm(embeds_packed, self.hidden)
            unpacked = nn.utils.rnn.pad_packed_sequence(lstm_out)[0]
        else:
            embeds = self.word_embeds(embeds_packed).view(len(embeds_packed), 1, -1)
            unpacked = self.lstm(embeds, self.hidden)[0]
        emission = self.hidden2tag(unpacked)
        return emission

    def neg_log(self, sentence, tags, mask):
        batch_size = len(sentence)
        batch_max_len = len(sentence[0])
        seq_len = [len(seq) for seq in sentence]
        seq_len = torch.tensor(seq_len).cuda()

        for i in range(batch_size):
            zero = list(np.zeros(batch_max_len - len(sentence[i])))
            one = list(np.ones(batch_max_len - len(sentence[i])))
            sentence[i] = sentence[i] + zero
            tags[i] = tags[i] + one

        sentence = torch.tensor(sentence, dtype=torch.long).cuda()
        tags = torch.tensor(tags, dtype=torch.long).cuda()
        embeds = torch.transpose(self.word_embeds(sentence), 0, 1)
        sen_packed = nn.utils.rnn.pack_padded_sequence(embeds, seq_len)
        emission = self.lstm_out(sen_packed, batch_size)
        tags = torch.transpose(tags, 0, 1)
        mask = torch.transpose(mask, 0, 1)
        return -self.crf(emission, tags, mask=mask)

    def forward(self, sentence):
        lstm_feats = self.lstm_out(sentence)
        tagseq = self.crf.decode(lstm_feats)
        return tagseq



