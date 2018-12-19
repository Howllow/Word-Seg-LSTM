import torch
import torch.nn as nn
import numpy as np
import codecs
from torchcrf import CRF

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.hidden_dim = hidden_dim
        self.tag_num = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True)
        self.crf = CRF(self.tag_num)

    def init_h(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim).cuda(),
                torch.randn(2, batch_size, self.hidden_dim).cuda())

    def lstm_out(self, embeds_packed, batch_size=1):
        self.hidden = self.init_h(batch_size)
        hidden2tag = nn.Linear(self.hidden_dim * 2, self.tag_num).cuda()
        if batch_size != 1:
            lstm_out, self.hidden = self.lstm(embeds_packed, self.hidden)
            unpacked = nn.utils.rnn.pad_packed_sequence(lstm_out)[0]
        else:
            embeds = self.word_embeds(embeds_packed).view(len(embeds_packed), 1, -1)
            unpacked = self.lstm(embeds.view(len(embeds_packed), 1, -1), self.hidden)[0]
        return hidden2tag(unpacked)

    '''def crf_forward(self, feats):
        init_alphas = torch.full((1, self.tag_num), -10000.).cuda()
        init_alphas[0][self.tag_to_ix['Start']] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_num):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_num)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1).cuda()
        terminal_var = forward_var + self.transitions[self.tag_to_ix['Stop']]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def score_sentence(self, feats, tags):
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([self.tag_to_ix['Start']], dtype=torch.long).cuda(), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['Stop'], tags[-1]]
        return '''

    def neg_log(self, sentence, tags, mask):
        batch_size = len(sentence)
        batch_max_len = len(sentence[0])
        seq_len = [len(seq) for seq in sentence]
        seq_len = torch.tensor(seq_len).cuda()

        for i in range(batch_size):
            zero = list(np.zeros(batch_max_len - len(sentence[i])))
            sentence[i] = sentence[i] + zero
            tags[i] = tags[i] + zero

        sentence = torch.tensor(sentence, dtype=torch.long).cuda()
        tags = torch.tensor(tags, dtype=torch.long).cuda()
        embeds = self.word_embeds(sentence).view(batch_max_len, batch_size, -1)
        sen_packed = nn.utils.rnn.pack_padded_sequence(embeds, seq_len)
        emission = self.lstm_out(sen_packed, batch_size)
        return self.crf(emission.view(batch_max_len, batch_size, self.tag_num),
                        tags.view(batch_max_len, batch_size), mask=mask.view(batch_max_len, batch_size))

    '''def viterbi(self, feats):
        backpointers = []
        init_vitvars = torch.full((1, self.tag_num), -10000.).cuda()
        init_vitvars[0][self.tag_to_ix['Start']] = 0
        forward_var = init_vitvars
        for feat in feats:
            backptrs_t = []
            vitvars_t = []

            for next_tag in range(self.tag_num):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                backptrs_t.append(best_tag_id)
                vitvars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(vitvars_t) + feat).view(1, -1).cuda()
            backpointers.append(backptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix['Stop']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for backptrs_t in reversed(backpointers):
            best_tag_id = backptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.pop()
        best_path.reverse()
        return path_score, best_path'''

    def forward(self, sentence):
        lstm_feats = self.lstm_out(sentence)
        tagseq = self.crf.decode(lstm_feats)
        return tagseq




