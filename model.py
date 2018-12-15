import torch
import torch.nn as nn
import torch.autograd as autograd

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item


def get_idxseq(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


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
        self.hidden = self.init_h()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tag_num)
        self.transitions = nn.Parameter(torch.randn(self.tag_num, self.tag_num))

    def init_h(self):
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))

    def Get_Emission(self, sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        self.hidden = self.init_h()
        lstm_out, self.hidden = self.lstm(embeds, self.hidden_dim)
        lstm_out = lstm_out.view(len(sentence), 2 * self.hidden_dim)
        return self.hidden2tag(lstm_out)





