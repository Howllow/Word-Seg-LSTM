import torch
import torch.nn as nn

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
        self.hidden = self.init_h()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tag_num)
        self.transitions = nn.Parameter(torch.randn(self.tag_num, self.tag_num))

        self.transitions.data[tag_to_ix['Start'], :] = -10000
        self.transitions.data[:, tag_to_ix['Stop']] = -10000

    def init_h(self):
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))

    def lstm_out(self, sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        self.hidden = self.init_h()
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
        return self.hidden2tag(lstm_out)

    def crf_forward(self, feats):
        init_alphas = torch.full((1, self.tag_num), -10000.)
        init_alphas[0][self.tag_to_ix['Start']] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_num):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_num)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix['Stop']]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix['Start']], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['Stop'], tags[-1]]
        return score

    def neg_log(self, sentence, tags):
        feats = self.lstm_out(sentence)
        forward_log_score = self.crf_forward(feats)
        gold_score = self.score_sentence(feats, tags)
        return forward_log_score - gold_score

    def viterbi(self, feats):
        backpointers = []
        init_vitvars = torch.full((1, self.tag_num), -10000.)
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
            forward_var = (torch.cat(vitvars_t) + feat).view(1, -1)
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
        return path_score, best_path

    def forward(self, sentence):
        lstm_feats = self.lstm_out(sentence)
        score, tagseq = self.viterbi(lstm_feats)
        return score, tagseq




