"""
@author:jinqingzhe
@file: traom.py
@time: 2019/12/25
@contact: 1600012896@pku.edu.cn

"""
from model import BiLSTM_CRF
import codecs
import torch
import torch.optim as optim
import numpy as np
from torchcrf import CRF

embedding_dim = 250
hidden_dim = 1024
batch_size = 64
line_end = '\r\n'

torch.manual_seed(123)
training_data = []


def get_idxseq(seq, to_ix):
    idx = []
    for w in seq:
        if w in word_to_ix:
            idx.append(to_ix[w])
        else:
            idx.append(to_ix['UNK'])
    return idx


with codecs.open("./sentences.txt", 'r', encoding='UTF-8') as f:
    sentences = f.readlines()
with codecs.open("./tagseq.txt", 'r', encoding='UTF-8') as f:
    tags = f.readlines()
for i in range(0, len(sentences)):
    training_data.append((sentences[i].strip(line_end), tags[i].strip(line_end)))
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) + 1

word_to_ix['UNK'] = 0

tag_to_ix = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


batch_data = []
batch_num = len(sentences) // batch_size
res_size = len(sentences) % batch_size
cnt = 0
for i in range(0, batch_num):
    tmp_batch = []
    for j in range(batch_size):
        tmp_batch.append(training_data[cnt])
        cnt += 1
    batch_data.append(tmp_batch)

tmp_batch = []
for i in range(res_size):
    tmp_batch.append(training_data[cnt + i])
batch_data.append(tmp_batch)
#print(tmp_batch)
#print(batch_data[100])


mask = []

for batch in batch_data:
    tmp_mask = []
    batch_max_len = len(batch[0][0])
    for data in batch:
        one = np.ones(len(data[0]))
        zero = np.zeros(batch_max_len - len(data[0]))
        tmp_mask.append(list(one) + list(zero))
    mask.append(tmp_mask)

crf = CRF(len(tag_to_ix))

crf.start_transitions.data = torch.randn(len(tag_to_ix))
crf.end_transitions.data = torch.randn(len(tag_to_ix))
crf.transitions.data = torch.randn(len(tag_to_ix), len(tag_to_ix))
crf.end_transitions.data[tag_to_ix['B']] = -1000000
crf.end_transitions.data[tag_to_ix['M']] = -1000000
#crf.end_transitions.data[tag_to_ix['P']] = -1000000
crf.start_transitions.data[tag_to_ix['M']] = -1000000
crf.start_transitions.data[tag_to_ix['E']] = -1000000
#crf.start_transitions.data[tag_to_ix['P']] = -1000000
crf.transitions.data[tag_to_ix['B'], tag_to_ix['S']] = -1000000
crf.transitions.data[tag_to_ix['B'], tag_to_ix['B']] = -1000000
crf.transitions.data[tag_to_ix['M'], tag_to_ix['B']] = -1000000
crf.transitions.data[tag_to_ix['M'], tag_to_ix['S']] = -1000000
crf.transitions.data[tag_to_ix['S'], tag_to_ix['M']] = -1000000
crf.transitions.data[tag_to_ix['S'], tag_to_ix['E']] = -1000000
crf.transitions.data[tag_to_ix['E'], tag_to_ix['M']] = -1000000
crf.transitions.data[tag_to_ix['E'], tag_to_ix['E']] = -1000000
#crf.transitions.data[tag_to_ix['P'], :] = -1000000
# crf.transitions.data[:, tag_to_ix['P']] = -1000000


print(crf.transitions.data)
model = BiLSTM_CRF(word_to_ix.__len__(), tag_to_ix, embedding_dim, hidden_dim, crf).cuda()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


for epoch in range(30):
    print(epoch)
    for i in range(len(batch_data)):
        model.zero_grad()
        sen_batch = []
        tag_batch = []
        for data in batch_data[len(batch_data)-1-i]:
            sen_batch.append(get_idxseq(data[0], word_to_ix))
            tag_batch.append([tag_to_ix[t] for t in data[1]])
        loss = model.neg_log(sen_batch, tag_batch, torch.tensor(mask[len(batch_data)-1-i], dtype=torch.long).cuda())
        loss.backward()
        print(loss)
        optimizer.step()
    torch.save(model.state_dict(), './params5.pkl')
print(crf.transitions.data)
'''''
model.load_state_dict(torch.load('./params3.pkl'))
print(crf.transitions.data)
with codecs.open('./newtrain.txt', encoding='UTF-8') as f:
    train = f.readlines()
    for i in range(20):
        if i % 2 != 0:
            sen = train[i].strip(line_end)
            sen = get_idxseq(sen, word_to_ix)
            print(model(torch.tensor(sen, dtype=torch.long).cuda())[0])
'''''

with codecs.open("./test.txt", 'r', encoding='UTF-8') as f:
    test_data = f.readlines()

for i in range(test_data.__len__()):
    test_data[i] = test_data[i].strip(line_end)

pre_res = []
for i in range(len(test_data)):
    sentence = test_data[i]
    sen_seq = get_idxseq(test_data[i], word_to_ix)
    pre_seq = model(torch.tensor(sen_seq, dtype=torch.long).cuda())[0]
    tmp_seq = ''
    for j in range(len(pre_seq)):
        if pre_seq[j] == 2 or pre_seq[j] == 3:
            tmp_seq += sentence[j] + '  '
        else:
            tmp_seq += sentence[j]
    tmp_seq += line_end
    pre_res.append(tmp_seq)


with codecs.open("./big_out.txt", 'w', encoding='UTF-8') as f:
    for i in range(len(pre_res)):
        f.write(pre_res[i])









