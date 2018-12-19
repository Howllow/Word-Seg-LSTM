from model import BiLSTM_CRF
import codecs
import torch
import random
import torch.optim as optim
import numpy as np

embedding_dim = 15
hidden_dim = 256
batch_size = 64
line_end = '\r\n'

torch.manual_seed(50)
training_data = []

def get_idxseq(seq, to_ix):
    idx = []
    for w in seq:
        if w in word_to_ix:
            idx.append(to_ix[w])
        else:
            idx.append(to_ix['UNF'])
    return idx


with codecs.open("./sentences.txt", 'r', encoding='UTF-8') as f:
    sentences = f.readlines()
with codecs.open("./tagseq.txt", 'r', encoding='UTF-8') as f:
    tags = f.readlines()
with codecs.open("./test.txt", 'r', encoding='UTF-8') as f:
    test_data = f.readlines()
for i in range(test_data.__len__()):
    test_data[i] = test_data[i].strip(line_end)
for i in range(0, len(sentences)):
    training_data.append((sentences[i].strip(line_end), tags[i].strip(line_end)))


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) + 1

word_to_ix['UNF'] = len(word_to_ix)

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

model = BiLSTM_CRF(word_to_ix.__len__(), tag_to_ix, embedding_dim, hidden_dim)#.cuda()
optimizer = optim.ASGD(model.parameters(), lr=0.001, weight_decay=1e-4)


for epoch in range(3):
    print(epoch)
    for i in range(len(batch_data) - 1000, len(batch_data)):
        model.zero_grad()
        sen_batch = []
        tag_batch = []
        for data in batch_data[i]:
            sen_batch.append(get_idxseq(data[0], word_to_ix))
            tag_batch.append([tag_to_ix[t] for t in data[1]])
        loss = model.neg_log(sen_batch, tag_batch, torch.tensor(mask[i], dtype=torch.long))#.cuda())
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), './params1.pkl')
print(model(torch.tensor(get_idxseq(sentences[len(sentences) - 1], word_to_ix), dtype=torch.long)))
pre_res = []
for i in range(5):
    sentence = test_data[i]
    sen_seq = get_idxseq(test_data[i], word_to_ix)
    pre_seq = model(torch.tensor(sen_seq, dtype=torch.long))[0]#.cuda())[0]
    tmp_seq = ''
    for j in range(len(pre_seq)):
        if pre_seq[j] == 2 or pre_seq[j] == 3:
            tmp_seq += sentence[j] + '  '
        else:
            tmp_seq += sentence[j]
    tmp_seq += line_end
    pre_res.append(tmp_seq)

with codecs.open("./out.txt", 'w', encoding='UTF-8') as f:
    for i in range(len(pre_res)):
        f.write(pre_res[i])










