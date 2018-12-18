from model import BiLSTM_CRF
import codecs
import torch
import random
import torch.optim as optim
embedding_dim = 6
hidden_dim = 2
batch_size = 64
line_end = '\r\n'

torch.manual_seed(10)
training_data = []

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


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) + 1

word_to_ix['UNF'] = len(word_to_ix)
#print (word_to_ix)

tag_to_ix = {'B': 0, 'M': 1, 'E': 2, 'S': 3, 'Start': 4, 'Stop': 5}
model = BiLSTM_CRF(word_to_ix.__len__(), tag_to_ix, embedding_dim, hidden_dim).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

def get_idxseq(seq, to_ix):
    idx = []
    for w in seq:
        if w in word_to_ix:
            idx.append(word_to_ix[w])
        else:
            idx.append(word_to_ix['UNF'])
    return torch.tensor(idx, dtype=torch.long)

#for i in range(0, len(sentences), 5000):
    #print(len(sentences[i].strip('\n')))


random.shuffle(training_data)

for epoch in range(2):
    print(epoch)
    for sentence, tag in training_data[:5000]:
        model.zero_grad()
        sentence_in = get_idxseq(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tag], dtype=torch.long)
        loss = model.neg_log(sentence_in.cuda(), targets.cuda())
        loss.backward()
        optimizer.step()

pre_res = []
for i in range(1000):
    sentence = test_data[i]
    sen_seq = get_idxseq(test_data[i], word_to_ix)
    pre_seq = model(sen_seq.cuda())[1]
    tmp_seq = ''
    for j in range(len(pre_seq)):
        if pre_seq[j] == 2 or pre_seq[j] == 3:
            tmp_seq += sentence[j] + '  '
        else:
            tmp_seq += sentence[j]
    tmp_seq += line_end
    pre_res.append(tmp_seq)

with codecs.open("./out.txt", 'w', encoding='UTF-8') as f:
    for i in range(1000):
        f.write(pre_res[i])










