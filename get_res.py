pre_res = []
for i in range(len(test_data)):
    sentence = test_data[i]
    sen_seq = get_idxseq(test_data[i], word_to_ix)
    pre_seq = model(torch.tensor(sen_seq, dtype=torch.long)[0].cuda())[0]
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