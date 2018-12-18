import codecs

def ReadTrain():
    sentences = []
    tags = []
    with codecs.open("./train.txt", encoding='UTF-8') as f:
        lines = f.readlines()

    for i in range(0, lines.__len__()):
        lines[i] = lines[i].strip('\r\n')

    for i in range(0, lines.__len__()):
        tmp_tag = ''
        tmp_sentence = ''
        if lines[i].__len__() < 2:
            continue
        for words in lines[i].split('  '):
            tmp_sentence += words
            if words.__len__() == 1:
                tmp_tag += 'S'
            elif words.__len__() > 1:
                for j in range(0, words.__len__()):
                    if not j:
                        tmp_tag += 'B'
                    elif j == words.__len__() - 1:
                        tmp_tag += 'E'
                    else:
                        tmp_tag += 'M'
        sentences.append(tmp_sentence)
        tags.append(tmp_tag)

    tags.sort(key=lambda x: len(x), reverse=False)
    sentences.sort(key=lambda x: len(x), reverse=False)

    with codecs.open("./tagseq.txt", 'w', encoding='UTF-8') as f:
        for i in range(0, tags.__len__() - 10):
            f.write(tags[len(tags) - 1 - i] + '\n')

    with codecs.open("./sentences.txt", 'w', encoding='UTF-8') as f:
        for i in range(0, sentences.__len__() - 10):
            f.write(sentences[len(sentences) - 1 - i] + '\n')

ReadTrain()


