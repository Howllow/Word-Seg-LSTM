import codecs

def ReadTrain():
    sentences = []
    tags = []
    with codecs.open("./train.txt", encoding='UTF-8') as f:
        lines = f.readlines()
        #lines.sort(key=lambda k:len(k) , reverse=False)
        tmp_tag = ''
        for i in range(0, lines.__len__()):
            tmp_sentence = lines[i]

            for words in tmp_sentence.split(' '):
                if words == '\r\n':
                    tmp_tag += 'F'
                elif words.__len__() == 1:
                    tmp_tag += 'S'
                else:
                    for j in range(0, words.__len__()):
                        if not j:
                            tmp_tag += 'B'
                        elif j == words.__len__() - 1:
                            tmp_tag += 'E'
                        else:
                            tmp_tag += 'M'

            tmp_sentence = lines[i].replace(' ', '')
            if i < 155 and (tmp_sentence[0] == '“' or tmp_sentence[0] == '’'):
                sentences.append(tmp_sentence[1:-1])
            else:
                sentences.append(tmp_sentence[:-1])
            tags.append(tmp_tag + 'F')

    with codecs.open("./newtrain.txt", 'w', encoding='UTF-8') as f:
        for i in range(0, sentences.__len__()):
            f.write(sentences[i])
            f.write(tags[i] + '\r')

    return sentences, tags

ReadTrain()


