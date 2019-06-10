import os
import sys
import codecs
from crf import LinearCRF


def test_file(model_path, test_file_path):
    """Test model

    test file format
    今   B
    晚   E
    月   B
    色   E
    真   S
    美   S
    。   S
    <MUST SEPERATE BY SPACE LINE> 
    我   S

    output file format
    今   B   preTag
    晚   E   preTag
    """
    if not os.path.isfile(model_path) or not os.path.isfile(test_file_path):
        print("File don't exist!")
    model = LinearCRF()
    model.load(model_path)

    f = codecs.open(test_file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    sentences = []
    labels = []
    sentence = [model.start_char]
    label = [model.start_tag]
    for line in lines:
        if len(line) < 2:
                # sentence end
            sentence.append(model.end_char)
            label.append(model.end_tag)
            sentences.append(sentence)
            labels.append(label)
            sentence = [model.start_char]
            label = [model.start_tag]
        else:
            char, tag = line.split()
            sentence.append(char)
            label.append(tag)
            if char not in model.word_index:
                model.word_index[char] = model.nwords
                model.index_word[model.nwords] = char
                model.nwords += 1

    sentences_in_id = [[model.word_index[char] if char in model.word_index else model.nwords for char in s] for s in sentences]

    pre_tags = []
    for sen in sentences_in_id:
        pre_tag = ['s']
        pre_tag += model.inference_viterbi(sen)
        pre_tag += ['S']
        pre_tags.append(pre_tag)

    with open('test_result.txt', 'w+') as f:
        for sen_raw, sen_tag, sen_pre in zip(sentences, labels, pre_tags):
            for i in range(1, len(sen_raw)-1):
                f.write('{}\t{}\t{}\n'.format(sen_raw[i], sen_tag[i], sen_pre[i]))
            f.write('\n')

    print('Test finished!')


if __name__ == '__main__':
    test_file('linear_crf.model', '../data/test.data')

