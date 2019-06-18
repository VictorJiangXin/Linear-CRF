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
    sentence = []
    label = []
    for line in lines:
        if len(line) < 2:
                # sentence end
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            char, tag = line.split()
            sentence.append(char)
            label.append(tag)

    pre_tags = [model.inference_viterbi(sen) for sen in sentences]

    with open('test_result.txt', 'w+') as f:
        for sen, sen_tag, sen_pre in zip(sentences, labels, pre_tags):
            for i in range(len(sen)):
                f.write('{}\t{}\t{}\n'.format(sen[i], sen_tag[i], sen_pre[i]))
            f.write('\n')

    print('Test finished!')


if __name__ == '__main__':
    test_file('linear_crf.model', '../data/mst_test.data')

