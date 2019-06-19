from crf import LinearCRF


class Segmentation(object):
    def __init__(self, model_path='model/linear_crf.model'):
        self.model = LinearCRF()
        self.model.load(model_path)


    def seg(self, sentence):
        sentence.strip()
        tags = self.model.inference_viterbi(sentence)

        str_seg = ""
        for word, tag in zip(sentence, tags):
            str_seg += word
            if tag == 'S' or tag == 'E':
                str_seg += ' '
        result = str_seg.split()
        return result