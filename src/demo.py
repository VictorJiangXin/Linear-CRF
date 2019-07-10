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


seg = Segmentation('model/linear_crf.model')
sen1 = '今晚的月色真美呀！'
sen2 = '生命在于奋斗！'
sen3 = '小哥哥，别复习了，来玩吧！'

print('测试句:', sen1)
print('分词后:', seg.seg(sen1))

print('测试句:', sen2)
print('分词后:', seg.seg(sen2))

print('测试句:', sen3)
print('分词后:', seg.seg(sen3))