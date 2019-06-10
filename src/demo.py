from crf import LinearCRF


class CN_Segementation(object):
    def __init__(self, model_path):
        self.model = LinearCRF()
        self.model.load(model_path)


    def chinese_word_segementation(sentence):
        sen = [self.model.start_char]
        sen += sentence
        sen += [self.model.end_char]
        sen = [self.model.word_index[word] if word in self.model.word_index else self.model.nwords for word in sen]
        tags = self.model.inference(sen)

        str_seg = ""
        for word, tag in zip(sentence, tags):
            str_seg += word
            if tag == 'S' or tag == 'E':
                str_seg += ' '
        return str_seg