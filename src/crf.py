# MIT License
# 
# Copyright (c) 2019 Jiang Xin, jiangxin.hust@foxmail.com
# 


import os
import sys
import time

import pickle
import codecs
import pickle
import numpy as np
from scipy import optimize

class LinearCRF(object):
    """Simple implementation of linear-chain CRF for Chinese word segmentation task.
    
    This class is a simple implementation of linear-chain conditional random field
    for Chinese word segementation task. So many function are designed for this
    particular task.
    There are two types of feature templates:
    Unigram template: first character, 'U'
    ('U', pos, word, tag)
    Bigram template: first character, 'B'
    ('B', pos, word, tag1, tag2)
    You can get more information from my blog (PS. the blog is in Chinese)
    https://victorjiangxin.github.io/Chinese-Word-Segmentation/
    """

    def __init__(self):
        self.ntags = 4  # {'B', 'I', 'E', 'S'}
        self.index_tag = {0:'B', 1:'I', 2:'E', 3:'S'}
        self.tag_index = {'B':0, 'I':1, 'E':2, 'S':3}

        self.start_char = '<START>'
        self.end_char = '<END>'
        self.start_tag = 'S'
        self.end_tag = 'S'
        self.start_index = self.tag_index[self.start_tag]
        self.end_index = self.tag_index[self.end_tag]

        self.nwords = 2
        self.index_word = {0:self.start_char, 1:self.end_char}    # {0:'今', 1:'晚', ..., n:'美'}
        self.word_index = {self.start_char:0, self.end_char:1}    # {'今':0, '晚':1, ...m, '美':n}

        self.U_feature_pos = [-2, -1, 0, 1, 2]
        self.B_feature_pos = [0]
        self.nU_features = 5
        self.nB_features = 1

        self.nfeatures = 0
        self.feature_index = {} # {('U', 0, word_id, tag_id):0}
        self.index_feature = {} # {0:('U', 0, word_id, tag_id)}

        self.nweights = 0
        self.weights = np.zeros(self.nweights)
        self.theta = 1e-3   # theta should in the range of (1e-6 ~ 1e-3)


    def feature_at(self, k, x, yi_1, yi, i):
        """Get f_k(yt_1, yt, x, i).

        Args:
            k: (int) the Kth feature
            x: (list(int)) word list [word_index['<START'>], word_index['今'],]
            yi_1: tag of y_[i-1]
            yi: tag of yi
            i: (int) index

        Return:
            1 or 0
        """
        feature = self.index_feature[k]

        if feature[0] == 'U':
            _, pos, word, tag = feature
            if i + pos >= 0 and i + pos <= len(x) - 1 and yi == tag and x[i + pos] == word:
                return 1
        elif feature[0] == 'B':
            _, pos, word, tag1, tag2 = feature
            if i + pos >= 1 and x[i+pos] == word and yi_1 == tag1 and yi == tag2:
                return 1

        return 0


    def log_M_at(self, x, yi_1, yi, i):
        """Calc log M(yi_1, yi|x) = W.F_i(yi_1, yi|x)
        """
        activate_feature = []
        for pos in self.U_feature_pos:
            if pos + i >= 0 and pos + i < len(x):
                feature = ('U', pos, x[pos + i], yi)
                if feature in self.feature_index:
                    activate_feature.append(self.feature_index[feature])

        for pos in self.B_feature_pos:
            if pos + i >= 1 and pos + i < len(x):
                feature = ('B', pos, x[pos + i], yi_1, yi)
                if feature in self.feature_index:
                    activate_feature.append(self.feature_index[feature])
        return self.weights[activate_feature].sum()


    def log_M(self, x):
        """Get log probablity matrix M(x)

        Return:
            M(x): tensor(nwords_x+2, ntags, ntags) M(0) means nothing
        """
        nwords_x = len(x) - 2   # x include '<START>' and '<END>'
        M = np.zeros((nwords_x + 2, self.ntags, self.ntags))
        for i in range(1, nwords_x + 2):
            for tag1 in range(self.ntags):
                for tag2 in range(self.ntags):
                    M[i, tag1, tag2] = self.log_M_at(x, tag1, tag2, i)
        return M


    def log_sum_exp(self, arr):
        max_value = np.max(arr) # For numerically stablity
        return max_value + np.log(np.sum(np.exp(arr - max_value)))


    def log_alpha(self, x, M=None):
        """Get forward probablity log a(i, x).
        
        a(i, x, Yt) = sum_{y}a(i-1, x, y)*M(i-1, x, y, Yt)
        so log(a(i, tag)) = log sum exp(log a(i-1) + log M(, :, tag))

        Args:
            x: sequence
            M: log potential matrix M(x)

        Return:
            alpha: tensor(nwords_x+1, ntags)
        """
        nwords_x = len(x) - 2
        alpha = np.zeros((nwords_x + 1, self.ntags))

        if M is None:
            M = self.log_M(x)

        alpha[1] = M[1, self.start_index, :]
        for i in range(2, nwords_x + 1):
            for tag in range(self.ntags):
                alpha[i, tag] = self.log_sum_exp(alpha[i-1] + M[i, :, tag])
        return alpha


    def log_beta(self, x, M=None):
        """Get backward probablity log b(i, x)

        b(i, x, Yt) = sum_{y}M(i, x, Yt, y)b(i+1, x, y)
        
        Args:
            x: sequence
            M: log potential matrix M(x)

        Return:
            beta: tensor(nwords_x+1, ntags)
        """
        nwords_x = len(x) - 2
        beta = np.zeros((nwords_x + 1, self.ntags))

        if M is None:
            M = self.log_M(x)

        beta[nwords_x] = M[nwords_x+1, :, self.end_index]
        for i in range(nwords_x - 1, 0, -1):
            for tag in range(self.ntags):
                beta[i, tag] = self.log_sum_exp(beta[i+1] + M[i+1, tag, :])
        return beta


    def log_z(self, x, M=None, beta=None):
        """Get log Z(x)
        """
        nwords_x = len(x) - 2

        if M is None:
            M = self.log_M(x)
        if beta is None:
            beta = self.log_beta(x, M)

        z = self.log_sum_exp(beta[1] + M[1, self.start_index, :])
        return z


    def log_potential(self, x, y, M=None, beta=None):
        """Calculate log p(y|x).

        log p(y|x) = log exp(sum(W.Feature)) - log Z(x)
        """
        nwords_x = len(y) - 2   # every sentence include <START> and <END>

        if M is None:
            M = self.log_M(x)
        if beta is None:
            beta = self.log_beta(x, M)

        log_p = 0
        for i in range(1, nwords_x + 1):
            log_p += self.log_M_at(x, y[i-1], y[i], i)
        z = self.log_z(x, M, beta)
        log_p -= z
        return log_p


    def inference_viterbi(self, x, M=None):
        """Inference tags of x

        Return:
            y_char: ['B', 'S', ..., ] in char not in int
        """
        nwords_x = len(x) - 2
        delta = np.zeros((nwords_x + 1, self.ntags))
        trace = np.zeros((nwords_x + 1, self.ntags), dtype='int')

        if M is None:
            M = self.log_M(x)
        
        delta[1] = M[1, self.start_index, :]
        for i in range(2, nwords_x + 1):
            for tag in range(self.ntags):
                delta[i, tag] = np.max(delta[i-1] + M[i, :, tag])
                trace[i, tag] = np.argmax(delta[i-1] + M[i, :, tag])

        y_char = nwords_x * [self.start_tag]
        best = np.argmax(delta[nwords_x])
        y_char[nwords_x - 1] = self.index_tag[best]

        index = nwords_x - 1
        while index > 0:
            best = trace[index + 1][best]
            y_char[index - 1] = self.index_tag[best]
            index -= 1
        return y_char


    def model_gradient_x(self, x, M=None, alpha=None, beta=None):
        """Get sum_y p(y|x)C_k(y, x).
        
        log P(yi_1, yi|x) = log alpha(i-1, yi_1) + log M(i, yi_1, yi, x) + log beta(i, yi) - log z(x)
        One item in gradient, get more information from
        https://victorjiangxin.github.io/Chinese-Word-Segmentation/
        """
        nwords_x = len(x) - 2

        if M is None:
            M = self.log_M(x)
        if alpha is None:
            alpha = self.log_alpha(x, M)
        if beta is None:
            beta = self.log_beta(x, M)

        z = self.log_z(x, M, beta)
        P = np.zeros((nwords_x + 1, self.ntags, self.ntags))
        gradient = np.zeros(self.nweights)

        for i in range(1, nwords_x + 1):
            for yi_1 in range(self.ntags):
                for yi in range(self.ntags):
                    if i == 1 and yi_1 != self.start_index:
                        continue
                    P[i, yi_1, yi] = alpha[i-1, yi_1] + M[i, yi_1, yi] + beta[i, yi] - z

        # gradient = p(x, yi_1, yi) * C_k(x, yi_1, yi), not log p(x, yi_1, yi)!!
        P = np.exp(P)
        activate_feature = []
        for i in range(1, nwords_x + 1):
            for yi_1 in range(self.ntags):
                for yi in range(self.ntags):
                    # U feature
                    for pos in self.U_feature_pos:
                        if pos + i >= 0 and pos + i < len(x):
                            feature = ('U', pos, x[pos + i], yi)
                            if feature in self.feature_index:
                                activate_feature.append(self.feature_index[feature])
                    # B feature
                    for pos in self.B_feature_pos:
                        if pos + i >= 1 and pos + i < len(x):
                            feature = ('B', pos, x[pos + i], yi_1, yi)
                            if feature in self.feature_index:
                                activate_feature.append(self.feature_index[feature])
                    gradient[activate_feature] += P[i, yi_1, yi]

        return gradient


    def neg_likelihood_and_gradient(self, weights, prior_feature_count, train_data):
        """Return -L(x), f'(L)
        """
        self.weights = weights
        likelihood = 0
        gradient = np.zeros(self.nweights)
        for x, y in train_data:
            M = self.log_M(x)
            alpha = self.log_alpha(x, M)
            beta = self.log_beta(x, M)
            likelihood += self.log_potential(x, y, M, beta)
            gradient -= self.model_gradient_x(x, M, alpha, beta)
        # add regulariser
        likelihood = likelihood - np.dot(self.weights, self.weights) * self.theta / 2
        gradient = prior_feature_count - gradient - self.weights * self.theta

        return -likelihood, -gradient


    def train(self, file_name):
        """Train this model

        Args:
            file_name: corpus file
        """
        sentences = []
        labels = []

        f = codecs.open(file_name, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()

        sentence = [self.start_char]
        label = [self.start_tag]
        for line in lines:
            if len(line) < 2:
                # sentence end
                sentence.append(self.end_char)
                label.append(self.end_tag)
                sentences.append(sentence)
                labels.append(label)
                sentence = [self.start_char]
                label = [self.start_tag]
            else:
                char, tag = line.split()
                sentence.append(char)
                label.append(tag)
                if char not in self.word_index:
                    self.word_index[char] = self.nwords
                    self.index_word[self.nwords] = char
                    self.nwords += 1

        print("Total words in corpus is {}".format(self.nwords))
        print("sentence[0]:{} labels[0]:{}".format(''.join(sentences[0]), ''.join(labels[0])))

        feature_id = 0
        for pos in self.U_feature_pos:
            for word in range(self.nwords):
                for tag in range(self.ntags):
                    feature = ('U', pos, word, tag)
                    self.feature_index[feature] = feature_id
                    self.index_feature[feature_id] = feature
                    feature_id += 1

        for pos in self.B_feature_pos:
            for word in range(self.nwords):
                for yi_1 in range(self.ntags):
                    for yi in range(self.ntags):
                        feature = ('B', pos, word, yi_1, yi)
                        self.feature_index[feature] = feature_id
                        self.index_feature[feature_id] = feature
                        feature_id += 1

        self.nfeatures = feature_id
        self.nweights = self.nfeatures
        prior_feature_count = np.zeros(self.nfeatures)
        self.weights = np.random.randn(self.nweights)

        print("Feature nums: {}".format(self.nfeatures))

        sentences = [[self.word_index[char] for char in s] for s in sentences]
        labels = [[self.tag_index[tag] for tag in label] for label in labels]

        print("sentence[0]:\n{}\n labels[0]:\n{}\n".format(sentences[0], labels[0]))

        train_data = [(x, y) for (x, y) in zip(sentences, labels)]

        del sentences
        del labels
        # get C(y, x)
        for x, y in train_data:
            n = len(x) - 2
            for i in range(1, n + 1):
                activate_feature = []
                for pos in self.U_feature_pos:
                    if pos + i >= 0 and pos + i < len(x):
                        feature = ('U', pos, x[pos + i], y[i])
                        if feature in self.feature_index:
                            activate_feature.append(self.feature_index[feature])

                for pos in self.B_feature_pos:
                    if pos + i >= 1 and pos + i < len(x):
                        feature = ('B', pos, x[pos + i], y[i-1], y[i])
                        if feature in self.feature_index:
                            activate_feature.append(self.feature_index[feature])

                prior_feature_count[activate_feature] += 1

        print("prior_feature_count[0]: {} {}".format(self.index_feature[0], prior_feature_count[0]))

        print("Start training!")
        func = lambda weights : self.neg_likelihood_and_gradient(weights, prior_feature_count, train_data)
        start_time = time.time()
        res = optimize.fmin_l_bfgs_b(func, self.weights, iprint=0, disp=1, maxiter=300)
        print("Training time:{}s".format(time.time() - start_time))

        self.save()
        

    def save(self, file_path='linear_crf.model'):
        save_dict = {}
        save_dict['nwords'] = self.nwords
        save_dict['nfeatures'] = self.nfeatures
        save_dict['feature_index'] = self.feature_index
        save_dict['index_feature'] = self.index_feature
        save_dict['index_word'] = self.index_word
        save_dict['word_index'] = self.word_index
        save_dict['nweights'] = self.nweights
        save_dict['index_word'] = self.index_word
        save_dict['weights'] = self.weights
        with open(file_path, 'wb') as f:
            pickle.dump(save_dict, f)
        print("Save model successful!")


    def load(self, file_path):
        with open(file_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.nwords = save_dict['nwords']
        self.nfeatures = save_dict['nfeatures']
        self.feature_index = save_dict['feature_index']
        self.index_feature = save_dict['index_feature'] 
        self.index_word = save_dict['index_word']
        self.word_index = save_dict['word_index']
        self.nweights = save_dict['nweights']
        self.index_word = save_dict['index_word']
        self.weights = save_dict['weights']

        print("Load model successful!")

