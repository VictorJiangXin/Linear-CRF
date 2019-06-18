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
    ('B', tag_pre, tag_now)
    You can get more information from my blog (PS. the blog is in Chinese)
    https://victorjiangxin.github.io/Chinese-Word-Segmentation/
    """
    def __init__(self):
        super(LinearCRF, self).__init__()
        self.ntags = 4 # {'B', 'I', 'E', 'S'}
        self.index_tag = {} # {0:'B', 1:'I', 2:'E', 3:'S'}
        self.tag_index = {} # {'B':0, 'I':1, 'E':2, 'S':3}

        self.start_tag = 'S'
        self.end_tag = 'S'

        self.U_feature_pos = [-2, -1, 0, 1, 2]

        self.index_feature = {} # {0 : ('U', -2, word, tag)}
        self.feature_index = {} # {('U', -2, word, tag)}

        self.nweights = 0
        self.weights = np.zeros(self.nweights)
        self.theta = 1e-4 # theta should in the range of (1e-6 ~ 1e-3)


    def get_word(self, x, i):
        """Return x[i]
        """
        if i == -1:
            return '_B-1'
        elif i == -2:
            return '_B-2'
        elif i == len(x):
            return '_B+1'
        elif i == len(x) + 1:
            return '_B+2'
        else:
            return x[i]


    def feature_at(self, k, x, y_pre, y_now, i):
        """Get f_k(yt_1, yt, x, t).

        Args:
            k: (int) the Kth feature
            x: str word list [word_index['<START'>], word_index['今'],]
            yi_1: tag of y_[i-1]
            yi: tag of yi
            i: (int) index

        Return:
            1 or 0
        """
        if k < self.nweights:
            feature = self.index_feature[k]
            if feature[0] == 'U':
                _, pos, word, tag = feature
                if y_now == tag and self.get_word(x, i + pos) == word:
                    return 1
            elif feature[0] == 'B':
                _, tag_pre, tag_now = feature
                if tag_pre == y_pre and tag_now == y_now:
                    return 1

        return 0


    def log_M_at(self, x, y_pre, y_now, i):
        """Calc log M(yi_1, yi|x) = W.F_i(yi_1, yi|x)
        """
        nwords = len(x)
        activate_feature = []
        if i == 0 and y_pre != self.tag_index[self.start_tag]:
            return float('-inf')
        elif i == nwords and y_now != self.tag_index[self.end_tag]:
            return float('-inf')
        elif i == nwords and y_now == self.tag_index[self.end_tag]:
            return 0

        # U feature
        for pos in self.U_feature_pos:
            feature = ('U', pos, self.get_word(x, i + pos), y_now)
            if feature in self.feature_index:
                activate_feature.append(self.feature_index[feature])

        # B feature
        feature = ('B', y_pre, y_now)
        if feature in self.feature_index:
            activate_feature.append(self.feature_index[feature])

        return self.weights[activate_feature].sum()


    def log_M(self, x):
        """Get log probablity matrix M(x)

        Return:
            M(x): tensor(nwords_x+1, ntags, ntags)
        """
        nwords = len(x)
        M = np.ones((nwords + 1, self.ntags, self.ntags)) * float('-inf')
        for i in range(nwords + 1):
            for tag_pre in range(self.ntags):
                for tag_now in range(self.ntags):
                    M[i, tag_pre, tag_now] = self.log_M_at(x, tag_pre, tag_now, i)
        return M


    def log_sum_exp(self, a, b):
        """
        a = [a1, a2, a3]
        b = [b1, b2, b3]
        return log(e^a1*e^b1+e^a2*e^b2+e^a3*e^b3)
        """
        return np.log(np.sum(np.exp(a) * np.exp(b)))


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
        nwords = len(x)
        alpha = np.ones((nwords + 1, self.ntags)) * float('-inf')

        if M is None:
            M = self.log_M(x)

        alpha[0] = M[0, self.tag_index[self.start_tag], :]
        for i in range(1, nwords + 1):
            for tag in range(self.ntags):
                alpha[i, tag] = self.log_sum_exp(alpha[i - 1], 
                                                M[i, :, tag])
        return alpha


    def log_beta(self, x, M=None):
        """Get backward probablity log b(i, x)

        b(i, x, Yt) = sum_{y}M(i, x, Yt, y)b(i+1, x, y)
        Warnning: because beta[len(x)] = [0, 0, 0, 1] is certain
        so we use beta[len(x)] to store beta[-1]
        
        Args:
            x: sequence
            M: log potential matrix M(x)

        Return:
            beta: tensor(nwords_x+1, ntags)
        """
        nwords = len(x)
        beta = np.ones((nwords + 1, self.ntags)) * float('-inf')

        if M is None:
            M = self.log_M(x)

        beta[nwords - 1] = 0 # because the last one must be 'S'
        for i in range(nwords - 2, -2, -1):
            for tag in range(self.ntags):
                beta[i, tag] = self.log_sum_exp(beta[i + 1], 
                                                M[i + 1, tag, :])
        return beta


    def log_z(self, x, M=None, alpha=None):
        """Get log Z(x)
        """
        nwords = len(x)

        if M is None:
            M = self.log_M(x)

        if alpha is None:
            alpha = self.log_alpha(x, M)

        return alpha[nwords, self.tag_index[self.end_tag]]


    def log_potential(self, x, y, M=None, alpha=None):
        """Calculate log p(y|x).

        log p(y|x) = log exp(sum(W.Feature)) - log Z(x)
        """
        nwords = len(x)

        if M is None:
            M = self.log_M(x)
        if alpha is None:
            alpha = self.log_alpha(x, M)

        log_p = 0
        for i in range(nwords):
            if i == 0:
                log_p += self.log_M_at(x, self.tag_index[self.start_tag], 
                                        y[i], i)
            else:
                log_p += self.log_M_at(x, y[i - 1], y[i], i)

        z = self.log_z(x, M, alpha)
        log_p -= z

        return log_p


    def inference_viterbi(self, x, M=None):
        """Inference tags of x

        Return:
            y_char: ['B', 'S', ..., ] in char not in int
        """
        nwords = len(x)
        delta = np.zeros((nwords, self.ntags))
        trace = np.zeros((nwords, self.ntags), dtype='int')

        if M is None:
            M = self.log_M(x)

        delta[0] = M[0, self.tag_index[self.start_tag], :]
        for i in range(1, nwords):
            for tag in range(self.ntags):
                delta[i, tag] = np.max(delta[i - 1] + M[i, :, tag])
                trace[i, tag] = np.argmax(delta[i-1] + M[i, :, tag])

        y_char = nwords * [self.start_tag]
        best = np.argmax(delta[nwords - 1])
        y_char[nwords - 1] = self.index_tag[best]

        for i in range(nwords - 2, -1, -1):
            best = trace[i + 1, best]
            y_char[i] = self.index_tag[best]

        return y_char


    def model_gradient_x(self, x, M=None, alpha=None, beta=None):
        """Get sum_y p(y|x)C_k(y, x).
        
        log P(yi_1, yi|x) = log alpha(i-1, yi_1) + log M(i, yi_1, yi, x) + log beta(i, yi) - log z(x)
        One item in gradient, get more information from
        https://victorjiangxin.github.io/Chinese-Word-Segmentation/
        """
        nwords = len(x)

        if M is None:
            M = self.log_M(x)
        if alpha is None:
            alpha = self.log_alpha(x, M)
        if beta is None:
            beta = self.log_beta(x, M)

        z = self.log_z(x, M, alpha)
        P = np.zeros((nwords, self.ntags, self.ntags))
        gradient = np.zeros(self.nweights)

        for i in range(nwords):
            for y_pre in range(self.ntags):
                for y_now in range(self.ntags):
                    if i == 0 and y_pre != self.tag_index[self.start_tag]:
                        pass
                    elif i == 0 and y_pre == self.tag_index[self.start_tag]:
                        P[i, y_pre, y_now] = M[i, y_pre, y_now] + beta[i, y_now] - z
                    else:
                        P[i, y_pre, y_now] = alpha[i - 1, y_pre] +\
                                            M[i, y_pre, y_now] + beta[i, y_now] - z

        P = np.exp(P)
        for i in range(nwords):
            for y_pre in range(self.ntags):
                for y_now in range(self.ntags):
                    activate_feature = []
                    # U feature
                    for pos in self.U_feature_pos:
                        feature = ('U', pos, self.get_word(x, i + pos), y_now)
                        if feature in self.feature_index:
                            activate_feature.append(self.feature_index[feature])

                    # B feature
                    if i == 0 and y_pre != self.tag_index[self.start_tag]:
                        pass
                    else:
                        feature = ('B', y_pre, y_now)
                        if feature in self.feature_index:
                            activate_feature.append(self.feature_index[feature])
                    gradient[activate_feature] += P[i, y_pre, y_now]

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
            liktlihood += self.log_potential(x, y, M, alpha)
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
        print('Start training!')
        self.ntags = 4
        self.index_tag = {0:'B', 1:'I', 2:'E', 3:'S'}
        self.tag_index = {'B':0, 'I':1, 'E':2, 'S':3}

        sentences = []
        labels = []

        f = codecs.open(file_name, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()

        words = []
        sentence = []
        label = []
        for line in lines:
            if len(line) < 2:
                # sentence end
                if len(sentence) > 3:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
            else:
                char, tag = line.split()
                sentence.append(char)
                label.append(tag)
                if char not in words:
                    words.append(char)

        print("Total sentences is {}".format(len(sentences)))
        print("Total words in corpus is {}".format(len(words)))
        print("sentence[0]:{} labels[0]:{}".format(''.join(sentences[0]), ''.join(labels[0])))

        labels = [[self.tag_index[tag] for tag in label] for label in labels]
        train_data = zip(sentences, labels)

        del sentences
        del labels

        # construct features
        # B features
        feature_id = 0
        for tag_pre in range(self.ntags):
            for tag_now in range(self.ntags):
                feature = ('B', tag_pre, tag_now)
                self.feature_index[feature] = feature_id
                self.index_feature[feature_id] = feature
                feature_id += 1

        # U features
        for x, _ in train_data:
            nwords = len(x)
            for i in range(nwords):
                for pos in self.U_feature_pos:
                    for tag in range(self.ntags):
                        feature = ('U', pos, self.get_word(x, i + pos), tag)
                        if feature not in self.feature_index:
                            self.feature_index[feature] = feature_id
                            self.index_feature[feature_id] = feature
                            feature_id += 1

        self.nweights = len(self.feature_index)
        self.weights = np.random.randn(self.nweights)
        print('Total features is {}'.format(self.nweights))
        print('Feature[0]={}, Feature[16]={}', self.index_feature[0],
                                             self.index_feature[16])

        print('Statistic Count of feature k ....')
        prior_feature_count = np.zeros(self.nweights)
        for x, y in train_data:
            nwords = len(x)
            for i in range(nwords):
                activate_feature = []
                # U feature
                for pos in self.U_feature_pos:
                    feature = ('U', pos, self.get_word(x, i + pos), y[i])
                    activate_feature.append(feature)
                # B feature
                if i == 0:
                    feature = ('B', self.tag_index[self.start_tag], y[i])
                else:
                    feature = ('B', y[i - 1], y[i])
                activate_feature.append(feature)
                prior_feature_count[activate_feature] += 1

        print("prior_feature_count[0]: {} {}".format(self.index_feature[0], prior_feature_count[0]))

        print("Start training!")
        func = lambda weights : self.neg_likelihood_and_gradient(weights, prior_feature_count, train_data)
        start_time = time.time()
        res = optimize.fmin_l_bfgs_b(func, self.weights, iprint=0, disp=1, maxiter=300, maxls=100)

        print("Training time:{}s".format(time.time() - start_time))

        self.save()


    def save(self, file_path='linear_crf.model'):
        save_dict = {}
        save_dict['ntags'] = self.ntags
        save_dict['index_tag'] = self.index_tag
        save_dict['tag_index'] = self.tag_index
        save_dict['feature_index'] = self.feature_index
        save_dict['index_feature'] = self.index_feature
        save_dict['nweights'] = self.nweights
        save_dict['weights'] = self.weights
        with open(file_path, 'wb') as f:
            pickle.dump(save_dict, f)
        print("Save model successful!")


    def load(self, file_path):
        with open(file_path, 'rb') as f:
            save_dict = pickle.load(f)

        self.ntags = save_dict['ntags']
        self.index_tag = save_dict['index_tag']
        self.tag_index = save_dict['tag_index']
        self.feature_index = save_dict['feature_index']
        self.index_feature = save_dict['index_feature'] 
        self.nweights = save_dict['nweights']
        self.weights = save_dict['weights']

        print("Load model successful!")


    def load_crfpp_model(self, model_path):
        """Load model which is trained by crf++
        """
        with open(model_path, 'r') as f:
            lines = f.readlines()

        tags_id = 0

        i = 0
        # print plus information
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            print(line)
            i += 1

        i += 1
        # get tags 
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            self.tag_index[line] = tags_id
            self.index_tag[tags_id] = line
            tags_id += 1
            i += 1

        self.ntags = len(self.tag_index)
        print(self.tag_index)

        i += 1
        # map
        feature_map = {} # {'U00', -2}
        self.U_feature_pos = []
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            if line != 'B':
                feature_template = line.split(':')[0]
                pos = line.split('[')[1].split(',')[0]
                feature_map[feature_template] = int(pos)
                self.U_feature_pos.append(int(pos))
            i += 1
        print('self.U_feature_pos', self.U_feature_pos)
        print('feature_map:', feature_map)

        i += 1
        # construct feature
        feature_id = 0
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip().split()[1]
            if line == 'B':
                for tag_pre in range(self.ntags):
                    for tag_now in range(self.ntags):
                        feature = ('B', tag_pre, tag_now)
                        self.feature_index[feature] = feature_id
                        self.index_feature[feature_id] = feature
                        feature_id += 1
            else:
                feature_template = line.split(':')[0]
                word = line.split(':')[1]
                pos = feature_map[feature_template]
                for tag in range(self.ntags):
                    feature = ('U', pos, word, tag)
                    self.feature_index[feature] = feature_id
                    self.index_feature[feature_id] = feature
                    feature_id += 1
            i += 1

        print('Total features:', len(self.feature_index))
        i += 1
        # read weights
        self.nweights = len(self.feature_index)
        self.weights = np.zeros(self.nweights)
        feature_id = 0
        while i < len(lines) and lines[i] != '\n':
            line = lines[i].strip()
            self.weights[feature_id] = float(line)
            feature_id += 1
            i += 1

        print('Record weights = ', feature_id)
        print("The last feature is {}, it's weight is {}".format(
                    self.index_feature[feature_id-1], self.weights[feature_id-1]))
        self.save()










