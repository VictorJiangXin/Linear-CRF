# Linear-CRF
An simple implemention of Linear CRF to Chinese Word Segmentaion.
# Introdution
We implemented a simple linear chain CRF(conditional random field), which can be used for Chinese word segmentation tasks. Also you can used it for other tagging tasks, such as POS(part of speech) Tagging, NER(name entity recognition) and so on. 
# Author
Jiang Xin(姜鑫) jiangxin18s@ict.ac.cn  
Li Bo(李博)  
Li Zhenlong(栗正隆)
# Performance
$$F1 = 2 * P * R / (P + R)$$

| Corpus | P | R | F1 |
| ------ | ---- | ---- | ---- |
| msr | 76.84263691915065 | 80.16949059509177 | 78.47081821412925 |
| pku | 89.23635340814322 | 88.86412962515287 | 89.04985254930146 |
| weibo | 71.06637402558087 | 72.00920245398773 | 71.53468175065706 |
# License
This project is released under the MIT.
# Preparation
First of all, clone the code
```
git clone https://github.com/VictorJiangXin/Linear-CRF.git
```
Then install all the python dependencies using pip:
```
pip install -r requirements.txt
```
# How to use
```
root@:path$ python demo.py
```
```
>>> from segmentation import *
>>> ucas_seg = Segmentation() # also you can load your model ucas_seg = Segmentation('your_model')
>>> sentence = '今晚的月色真美呀。'
>>> ucas_seg.seg(sentence)
```
# How to test
```
$ python test.py
```
You can change the test file by altering the path in test.py.
# How to eval
```
$ cd utils
$ python eval.py
```
# How to train
Firstly, you should change the corpus into this format.  
```
今   B
晚   E
月   B
色   E
真   B
美   E
。   S


```
You can use `src/utils/make_crf_trainset.py` to convert your corpus.

Then, you have to kinds of ways to train your model.  
You can use python to train your model. Python don't support multithreading, so this way will cost lots of time to train your model.   
```
cd 'src'
python train.py # you can change the file path in train.py to change the corpus
```
Also you can use crf++ to train your model.
```
$ crf_learn -c 2 template ../data/pku_training.data -t crfpp.pku.model
$ python
>>> from crf import *
>>> model = LinearCRF()
>>> model.load_crfpp_model('crfpp.pku.model')
```
# Others
Welcome to see my[blog](https://victorjiangxin.github.io/Chinese-Word-Segmentation/)

