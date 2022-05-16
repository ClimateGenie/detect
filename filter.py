from numpy import product
import os
from dataset import Dataset
from utils import *
from collections import Counter
import dill
import pandas as pd

class Filter():
    def __init__(self,target_words,general_words,**kwargs):

        self.alpha = kwargs.get('alpha',1)
        self.min_count = kwargs.get('min_count', 10)
        self.threshold = kwargs.get('threshold', 0.9)
        self.word_count = sum([len(target_words), len(general_words)])

        self.pickle_string = '-'.join([f'{key}:{str(x)}' for key, x in self.__dict__.items()])

        self.target_words = target_words
        self.general_words = general_words


    def train(self):
        target_count = dict(Counter(self.target_words))
        general_count = dict(Counter(self.general_words))

        total_count = {k: general_count.get(k, 0) + target_count.get(k, 0) for k in set(target_count) | set(general_count)}

        word_dict = set([k for k,v in total_count.items() if v > self.min_count])
        self.normed_target = {k: target_count[k] if k in target_count.keys() else 0 for k in word_dict}
        self.normed_general = {k: general_count[k] if k in general_count.keys() else 0 for k in word_dict}

        self.ratio = {k: (self.normed_target[k] +self.alpha)*sum(self.normed_general.values())/(self.normed_general[k] + 2*self.alpha)/sum(self.normed_target.values())  for k in word_dict }
        self.norm = pd.Series({k: v/(v+1)  for k,v in self.ratio.items()})

    def predict(self, sentence):
        words = [x for x in word_token(sentence) if x in self.norm.index]
        if len(words):
            probs = self.norm[words]
            pr =  product(probs)/(product(probs) + product([1-x for x in probs]))
        else:
            pr =  0.5
        threshold_bool = pr > self.threshold
        return threshold_bool


    def save(self):
        print('Pickling Filter')
        with open(os.path.join('picklejar','filter',self.pickle_string), 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling Filter')
        with open(os.path.join('picklejar','filter',self.pickle_string), 'rb') as f:
            tmp_dic = dill.load(f)
            self.__dict__.update(tmp_dic)

        


