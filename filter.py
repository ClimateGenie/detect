from numpy import product
import os
from dataset import Dataset
from utils import *
from collections import Counter
import dill
import pandas as pd

class Filter():
    def __init__(self,target_words= None,general_words = None,load = False, min_count = 10, alpha = 1):

        self.target_words = target_words
        self.alpha = alpha
        self.general_words = general_words
        self.min_count = min_count
        self.load_from_file = load
        if os.path.exists(os.path.join('picklejar','filter.pickle')):
            self.load()
        else:
            self.build_model()
            self.save()


    def build_model(self):
        target_count = dict(Counter(self.target_words))
        general_count = dict(Counter(self.general_words))

        total_count = {k: general_count.get(k, 0) + target_count.get(k, 0) for k in set(target_count) | set(general_count)}

        word_dict = set([k for k,v in total_count.items() if v > self.min_count])
        self.normed_target = {k: target_count[k] if k in target_count.keys() else 0 for k in word_dict}
        self.normed_general = {k: general_count[k] if k in general_count.keys() else 0 for k in word_dict}

        self.ratio = {k: (self.normed_target[k] +self.alpha)*sum(self.normed_general.values())/(self.normed_general[k] + 2*self.alpha)/sum(self.normed_target.values())  for k in word_dict }
        self.norm = pd.Series({k: v/(v+1)  for k,v in self.ratio.items()})

    def prob(self, sentence):
        words = [x for x in word_token(sentence) if x in self.norm.index]
        if len(words):
            probs = self.norm[words]
            return product(probs)/(product(probs) + product([1-x for x in probs]))
        else:
            return 0.5


    def save(self):
        print('Pickling Filter')
        with open(os.path.join('picklejar','filter.pickle'), 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling Filter')
        with open(os.path.join('picklejar','filter.pickle'), 'rb') as f:
            tmp_dic = dill.load(f)

            if (tmp_dic['target_words']==self.target_words and tmp_dic['general_words'] == self.general_words and tmp_dic['min_count']==self.min_count) or self.load_from_file:
                self.__dict__.clear()
                self.__dict__.update(tmp_dic)
            else:
                self.build_model()
                self.save()

        


