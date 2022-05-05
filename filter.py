from numpy import product
import os
from dataset import Dataset
from utils import *
from collections import Counter
import dill

class Filter():
    def __init__(self,target_words,general_words, min_count = 100):

        self.target_words = target_words
        self.general_words = general_words
        self.min_count = min_count
        if os.path.exists(os.path.join('picklejar','filter.pickle')):
            self.load()
        else:
            self.build_model()
            self.save()


    def build_model(self):
        target_count = dict(Counter(self.target_words))
        general_count = dict(Counter(self.general_words))


        self.normed_target = {k: v for k, v in target_count.items() if v > self.min_count}
        self.normed_general = {k: v for k, v in general_count.items() if v > self.min_count}

        self.word_dict = set(self.normed_target.keys()).intersection(set(self.normed_general.keys()))
        self.ratio = {k: self.normed_target[k]*sum(self.normed_general.values())/self.normed_general[k]/sum(self.normed_target.values())  for k in self.word_dict }
        self.norm = {k: v/(v+1)  for k,v in self.ratio.items()}

    def prob(self, sentence):
        words = [x for x in word_token(sentence) if x in self.word_dict]
        probs = {x: self.norm[x] for x in words}
        return product(list(probs.values()))/(product(list(probs.values())) + product([1-x for x in probs.values()]))


    def save(self):
        print('Pickling')
        with open(os.path.join('picklejar','filter.pickle'), 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling')
        with open(os.path.join('picklejar','filter.pickle'), 'rb') as f:
            tmp_dic = dill.load(f)

            if tmp_dic['target_words']==self.target_words and tmp_dic['general_words'] == self.general_words and tmp_dic['min_count']==self.min_count:
                self.__dict__.clear()
                self.__dict__.update(tmp_dic)
            else:
                self.build_model()
                self.save()

        

if __name__ == "__main__":
    d = Dataset()
    climate_words = flatten(flatten([simple_map(word_token,d.df_climate['article']), simple_map(word_token, d.df_skeptics['article'])]))
    news_words = flatten(simple_map(word_token,d.df_news['article']))
    f = Filter(climate_words,news_words)


