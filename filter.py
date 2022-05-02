from numpy import product
from dataset import Dataset
from utils import *
from collections import Counter

class Filter():
    def __init__(self,target_words,general_words, min_count = 100):

        target_count = dict(Counter(target_words))
        general_count = dict(Counter(general_words))


        self.normed_target = {k: v for k, v in target_count.items() if v > min_count}
        self.normed_general = {k: v for k, v in general_count.items() if v > min_count}

        self.word_dict = set(self.normed_target.keys()).intersection(set(self.normed_general.keys()))
        self.ratio = {k: self.normed_target[k]*sum(self.normed_general.values())/self.normed_general[k]/sum(self.normed_target.values())  for k in self.word_dict }
        self.norm = {k: v/(v+1)  for k,v in self.ratio.items()}

    def prob(self, sentence):
        words = [x for x in word_token(sentence) if x in self.word_dict]
        probs = {x: self.norm[x] for x in words}
        print(probs)
        return product(list(probs.values()))/(product(list(probs.values())) + product([1-x for x in probs.values()]))



        

if __name__ == "__main__":
    d = Dataset()
    climate_words = flatten(flatten([simple_map(word_token,d.df_climate['article']), simple_map(word_token, d.df_skeptics['article'])]))
    news_words = flatten(simple_map(word_token,d.df_news['article']))
    f = Filter(climate_words,news_words)


