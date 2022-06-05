from numpy import product
import os
from dataset import Dataset
from utils import *
from collections import Counter
import dill
import pandas as pd

class Filter():
    def __init__(self,kwargs = {}):

        self.alpha = kwargs.get('alpha',1)
        self.min_count = kwargs.get('min_count', 10)
        self.threshold = kwargs.get('threshold', 0.9)
        self.model_size = int(kwargs.get('model_size', 500))


    def train(self, target_words,general_words):
        self.target_words =flatten(map(word_token,target_words))
        self.general_words =flatten(map(word_token, general_words))
        self.n_sentence = len(target_words)+len(general_words)


        target_count = dict(Counter(self.target_words))
        general_count = dict(Counter(self.general_words))

        total_count = {k: general_count.get(k, 0) + target_count.get(k, 0) for k in set(general_count)}

        # Can only concider words that are in the general list to avoid division by zero errors 
        word_dict = set([k for k,v in total_count.items() if v > self.min_count])
        self.normed_target = {k: target_count[k] if k in target_count.keys() else 0 for k in word_dict}
        self.normed_general = {k: general_count[k] if k in general_count.keys() else 0 for k in word_dict}

        self.ratio = {k: (self.normed_target[k] +self.alpha)*sum(self.normed_general.values())/(self.normed_general[k] + 2*self.alpha)/sum(self.normed_target.values())  for k in word_dict}
        self.norm = pd.Series({k: v/(v+1)  for k,v in self.ratio.items()})



        """
        Find which words in dictionary provide the most information about any given sentence.

        info from pr -> greatest deviation from 0.5
        
        combining this with the probabilty that a word apears in a sentence
        
        
        minimising expected loss:
            if a word is removed from the model, it maps to a probabilty of 0.5, which correspond to a information loss of abs(pr-0.5)
            for a word with x probabilty of being in a sentence, expected loss = abs(pr-0.5)*x
        """
        if len(self.norm) > 0:
            self.stat = pd.merge(self.norm.rename('pr'), pd.Series(total_count).rename('count'), right_index=True, left_index=True)
            self.stat['pr'] = abs(self.stat['pr']-0.5)
            self.stat['count'] = self.stat['count']/self.n_sentence
            self.stat['score'] = self.stat.apply(lambda x: x['pr']*x['count'], axis = 1)
            indecies = self.stat.sort_values('score',ascending = False).iloc[0:self.model_size,].index

            self.model = self.norm.loc[indecies]
        else:
            self.model = self.norm


    def predict(self, df, return_prob = False, quiet= False):
        df_store = df.copy()
        if len(self.model) >0:
            df['word'] = df['sentence'].apply(lambda x: word_token(x))
            df = df.explode('word')

            words = set(self.model.index).intersection(set(df['word']))


            df[['p', '!p']] = None,None
            if quiet:
                for word in words:
                    df.loc[df['word'] == word,'p'] = self.model[word]
                    df.loc[df['word'] == word,'!p'] = 1-self.model[word]
            else:
                for word in tqdm(words, total=len(words)):
                    df.loc[df['word'] == word,'p'] = self.model[word]
                    df.loc[df['word'] == word,'!p'] = 1-self.model[word]
            
            
            df_pr = df.groupby(by=lambda x: x)[['p','!p']].prod()



            df_pr['prob'] = df_pr['p']/ (df_pr['p'] + df_pr['!p'])
            df_pr['climate'] = df_pr['prob'].apply(lambda x: x >= self.threshold)
            df_store =  df_store.join(df_pr, how = 'left',lsuffix='old')
        else:
            df_store['prob'] = 0.5
            df_store['climate'] = df_store['prob'].apply(lambda x: x >= self.threshold)
        df_store.loc[df_store['climate'].isna(),'climate'] = self.threshold <= 0.5
        if return_prob:
            return df_store['prob']
        else:
            return df_store['climate']

    def predict_single(self, sentence):
        words = [x for x in word_token(sentence) if x in self.model.index]
        if len(words):
            probs = self.model[words]
            pr =  product(probs)/(product(probs) + product([1-x for x in probs]))
        else:
            pr =  0.5
        threshold_bool = pr > self.threshold
        return threshold_bool


