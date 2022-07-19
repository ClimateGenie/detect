from numpy import product
from utils import *
from collections import Counter
import pandas as pd

class Filter():
    def __init__(self,kwargs = {}):

        self.min_count = kwargs.get('min_count', 1000)
        self.threshold = kwargs.get('threshold', 0.9)
        self.model_size = int(kwargs.get('model_size', 500))


    def train(self, target_words,general_words):
        self.target_words = word_token(' '.join(target_words))
        self.general_words = word_token(' '.join(general_words))
        self.n_sentence = len(target_words)+len(general_words)


        target_count = dict(Counter(self.target_words))
        general_count = dict(Counter(self.general_words))

        self.total_count = {k: general_count.get(k, 0) + target_count.get(k, 0) for k in set(general_count).union(set(target_count))}

        # If words only show up in target count, given ratio based on min count -> grateer min count gives more confidence
        self.normed_target = {k: target_count[k] if k in target_count.keys() else 0 for k in self.total_count.keys()}
        self.normed_general = {k: general_count[k] if k in general_count.keys() else 0 for k in self.total_count.keys()}

        words = set(clean_words([k for k,v in self.total_count.items() if v >= self.min_count]))
        self.ratio = {k: (self.normed_target[k])*sum(self.normed_general.values())/(self.normed_general[k])/sum(self.normed_target.values())  for k in words.intersection(set(general_count))}
        # The assume the next sample for words not in general corpus is in the general cormus 
        for word in words - (set(general_count)):
            self.ratio[word] = target_count[word]
        self.norm = pd.Series({k: v/(v+1)  for k,v in self.ratio.items()})


        """
        Find which words in dictionary provide the most information about any given sentence.

        info from pr -> greatest deviation from 0.5
        
        combining this with the probabilty that a word apears in a sentence
        
        
        minimising expected loss:
            if a word is removed from the model, it maps to a probabilty of 0.5, which correspond to a information loss of abs(pr-0.5)
            for a word with x probabilty of being in a sentence, expected loss = abs(pr-0.5)*x
        """

        score = pd.Series([abs(0.5-x) * general_count.get(i,0) for i,x in self.norm.iteritems()], index=self.norm.index)
        self.norm =  self.norm.loc[score.index]
        self.model = self.norm.iloc[0:self.model_size]


    def predict(self, df, return_prob = False, quiet= False):
        df_store = df.copy()
        if len(self.model) >0:
            df['word'] = mult_word_token(df['sentence'].values)
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
    
    


