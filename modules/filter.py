from numpy import product
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from modules.utils import *
from collections import Counter
import pandas as pd
import wandb

class Filter():
    def __init__(self,wandb_run,min_count = 1000, threshold = 0.9, model_size = 500, rank_score = 0.5):
        self.min_count = min_count
        self.threshold = threshold
        self.model_size =  model_size
        self.rank_score = rank_score
        if wandb_run:
            self.run = wandb_run
        else:
            self.run = wandb.init()

    def get_params(self, deep = True):
        return {
                'min_count': self.min_count,
                'threshold': self.threshold,
                'model_size': self.model_size,
                'rank_score': self.rank_score
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):

        self.X_ = X
        self.y_ = y

        self.classes_ = [False,True]

        target_words = X[np.where(y==1)].astype(str)
        general_words = X[np.where(y==-1)].astype(str)
        self.target_words = word_token(' '.join(flatten(target_words)))
        self.general_words = word_token(' '.join(flatten(general_words)))
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
        if self.rank_score != 1:
            alpha = 1/(3*(self.rank_score-1)**2) - (1/3)
            score = pd.Series([ (1+alpha**2)*abs(0.5-x) * general_count.get(i,0)/((alpha**2)*abs(0.5-x) + general_count.get(i,0)) for i,x in self.norm.iteritems()], index=self.norm.index)
        else:
            score = pd.Series([ abs(0.5-x) for i,x in self.norm.iteritems()], index=self.norm.index).sort_values(ascending=False)
    
        self.norm =  self.norm.loc[score.index]
        self.model = self.norm.iloc[0:self.model_size]

        return self


    def predict(self, X):
        check_is_fitted(self)
        df = pd.DataFrame(X,columns=['sentence'])
        df_store = df.copy()
        if len(self.model) >0:
            df['word'] = mult_word_token(df['sentence'].values)
            df = df.explode('word')

            words = set(self.model.index).intersection(set(df['word']))


            df[['p', '!p']] = None,None
            for word in words:
                df.loc[df['word'] == word,'p'] = self.model[word]
                df.loc[df['word'] == word,'!p'] = 1-self.model[word]
            
            
            df_pr = df.groupby(by=lambda x: x)[['p','!p']].prod()


            df_pr['prob'] = df_pr['p']/ (df_pr['p'] + df_pr['!p'])
            df_pr['climate'] = df_pr['prob'].apply(lambda x: x >= self.threshold)
            df_store =  df_store.join(df_pr, how = 'left',lsuffix='old')

        else:
            df_store['prob'] = 0.5

        df_store['climate'] = df_store['prob'].apply(lambda x: x >= self.threshold)

        return df_store['climate'].values


    def predict_proba(self,X):
        check_is_fitted(self)
        df = pd.DataFrame(X,columns=['sentence'])
        df_store = df.copy()
        if len(self.model) >0:
            df['word'] = mult_word_token(df['sentence'].values)
            df = df.explode('word')

            words = set(self.model.index).intersection(set(df['word']))


            df[['p', '!p']] = None,None
            for word in words:
                df.loc[df['word'] == word,'p'] = self.model[word]
                df.loc[df['word'] == word,'!p'] = 1-self.model[word]
            
            
            df_pr = df.groupby(by=lambda x: x)[['p','!p']].prod()


            df_pr['prob'] = df_pr['p']/ (df_pr['p'] + df_pr['!p'])
            df_pr['climate'] = df_pr['prob'].apply(lambda x: x >= self.threshold)
            df_store =  df_store.join(df_pr, how = 'left',lsuffix='old')

        else:
            df_store['prob'] = 0.5
        return df_store['prob'].values

    def save(self, name):

        wandb.config(self.get_params())
        artifact = wandb.Artifact(name, type = 'filter')
        with artifact.new_file('filter.pickle', 'wb') as f:
            dill.dump(self.__dict__,f)
        self.run.log_artifact(artifact)


    def load(self,name):
        artifact = self.run.use_artifact(name)
        path = artifact.download()
        run = self.run
        with open(os.path.join(path,'filter.pickle'), 'rb') as f:
            tmp_dic = dill.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dic)
            self.run = run

def dataset_to_xy(dataset):
    X = np.array([dataset.df_sentence.sentence.values]).T
    y = dataset.df_sentence.parent.isin(pd.concat([dataset.df_climate,dataset.df_skeptics,dataset.df_seed]).index)
    y = np.where(y == False,-1,y)
    return X, y




    


