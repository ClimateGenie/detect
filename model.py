from dataset import Dataset
import numpy as np
from embedding import Embedding
from filter import Filter
from predictive_model import Predictive_model
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split


class Model():

    def __init__(self, kwargs = {
        'filter':{
            'model_size':100,
            'threshold':0.7
            },
        'embedding': {
            'model_type':'tfidf',
            'author_info':True,
            'args': {}
            },
        'predictive_model': {
            'model_type':'RandomForestClassifier',
            'args': {}
            }
        }) -> None:
        """

        kwargs are used to define parameters for each individual model

        kwargs = {

        'filter': {
            
            'alpha':
            'min_count':
            'threshold':
            'model_size':

        },

        'embedding': {
            'model_type':
            'args' = {}
        },

        'predictive_model': {
            'model_type': 
            'args' = {}
        }

        }

        """
        self.args  = kwargs

        self.filter = Filter( self.args['filter'])
        self.embedding_scheme = Embedding( model_type=self.args['embedding']['model_type'] ,author_info=self.args['embedding']['author_info'], kwargs=self.args['embedding']['args'])
        self.predictive_model = Predictive_model(model=self.args['predictive_model']['model_type'],kwargs=  self.args['predictive_model']['args'])

    def train(self,training_data):
        self.training_data = training_data

        self.filter.train(self.training_data[training_data['weak_climate']]['sentence'],self.training_data[~training_data['weak_climate']]['sentence'])
        self.training_data['climate'] = self.filter.predict(self.training_data)

        self.embedding_scheme.train(self.training_data[self.training_data['climate'] == True])
        self.training_data['vector'] = self.embedding_scheme.predict(self.training_data[self.training_data['climate'] == True])

        self.predictive_model.train(self.training_data[self.training_data['climate'] == True])



    def predict(self,df):
        df['climate'] = self.filter.predict(df)
        df.loc[df['climate'] == True,'vector'] = self.embedding_scheme.predict(df[df['climate'] == True])
        df.loc[df['climate'] == True,'class'] = self.predictive_model.predict(df[df['climate'] == True])
        df.loc[df['class'].isna(),'class'] = 0
        df['class'] = df['class'].apply(lambda x: int(x))

        return df
        
        


    def save(self):
        print('Pickling Filter')
        with open(os.path.join('picklejar','model'), 'wb') as f:

            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling Filter')
        with open(os.path.join('picklejar','model'), 'rb') as f:
            tmp_dic = dill.load(f)
            self.__dict__.update(tmp_dic)

if __name__ == "__main__":
    m = Model()
    d = Dataset()    
    training_data = d.encode_labels(d.apply_labels(d.df_sentence))
    training_data['domain'] = d.domains(training_data)
    training_data['weak_climate'] =  training_data['parent'].isin(np.concatenate((d.df_seed.index,d.df_climate.index,d.df_skeptics.index)))
    m.train(training_data)


    df = pd.DataFrame({'sentence': ['Climate Change is cool', 'Ice is not melting in antartica'], 'domain':['abc.net.au', None]})
    a = m.predict(df)
    a['predicted'] = d.encoder.inverse_transform(df['class'])
