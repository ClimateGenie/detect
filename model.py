from enum import auto
from sklearn import semi_supervised
from dataset import Dataset
from embedding import Embedding
from filter import Filter
from predictive_model import Predictive_model
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split


class Model():

    def __init__(self, kwargs = {
        'filter':{
            'model_size':50,
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
        self.d = Dataset()    

        self.training_data, self.test_data = train_test_split(self.d.df_sentence)


    def train(self):

        f = Filter(self.d.climate_words(),self.d.news_words(), self.args['filter'])
        f.train()
        self.training_data['climate'] = f.predict(self.training_data)

        self.training_data['domain'] = self.d.domains(self.training_data)
        e = Embedding(self.training_data[self.training_data['climate'] == True], model_type=self.args['embedding']['model_type'] ,author_info=self.args['embedding']['author_info'], kwargs=self.args['embedding']['args'])
        e.train()

        self.training_data['vector'] = e.predict(self.training_data[self.training_data['climate'] == True])

        self.training_data = self.d.encode_labels(self.d.apply_labels(self.training_data))

        m = Predictive_model(self.training_data[self.training_data['climate'] == True], model=self.args['predictive_model']['model_type'],kwargs=  self.args['predictive_model']['args'])
        m.train()


        self.f, self.e, self.m = f,e,m

    def predict(self,df):
        df['climate'] = self.f.predict(df)
        df.loc[df['climate'] == True,'vector'] = self.e.predict(df[df['climate'] == True])
        df.loc[df['climate'] == True,'class'] = self.m.predict(df[df['climate'] == True])
        df.loc[df['class'].isna(),'class'] = 0
        df['class'] = df['class'].apply(lambda x: int(x))
        df['predicted'] = self.d.encoder.inverse_transform(df['class'])

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
    m.train()
    df = pd.DataFrame({'sentence': ['Climate Change is cool', 'Ice is not melting in antartica'], 'domain':['abc.net.au', 'infowars.com']})
    a = m.predict(df)
