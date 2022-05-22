from sklearn import semi_supervised
from dataset import Dataset
from embedding import Embedding
from filter import Filter
from predictive_model import Predictive_model
import pandas as pd
from utils import *


class Model():

    def __init__(self, kwargs = {
        'filter':{
            'min_count': 5000
            },
        'embedding': {
            'model_type':'tfidf',
            'args': {}
            },
        'predictive_model': {
            'model_type':'random_forrest',
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

        },

        'embedding': {
            'model_type' =['doc2vecdbow','doc2vecdm','tfidf','bow', 'word2vecsum', 'word2vecmean' ]
            'args' = {}
        },

        'predictive_model': {
            'model_type': ['semi_supervised', 'n_neighbors','rbf_svm','gaussian_process','decision_tree','random_forrest','nn','adaboost','qda'] 
            'args' = {}
        }

        }

        """
        self.args  = kwargs


    def train(self):
        d = Dataset()    
        print(d.df_sentence[d.df_sentence.index.duplicated()])
        self.training_data =d.encode_labels(d.apply_labels(d.df_sentence))
        print(self.training_data)

        f = Filter(d.climate_words(),d.news_words(), **self.args['filter'])
        f.train()
        self.training_data['climate'] = f.predict(self.training_data)
        print(self.training_data)

        e = Embedding(self.training_data[self.training_data['climate'] == True], model_type=self.args['embedding']['model_type'], kwargs=self.args['embedding']['args'])
        e.train()
        self.training_data['vector'] = e.predict(self.training_data[self.training_data['climate'] == True])
        print(self.training_data)

        m = Predictive_model(self.training_data[self.training_data['climate'] == True], model=self.args['predictive_model']['model_type'],kwargs=  self.args['predictive_model']['args'])
        m.train()
        print(self.training_data)


        self.d, self.f, self.e, self.m = d,f,e,m

    def predict(self,series):
        df = pd.DataFrame(series, columns=['sentence'])
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
    test = pd.Series(['Climate change is cool', 'Ice is not melting in antartica'])
    a = m.predict(test)
