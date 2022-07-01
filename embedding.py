import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils import *
import gensim
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix, hstack


class Embedding():

    def __init__(self,model_type = 'doc2vecdm', author_info = False, kwargs = {}):


        self.model_type = model_type
        self.args = kwargs
        self.author_info = author_info


    

    def train(self, training_data):
        print(f'Training {self.model_type} Scheme on {len(training_data)} examples')
        self.training_data = training_data

        if self.model_type == 'doc2vecdm':
            training_data = [gensim.models.doc2vec.TaggedDocument(word_token(x),[i]) for i,x in self.training_data['sentence'].iteritems()]
            self.model = gensim.models.doc2vec.Doc2Vec(**self.args)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)


        elif self.model_type == 'doc2vecdbow':
            training_data = [gensim.models.doc2vec.TaggedDocument(word_token(x),[i]) for i,x in self.training_data['sentence'].iteritems()]
            self.model = gensim.models.doc2vec.Doc2Vec(**self.args)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)

        elif self.model_type == 'tfidf':
            self.model = TfidfVectorizer(**self.args)
            train = self.training_data['sentence'].dropna()
            self.model.fit(train)

        elif self.model_type == 'bow':
            self.model = CountVectorizer(**self.args)
            train = self.training_data['sentence'].dropna()
            self.model.fit(train)

        elif self.model_type in ['word2vecsum', 'word2vecmean']:
            training_data = map(word_token, self.training_data['sentence'])
            self.model = gensim.models.word2vec.Word2Vec(**self.args)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)

        if self.author_info:
            self.authors = pd.read_json('https://raw.githubusercontent.com/drmikecrowe/mbfcext/main/docs/v3/csources-pretty.json', orient='index')
            self.authors.loc['dummy'] = np.nan
            self.encoders = {}
            for col in ['b','r','c','a','p']:
                le= LabelEncoder()
                le.fit(self.authors[col])
                self.encoders[col] = le



    def predict(self, df):
        ## First do sentece found in the dataset
        if self.model in ['doc2vecdbow', 'doc2vecdm']:
            vectors = pd.Series(self.model.dv[self.training_data.index].tolist())
            vectors.index = self.training_data.index
            df['vector'] = df.join(vectors.rename('vectors'), how = 'left')['vectors']
            df['vector'] = df['vector'].astype(object)

        else:
            df['vector'] = None
        

        for index, row in df[df['vector'].isna()].iterrows():
           vect  = self.predict_single(row['sentence'])
           df.at[index, 'vector'] = vect
        df['vector'] = df['vector'].apply(lambda x: csr_matrix(x))

        if self.author_info:
            df = df.merge(self.authors[['b','r','c','a','p']], left_on = 'domain', right_index = True,how = 'left')
            for key, e in self.encoders.items():
                df[key] = e.transform(df[key])
            df['author_vect'] = df[['b','r','c','a','p']].values.tolist()
            df['author_vect'] = df['author_vect'].apply(lambda x: csr_matrix(np.array(x)))
            df['vector'] = df.apply(lambda x: hstack([x['author_vect'], x['vector']] ), axis =1)

        return df['vector']


    def predict_single(self,sentence):
        if self.model_type in  ['doc2vecdbow','doc2vecdm']:
            return self.model.infer_vector(word_token(sentence))
        elif self.model_type  in ['tfidf','bow']:
            return self.model.transform([sentence])[0]
        elif self.model_type in ['word2vecsum', 'word2vecmean']:
            vect_list = []
            for word in word_token(sentence):
                try:
                    vect_list.append(self.model.wv[word])
                except:
                    pass
            if len(vect_list) > 0:
                if self.model_type == 'word2vecmean':
                    return np.mean(vect_list, axis=0)
                elif self.model_type == 'word2vecsum':
                    return np.sum(vect_list, axis=0)
            else:
                return [0 for _ in range(self.model.vector_size)]
    

