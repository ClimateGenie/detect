from gensim import models
import pandas as pd
from numpy import mod, negative
from dataset import Dataset
import dill
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils import *
from random import choice, random
import gensim
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import logging
import collections
import pickle
from scipy.sparse import csr_matrix, hstack


class Embedding():

    def __init__(self,training_data,model_type = 'doc2vecdm', author_info = False, kwargs = {}):


        self.model_type = model_type
        self.training_data = training_data['sentence']
        self.args = kwargs
        self.author_info = author_info


    

    def train(self):
        print('Training Embedding Scheme')

        if self.model_type == 'doc2vecdm':
            training_data = [gensim.models.doc2vec.TaggedDocument(word_token(x),[i]) for i,x in self.training_data.iteritems()]
            self.model = gensim.models.doc2vec.Doc2Vec(**self.args)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)


        elif self.model_type == 'doc2vecdbow':
            training_data = [gensim.models.doc2vec.TaggedDocument(x,[i]) for i,x in self.training_data.iteritems()]
            self.model = gensim.models.doc2vec.Doc2Vec(**self.args)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)

        elif self.model_type == 'tfidf':
            self.model = TfidfVectorizer(**self.args)
            self.model.fit(self.training_data)

        elif self.model_type == 'bow':
            self.model = CountVectorizer(**self.args)
            self.model.fit(self.training_data)

        elif self.model_type in ['word2vecsum', 'word2vecmean']:
            training_data = simple_map(word_token, self.training_data)
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
        ## First do senteces found in the dataset
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
            print(df['author_vect'])
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
            if self.model_type == 'word2vecmean':
                return np.mean(vect_list, axis=0)
            elif self.model_type == 'word2vecsum':
                return np.sum(vect_list, axis=0)
    

