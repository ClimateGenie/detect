from gensim import models
import pandas as pd
from numpy import mod, negative
from dataset import Dataset
import dill
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils import *
from random import choice, random
import gensim
import os
import numpy as np
import logging
import collections
import pickle


class Embedding():

    def __init__(self,training_data,model_type = 'doc2vecdm', kwargs = {}):


        self.model_type = model_type
        self.training_data = training_data['sentence']
        self.args = kwargs


    

    def train(self):
        print('Traing Embedding Scheme')

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


    def predict(self, df):
        ## First do senteces found in the datase
        vectors = pd.Series(self.model.dv[self.training_data.index].tolist())
        vectors.index = self.training_data.index
        df['vector'] = df.join(vectors.rename('vectors'), how = 'left')['vectors']
        df['vector'] = df['vector'].astype(object)
        

        for index, row in df[df['vector'].isna()].iterrows():
           vect  = self.predict_single(row['sentence'])
           df.at[index, 'vector'] = vect

        
        return df['vector']


    def predict_single(self,sentence):
        if self.model_type in  ['doc2vecdbow','doc2vecdm']:
            return self.model.infer_vector(word_token(sentence))
        elif self.model_type  in ['tfidf','bow']:
            return self.model.transform([sentence])[0].A
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
    

