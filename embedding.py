from gensim import models
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
logging.basicConfig(format='%(message)s', level=logging.INFO)
import collections
import pickle


class Embedding():

    def __init__(self,training_data,model_type = 'doc2vecdm', **kwargs):


        self.model_type = model_type
        if self.model_type == 'doc2vecdm':
            self.vect_size = kwargs.get('vect_size', 100)
            self.window = kwargs.get('window', 5)
            self.dm_concat = kwargs.get('dm_concat', 0)
            self.hs = kwargs.get('hs',3)
            self.alpha = kwargs.get('alpha',  0.025)
            self.epochs = kwargs.get('epochs',3)

        elif self.model_type == 'doc2vecdbow':
            self.vect_size = kwargs.get('vect_size', 100)
            self.window = kwargs.get('window', 5)
            self.dbow_words = kwargs.get('dbow_words', 0)
            self.hs = kwargs.get('hs',3)
            self.alpha = kwargs.get('alpha',  0.025)
            self.epochs = kwargs.get('epochs',3)

        elif self.model_type == 'tfidf':
            self.max_df = kwargs.get('max_df', 1)
            self.min_df = kwargs.get('min_df', 0)
            self.binary = kwargs.get('binary', False)
        
        elif self.model_type == 'bow':
            self.max_df = kwargs.get('max_df', 1)
            self.min_df = kwargs.get('min_df', 0)
            self.binary = kwargs.get('binary', False)

        elif self.model_type == 'word2vecsum':
            self.vect_size = kwargs.get('vect_size', 100)
            self.alpha = kwargs.get('alpha',  0.025)
            self.hs = kwargs.get('hs',0)
            self.negative = kwargs.get('negative',5)
            self.ns_exponent = kwargs.get('ns_exponent', 0.75)
            self.epochs = kwargs.get('epochs',3)
            
        elif self.model_type == 'word2vecmean':
            self.vect_size = kwargs.get('vect_size', 100)
            self.alpha = kwargs.get('alpha',  0.025)
            self.hs = kwargs.get('hs',0)
            self.negative = kwargs.get('negative',5)
            self.ns_exponent = kwargs.get('ns_exponent', 0.75)
            self.epochs = kwargs.get('epochs',3)

        else:
            raise ValueError()

        self.doc_num = len(training_data)

        self.pickle_string = '-'.join([f'{key}:{str(x)}' for key, x in self.__dict__.items()])

        self.training_data = training_data


    

    def train(self):

        if self.model_type == 'doc2vecdm':
            training_data = [gensim.models.doc2vec.TaggedDocument(word_token(x),[i]) for i,x in self.training_data.iteritems()]
            self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vect_size,dm=1,window=self.window,hs=self.hs,epochs = self.epochs, dm_concat=self.dm_concat)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)


        elif self.model_type == 'doc2vecdbow':
            training_data = [gensim.models.doc2vec.TaggedDocument(x,[i]) for i,x in self.training_data.iteritems()]
            self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vect_size,dm=0,window=self.window,hs=self.hs,epochs = self.epochs, dbow_words=self.dbow_words)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)

        elif self.model_type == 'tfidf':
            self.model = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, tokenizer=word_token, binary=self.binary)
            self.model.fit(self.training_data)

        elif self.model_type == 'bow':
            self.model = CountVectorizer(max_df=self.max_df, min_df=self.min_df, tokenizer=word_token, binary=self.binary)
            self.model.fit(self.training_data)

        elif self.model_type in ['word2vecsum', 'word2vecmean']:
            training_data = simple_map(word_token, self.training_data)
            print(training_data)
            self.model = gensim.models.word2vec.Word2Vec(vector_size=self.vect_size, alpha = self.alpha, hs=self.hs,negative = self.negative, ns_exponent=self.ns_exponent,epochs=self.epochs)
            self.model.build_vocab(corpus_iterable = training_data)
            self.model.train(training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)


    def predict(self,sentence):
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
            elif self.model_type == 'doc2vecsum':
                return np.sum(vect_list, axis=0)
    
    def save(self):
        print('Pickling Filter')
        with open(os.path.join('picklejar','embedding',self.pickle_string), 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling Filter')
        with open(os.path.join('picklejar','embedding',self.pickle_string), 'rb') as f:
            tmp_dic = dill.load(f)
            self.__dict__.update(tmp_dic)


