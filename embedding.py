from dataset import Dataset, istarmap
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import gensim
import nltk
import os
import logging
import random
from multiprocessing import Manager, Pool
import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(format='%(message)s', level=logging.INFO)
import pickle


class Embedding():

    def __init__(self,training_data,tokenisation_function,dm=0,vect_size=200,window = 5, hs = 1, epochs = 3):

        self.model_string = f'embedding-{dm}-{vect_size}-{window}-{hs}-{epochs}.pickle'
        self.pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'picklejar',self.model_string)
        self.training_data = training_data
        self.tokenisation_function = tokenisation_function

        if os.path.exists(self.pickle_path):
            with open(self.pickle_path,'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vect_size,dm=dm,window=window,hs=1,epochs = 3)
            self.model.build_vocab(corpus_iterable = self.training_data)
            self.model.train(self.training_data, total_examples=self.model.corpus_count, epochs=3)
            with open(self.pickle_path,'wb') as f:
                pickle.dump(self.model,f)


    def find_closest(self,sentence):
        inferred_vector = self.model.infer_vector(self.tokenisation_function(sentence))
        sims = self.model.dv.most_similar([inferred_vector], topn=1)
        print(' '.join(self.training_data[sims[0][0]].words))

if __name__ == "__main__":
    d = Dataset()
    d.gather_embedding_training(sample = 0.3)
    e = Embedding(d.embedding_train,d.word_token)
