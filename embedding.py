from dataset import Dataset
import dill
from utils import *
from random import choice, random
import gensim
import os
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import collections
import pickle


class Embedding():

    def __init__(self,training_data,dm=0,vect_size=200,window = 5, hs = 1, epochs = 20):

        self.model_string = f'embedding-{dm}-{vect_size}-{window}-{hs}-{epochs}.pickle'
        self.pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'picklejar',self.model_string)
        self.training_data = [gensim.models.doc2vec.TaggedDocument(word_token(x),[i]) for i,x in enumerate(training_data)]
        [self.vect_size, self.dm, self.window, self.hs, self.epochs] = [vect_size, dm, window, hs, epochs] 
        try:
            self.load()
        except FileNotFoundError:
            self.build_model()
    

    def build_model(self):
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vect_size,dm=self.dm,window=self.window,hs=self.hs,epochs = self.epochs)
        self.model.build_vocab(corpus_iterable = self.training_data)
        self.model.train(self.training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)
        self.save()




    def find_closest(self,sentence, num = 1, print_out = False):
        inferred_vector = self.model.infer_vector(word_token(sentence))
        sims = self.model.dv.most_similar([inferred_vector], topn=num)
        if print_out:
            for i in range(len(sims)):
                print(f'{str(i)}: '+  ' '.join(self.training_data[sims[i][0]].words))
        return sims[0][0]

    def validate(self):
        docs = [x.words for x in self.training_data] 
        most_similar = simple_map(self.find_closest,docs)
        c = 0
        for i,j in enumerate(docs):
            if i == most_similar[i]:
                 c+=1
        print(c/len(self.training_data))

    def eye_test(self):
        test = choice([x.words for x in self. training_data])
        print('Test: '+' '.join(test))
        self.find_closest(' '.join(test), num = 5, print_out=True)

    
    def save(self):
        print('Pickling')
        with open(self.pickle_path, 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling')
        with open(self.pickle_path, 'rb') as f:
            tmp_dic = dill.load(f)
            if tmp_dic['training_data'] == self.training_data:
                self.__dict__.clear()
                self.__dict__.update(tmp_dic)
            else:
                self.build_model()



if __name__ == "__main__":
    d = Dataset()
    training_data = flatten(flatten([simple_map(sent_token,d.df_climate['article']), simple_map(sent_token, d.df_skeptics['article'])]))
    e = Embedding(training_data)
    e.eye_test()
