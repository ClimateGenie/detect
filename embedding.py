from dataset import Dataset
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

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=vect_size,dm=dm,window=window,hs=1,epochs = epochs)
        self.model.build_vocab(corpus_iterable = self.training_data)
        self.model.train(self.training_data, total_examples=self.model.corpus_count, epochs = self.model.epochs)


    def find_closest(self,sentence, num = 1):
        inferred_vector = self.model.infer_vector(word_token(sentence))
        sims = self.model.dv.most_similar([inferred_vector], topn=num)
        for i in range(len(sims)):
            print(' '.join(self.training_data[sims[i][0]].words))

    def validate(self):
        ranks = []
        for doc_id in range(len(self.training_data)):
            inferred_vector = self.model.infer_vector(self.training_data[doc_id].words)
            sims = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)
        count = collections.Counter(ranks)
        print(count[0]/len(self.training_data))

    def eye_test(self):
        test = choice([x.words for x in self. training_data])
        print(' '.join(test))
        self.find_closest(' '.join(test), num = 5)


if __name__ == "__main__":
    d = Dataset()
    training_data = flatten(flatten([simple_map(sent_token,d.df_climate['article']), simple_map(sent_token, d.df_skeptics['article'])]))
    e = Embedding(training_data)
    e.eye_test()
