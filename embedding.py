from dataset import Dataset, istarmap
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import gensim
import nltk
import logging
import random
from multiprocessing import Manager, Pool
import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(format='%(message)s', level=logging.INFO)


d = Dataset()

d.gather_embedding_training(sample = 0.3)
d.gather_subclaim_data()

model = gensim.models.doc2vec.Doc2Vec(vector_size=300)

model.build_vocab(corpus_iterable = d.embedding_train)

model.train(d.embedding_train, total_examples=model.corpus_count, epochs=3)

def find_closest(sentence, model = model):
    inferred_vector = model.infer_vector(gensim.utils.simple_preprocess(sentence))
    sims = model.dv.most_similar([inferred_vector], topn=1)
    print(' '.join(d.embedding_train[sims[0][0]].words))
