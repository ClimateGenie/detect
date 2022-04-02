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
#logging.basicConfig(format='%(message)s', level=logging.INFO)


d = Dataset()

d.gather_embedding_training(sample = 0.1)
d.gather_subclaim_data()

model = gensim.models.doc2vec.Doc2Vec(vector_size=200)

model.build_vocab(corpus_iterable = d.embedding_train)

i = 1
le = LabelEncoder()
while i:
    model.train(d.embedding_train, total_examples=model.corpus_count, epochs=1)
    points = []
    labels = []
    for sen in d.validate_sub:
        points.append(model.infer_vector(sen.words))
        labels.append(''.join([str(x) for x in sen.tags]))
    labels = le.fit_transform(labels)

    print(silhouette_score(points,labels))

