from json import load
from dataset import Dataset
from embedding import Embedding
from filter import Filter
from predictive_model import Model
from utils import *


d = Dataset()

if d.reload:

    climate_words = d.climate_words()
    news_words = d.news_words()
    f = Filter(climate_words,news_words)
    d.filter_for_climate(f, threshold=0.9)

    training_data = d.df_filtered['sentence'].apply(word_token)
    e = Embedding(training_data)
    d.vectorise(e)

    d.add_seed_data(e,f)
    d.save_filtered()
else:
    d.load_filtered()
    e = Embedding(load = True)
    f = Filter(load=True)

while True:
    d.apply_labels()
    d.encode_labels()

    m = Model(d.df_filtered)
    d.predict_unlabeled(m)
    d.get_labels(n=10)
