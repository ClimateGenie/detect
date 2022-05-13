from json import load
from dataset import Dataset
from embedding import Embedding
from filter import Filter
from predictive_model import Model
from utils import *


d = Dataset()

if d.reload:
    
    d.add_seed_data()
    climate_words = d.climate_words()
    news_words = d.news_words()
    f = Filter(climate_words,news_words, min_count=100)
    d.filter_for_climate(f, threshold=0.8)

    training_data = d.df_filtered['sentence'].apply(word_token)
    e = Embedding(training_data)
    d.vectorise(e)
    d.save_filtered()
else:
    d.load_filtered()
    d.add_seed_data()
    e = Embedding(load = True)
    f = Filter(load=True)

while True:
    d.apply_labels()
    d.encode_labels()

    m = Model(d.df_filtered)
    d.predict_unlabeled(m)
    d.get_labels(n=10)
