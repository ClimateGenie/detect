from dataset import Dataset
from embedding import Embedding
from filter import Filter
from predictive_model import Model
from utils import *


d = Dataset()

climate_words = d.climate_words()
news_words = d.news_words()
f = Filter(climate_words,news_words)


training_data = d.filter_for_climate(f)['sentence'].apply(word_token).values.tolist()
e = Embedding(training_data)
d.vectorise(e)

d.apply_labels()

