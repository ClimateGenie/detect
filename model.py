from dataset import Dataset
from embedding import Embedding
from filter import Filter
from predictive_model import Predictive_model
from utils import *


class Model():

    def __init__(self) -> None:
        pass

    def load()
        

    def build_model(self):
        d = Dataset()    
        d.add_seed_data()

        climate_words = d.climate_words()
        news_words = d.news_words()
        f = Filter(climate_words,news_words, min_count=100)


        training_data = d.df_filtered['sentence'].apply(word_token)
        e = Embedding(training_data)
        d.vectorise(e)
        d.save_filtered()
        m = Predictive_model(d.df_filtered)

        


