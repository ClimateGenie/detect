from filter import Filter
from dataset import Dataset
import pandas as pd
from sklearn.metrics import classification_report
from itertools import product
from utils import simple_map
from copy import copy
from model import Model
from embedding import Embedding
from predictive_model import Predictive_model


"""
There will exist some optimal kwargs that will maximise any given evaluator for a model

Due to the nature of the pipline, this evaluation can be done in two stages:

    1. Evaluate the Filter
    2. Evaluate the Embedding Scheme + the Predictive_model

The filter can be evaluated seperatley as its goal is to determine if a sentence is about climate change or not.

The parameters for the filter is as follows:
    Min count -> provides a minumum number of aoccurances, higher means more confidence in pr(word) being correct in the population
    Threshold -> pr for true vs false
    Alpha -> Smoothing cooef
    model_size -> number of words taken into account in final model, smaller model = greater speed.

This is then being tested using the Clima-Text Dataset which gives climate sentence and non climate sentence

"""

d = Dataset()

model = Model()

climate_words = d.climate_words().copy()
news_words = d.news_words().copy()
test = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:ed47e4e1-353a-42cc-9f2e-0f199b85815a/Wiki-Doc-Dev.tsv', sep = '\t')
f = Filter(climate_words, news_words)


def evaluate_filter(row):
    test_cp = test.copy()
    f = row['f']
    f.min_count = row['min_count']
    f.model_size = row['model_size']
    f.alpha = row['alpha']
    f.threshold = row['threshold']
    f.train()
    test_cp['predicted'] = f.predict(test_cp)
    report = classification_report(test_cp['label'], test_cp['predicted'], output_dict=True)
    return report

min_count = [1,10,100,1000,5000]
model_size = [100,200,500, 1000, 5000, 100000]
alpha = [1,5,10]
threshold = [0.5,0.6,0.7,0.8,0.9]

df_filter = pd.DataFrame(list(product(min_count,model_size,alpha,threshold)), columns=['min_count','model_size','alpha','threshold'])
df_filter['f'] = [copy(f) for x in range(len(df_filter))]


df_filter['report'] = simple_map(evaluate_filter, df_filter.to_dict('records'))
df_filter['f1'] = df_filter['report'].apply(lambda x: x['macro avg']['f1-score'])
df_filter.to_pickle('./picklejar/filter_eval.pickle')


best_filter =  df_filter.sort_values('f1').iloc[0,'f']

model.f = best_filter








