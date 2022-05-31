from numpy import mod
from filter import Filter
from dataset import Dataset
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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


def evaluate_filter(f):
    test_cp = test.copy()
    test_cp['predicted'] = f.predict(test_cp)
    report = classification_report(test_cp['label'], test_cp['predicted'], output_dict=True)
    return report

def return_trained(f):
    print(f.min_count, f.alpha,f.threshold,f.model_size)
    f.train(climate_words, news_words)
    return f

'''
min_count = [10**x for x in range(0,6)]
model_size = [10**x for x in range(0,6)] 
alpha = [10*x for x in range(0,6)]
threshold = [0.1 *x for x in range(5,11)]
'''
min_count = [10**x for x in range(0,1)]
model_size = [10**x for x in range(2,3)] 
alpha = [10*x for x in range(0,1)]
threshold = [0.1 *x for x in range(7,8)]

df_filter = pd.DataFrame(list(product(min_count,model_size,alpha,threshold)), columns=['min_count','model_size','alpha','threshold'])
df_filter['f'] = df_filter.apply(lambda x: Filter(kwargs={'min_count': x['min_count'],'model_size': x['model_size'],'alpha': x['alpha'], 'threshold': x['threshold']}), axis = 1)

df_filter['f'] = simple_map(return_trained, df_filter['f'])
df_filter['report'] = simple_map(evaluate_filter, df_filter['f'])
df_filter['f1'] = df_filter['report'].apply(lambda x: x['macro avg']['f1-score'])


df_filter.to_pickle('./picklejar/filter_eval.pickle')

df_filter = pd.read_pickle('./picklejar/filter_eval.pickle')

best_filter =  (df_filter.sort_values('f1')).iloc[0]['f']

model.filter = best_filter

## Since embedding scheme can only be tested as part of the predictive model both have to be evaluated together

embedding_parms = ['doc2vecdm','doc2vecdbow','tfidf','bow','word2vecsum', 'word2vecmean']
author_info = [True, False]
predictive_model_parms = ['ExtraTree','DecisionTree','SGDClassifier','RidgeClassifier','PassiveAggressiveClassifier','AdaBoostClassifier','GradientBoostingClassifier', 'BaggingClassifier','ExtraTreeClassifier','RandomForestClassifier','BernoulliNB','LinearSVC','LogisticRegression','LogisticRegressionCV','MultinomialNB','NearestCentroid','NuSVC','Perceptron','SVC']

df_model = pd.DataFrame(columns=['min_count','model_size','alpha','threshold'])

data = d.encode_labels(d.apply_labels(d.df_sentence))
data['domain'] = d.domains(data)

data['climate'] = best_filter.predict(data)
## Remove unlabled and non-climate data
print('Filtering')
data = data[data['climate']]

labeled, unlabled = [x for _, x in data.group_by(data['labeled']!=-1)]

print(labeled, unlabled)
train, test = train_test_split(data)

for embedding in embedding_parms:
    for author in author_info:
        model.embedding_scheme = Embedding(model_type= embedding, author_info=author)
        print(embedding,author)
        model.embedding_scheme.train(train)
        train['vector'] = model.embedding_scheme.predict(train)
        for predictive_model in predictive_model_parms:
            model.predictive_model = Predictive_model(model=predictive_model)
            print(embedding,author,predictive_model)
            model.predictive_model.train(data)

