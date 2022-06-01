from math import prod
from posixpath import altsep
from numpy import mod
from numpy.random.mtrand import sample
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
import numpy as np


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
data = d.encode_labels(d.apply_labels(d.df_sentence))
data['domain'] = d.domains(data)
data['weak_climate'] =  data['parent'].isin(np.concatenate((d.df_seed.index,d.df_climate.index,d.df_skeptics.index)))

min_count_parms = [10**x for x in range(0,4)]
model_size_parms = [10**x for x in range(0,5)] 
alpha_parms = [10*x for x in range(0,3)]
threshold_parms = [0.1 *x for x in range(5,11)]
embedding_parms = ['doc2vecdm','doc2vecdbow','tfidf','bow','word2vecsum', 'word2vecmean']
author_info_parms = [True, False]
predictive_model_parms = ['ExtraTree','DecisionTree','SGDClassifier','RidgeClassifier','PassiveAggressiveClassifier','AdaBoostClassifier','GradientBoostingClassifier', 'BaggingClassifier','ExtraTreeClassifier','RandomForestClassifier','BernoulliNB','LinearSVC','LogisticRegression','LogisticRegressionCV','MultinomialNB','NearestCentroid','NuSVC','Perceptron','SVC']
unlabled_frac = [2**(-x) for x in range(0,12)]
labeled_frac = [0.1*x for x in range(1,10)] ## Will need a minimum of 10% for testing

df_eval = pd.DataFrame(columns=['min_count','model_size','alpha','threshold','embedding_parms','author_info','predictive_model_parms','unlabled','labeled', 'filter', 'embedding','model'])

labeled_data, unlabled_data = [x for _, x in data.groupby(data['class']==-1)]


"""
What Questions do we want to answer?

Basic Tuning:
 - What is the best filter
 - What is the best embedding scheme + model combo

Is it worthwhile to gather more data:
 - Does the model improve with more unlabled_data
 - Does the model improve with more labeled_data
 - Does author_info improve the model?
 - Do I even need a filter -> is embedding scheme improved by more test overall
 - If inconclusive -> is there a tradeoff between the best filter and the best model
 - Is there an ensemble model that improves on indivisual model

"""

## What is the best filter
"""
We itterate through the internal parts of the filter using grid search
Use all data as this is speedy af

"""



train = data.copy()
i = 0
total = prod([len(min_count_parms), len(model_size_parms),len(alpha_parms),len(threshold_parms)])

# Create all of the filters
for min_count in min_count_parms:
    for alpha in alpha_parms:
	## Due to how the filter is built, it can be trained once and altered each time for treashold and model_size
        f = Filter({'min_count':min_count,'alpha':alpha})
        f.train(train[train['weak_climate']]['sentence'],train[~train['weak_climate']]['sentence'])
        for model_size in model_size_parms:
            f.model_size = model_size
            indecies = f.stat.sort_values('score',ascending = False).iloc[0:f.model_size,].index
            f.model = f.norm.loc[indecies]
            for threshold in threshold_parms:
                f.threshold = threshold
                df_eval.loc[len(df_eval)] = [min_count,model_size,alpha,threshold,None,None,None,len(unlabled_data),len(labeled_data), copy(f),None,None]
                print(df_eval)

test = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:ed47e4e1-353a-42cc-9f2e-0f199b85815a/Wiki-Doc-Dev.tsv', sep = '\t')






