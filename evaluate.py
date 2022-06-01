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
unlabled_frac = [2**(-x) for x in range(8,11)]
labeled_frac = [0.1*x for x in range(1,10)] ## Will need a minimum of 10% for testing

df_eval = pd.DataFrame(columns=['min_count','model_size','alpha','threshold','embedding_parms','author_info','predictive_model_parms','unlabled','labeled', 'model'])

labeled_data, unlabled_data = [x for _, x in data.groupby(data['class']==-1)]

## First build the various traioning sets
i = 0
total = prod([len(min_count_parms), len(model_size_parms),len(alpha_parms),len(threshold_parms),len(unlabled_frac),len(labeled_frac)])
print(total)
for u in unlabled_frac:
    train_unlabeled = unlabled_data.sample(frac=u, random_state=1)
    for l in labeled_frac:
        train_labeled, test = train_test_split(labeled_data,train_size = l, random_state=1)
        train = pd.concat([train_labeled, train_unlabeled])
        ## Now initialise a filter
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
                        train['climate'] = f.predict(train)
                        for embedding_scheme in embedding_parms:
                            for author_info in author_info_parms:
                                e = Embedding(model_type=embedding_scheme,author_info=author_info)
                                e.train(train[train['climate']])
                                train['vector']= e.predict(train[train['climate']])
                                for predictive_model in predictive_model_parms:
                                    m = Predictive_model(model=predictive_model)
                                    m.train(train[train['climate']])
                                    model = Model()
                                    model.filter = f
                                    model.embedding_scheme = e
                                    model.predictive_model = m
                                    df_eval.loc[len(df_eval)] = [min_count,model_size,alpha,threshold,embedding_scheme,author_info,predictive_model,len(train_unlabeled), len(train_labeled), model]

