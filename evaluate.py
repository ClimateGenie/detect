from math import prod
from posixpath import altsep
from warnings import filters
from numpy import mod
from numpy.random.mtrand import random, sample
from pandas._libs.parsers import TextReader
from pandas.core.frame import DataFrame
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

min_count_parms =  [10**x for x in range(0,4)]
model_size_parms = [int(10**(x/2)) for x in range(2,12)] 
alpha_parms = [10*x for x in range(0,3)]
threshold_parms = [0.1 *x for x in range(0,11)]
embedding_parms = ['doc2vecdm','doc2vecdbow','tfidf','bow','word2vecsum', 'word2vecmean']
author_info_parms = [True, False]
predictive_model_parms = ['ExtraTree','DecisionTree','SGDClassifier','RidgeClassifier','PassiveAggressiveClassifier','AdaBoostClassifier','GradientBoostingClassifier', 'BaggingClassifier','ExtraTreeClassifier','RandomForestClassifier','BernoulliNB','LinearSVC','LogisticRegression','LogisticRegressionCV','MultinomialNB','NearestCentroid','NuSVC','Perceptron','SVC']
unlabled_frac = [2**(-x) for x in range(0,9)]
labeled_frac = [2**(-x - 0.2) for x in range(0,9)] ## Need to leave some testing data

df_eval = pd.DataFrame(columns=['min_count','model_size','alpha','threshold','embedding_parms','author_info','predictive_model_parms','unlabled','labeled', 'filter_evaluation', 'model_evaluation'])
labeled_data, unlabled_data = [x for _, x in data.groupby(data['class']==-1)]


"""
What Questions do we want to answer?

Basic Tuning:
 - What is the best filter
 - What is the best embedding scheme + model combo

Is it worthwhile to gather more data:
 - Does the filter improve with more data?
 - Does the model improve with more unlabled_data
 - Does the model improve with more labeled_data
 - Does author_info improve the model?
 - Do I even need a filter -> is embedding scheme improved by more test overall
 - If inconclusive -> is there a tradeoff between the best filter and the best model

"""
f_test = pd>DataFrame(pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:ed47e4e1-353a-42cc-9f2e-0f199b85815a/Wiki-Doc-Dev.tsv', sep = '\t')) #This is just for my autocomplete so it knows the class 
for l_frac in labeled_frac:
    for u_frac in unlabled_frac:
        labeled_train, m_test = train_test_split(labeled_data, train_size=l_frac, random_state = 1)
        unlabeled_train = unlabled_data.sample(frac = u_frac, random_state=1)
        train = pd.concat([labeled_train,unlabeled_train])

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
                    f_test['prob'] = f.predict(f_test,return_prob=True)
                    for threshold in threshold_parms:
                        f.threshold = threshold
                        f_test['predicted'] = f_test['prob'].apply(lambda x: x > threshold)
                        filter_report = classification_report(f_test['label'], f_test['predicted'], output_dict=True)
                        df_eval.loc[len(df_eval)] = [min_count,model_size,alpha,threshold,None,None,None,len(unlabeled_train),len(labeled_train), filter_report, None]
                        print(df_eval)

        ## Best flter is also small
        f = df_eval.loc[df_eval['filter_evaluation'][df_eval['model_size'] <= 500].apply(lambda x: x['macro avg']['f1-score']).sort_values(ascending = False).index[0]]
        f['filter'] = Filter({'min_count':f['min_count'],'alpha':f['alpha'],'model_size':f['model_size'],'threshold':f['threshold']})
        f['filter'].train(train[train['weak_climate']]['sentence'],train[~train['weak_climate']]['sentence'])

        
        # Given the Best Filter, What is the best model?
        probs = f['filter'].predict(train,return_prob=True)
        for threshold in threshold_parms:
            filtered_train = train[probs > threshold]
            for embedding in embedding_parms:
                author_info = True
                e = Embedding(model_type=embedding, author_info=author_info)
                e.train(filtered_train)
                train.loc[train['class'] != -1]['vector'] = e.predict(train[train['class'] != -1])
                test['vector'] = e.predict(test)
                for predictive_model in predictive_model_parms:
                    m = Predictive_model(model=predictive_model)
                    m.train(train)
                    predicted = m.predict(test)
                    report = classification_report(test['class'], predicted, output_dict = True)
                    df_eval.loc[len(df_eval)] = [f['min_count'],f['model_size'],f['alpha'],f['threshold'],embedding,author_info,predictive_model,len(unlabled_data),len(labeled_train), f['filter_report'], report]
                    print(df_eval)
                #Simply togle the embedding scheme author_info rather than 
                author_info = False
                e.author_info = author_info
                train.loc[train['class'] != -1]['vector'] = e.predict(train[train['class'] != -1])
                test['vector'] = e.predict(test)
                for predictive_model in predictive_model_parms:
                    m = Predictive_model(model=predictive_model)
                    m.train(train)
                    predicted = m.predict(test)
                    report = classification_report(test['class'], predicted, output_dict = True)
                    df_eval.loc[len(df_eval)] = [f['min_count'],f['model_size'],f['alpha'],threshold,embedding,author_info,predictive_model,len(unlabled_data),len(labeled_train), f['filter_report'], report]
                    print(df_eval)

df_eval.to_pickle('./picklejar/eval.pickle')


