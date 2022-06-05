from math import prod
from posixpath import altsep
from warnings import filters
from numpy import logical_xor, mod
from numpy.random.mtrand import random, sample
from pandas._libs.parsers import TextReader
from pandas.core.frame import DataFrame
from filter import Filter
from dataset import Dataset
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from itertools import product
from utils import simple_map, simple_starmap
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
f_test = pd.DataFrame(pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:ed47e4e1-353a-42cc-9f2e-0f199b85815a/Wiki-Doc-Dev.tsv', sep = '\t')) #This is just for my autocomplete so it knows the class 
data = d.encode_labels(d.apply_labels(d.df_sentence))
data['domain'] = d.domains(data)
data['weak_climate'] =  data['parent'].isin(np.concatenate((d.df_seed.index,d.df_climate.index,d.df_skeptics.index)))

min_count_parms =  [10**(2*x) for x in range(0,3)]
model_size_parms = [int(10**(x/2)) for x in range(1,7)] 
alpha_parms = [10**x for x in range(0,3)]
threshold_parms = [0.1 *x for x in range(0,11)]
embedding_parms =  ['doc2vecdm','doc2vecdbow','tfidf','bow','word2vecsum', 'word2vecmean']
author_info_parms = [True, False]
predictive_model_parms = ['ExtraTree','DecisionTree','SGDClassifier','RidgeClassifier','PassiveAggressiveClassifier','AdaBoostClassifier','GradientBoostingClassifier', 'BaggingClassifier','ExtraTreeClassifier','RandomForestClassifier','BernoulliNB','LinearSVC','LogisticRegression','NearestCentroid','SVC']
unlabled_frac = [np.e**(-x) for x in range(0,9)]
labeled_frac = [np.e**(-x - 0.2) for x in range(0,9)] ## Need to leave some testing data

df_eval = pd.DataFrame(columns=['min_count','model_size','alpha','threshold','embedding_parms','author_info','predictive_model_parms','unlabled','labeled', 'filter_evaluation', 'model_evaluation'])


labeled_data, unlabled_data = [x for _, x in data.groupby(data['class']==-1)]
model_size_parms.sort(reverse=True)


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
for u_frac in unlabled_frac:
    unlabeled_train = unlabled_data.sample(frac = u_frac, random_state=1)

    # Create all of the filters
    for min_count in min_count_parms:
        for alpha in alpha_parms:
            ## Due to how the filter is built, it can be trained once and altered each time for treashold and model_size
            f = Filter({'min_count':min_count,'alpha':alpha})
            f.train(unlabeled_train[unlabeled_train['weak_climate']]['sentence'],unlabeled_train[~unlabeled_train['weak_climate']]['sentence'])
            for model_size in model_size_parms:
                f.model_size = model_size
                f.model = f.model.iloc[0:model_size]
                f_test['prob'] = f.predict(f_test,return_prob=True)
                for threshold in threshold_parms:
                    f.threshold = threshold
                    f_test['predicted'] = f_test['prob'].apply(lambda x: x > threshold)
                    filter_report = classification_report(f_test['label'], f_test['predicted'], output_dict=True)
                    df_eval.loc[len(df_eval)] = [min_count,model_size,alpha,threshold,None,None,None,len(unlabeled_train),len(labeled_train), filter_report, None]

    ## Best flter is also small
    f = df_eval.loc[df_eval['filter_evaluation'][df_eval['model_size'] <= 500].apply(lambda x: x['macro avg']['f1-score']).sort_values(ascending = False).index[0]]
    f['filter'] = Filter({'min_count':f['min_count'],'alpha':f['alpha'],'model_size':f['model_size'],'threshold':f['threshold']})
    f['filter'].train(unlabeled_train[unlabeled_train['weak_climate']]['sentence'],unlabeled_train[~unlabeled_train['weak_climate']]['sentence'])

    # Given the Best Filter, What is the best model?
    probs = f['filter'].predict(unlabeled_train,return_prob=True)
    for threshold in [0,f['threshold']]:
        filtered_unlabeled_train = unlabeled_train[probs > threshold]
        for embedding in embedding_parms:
            for author_info in [True,False]:
                if author_info == True:
                    e = Embedding(model_type=embedding, author_info=author_info)
                    e.train(filtered_unlabeled_train)
                else:
                    e.author_info = False
                for l_frac in labeled_frac:
                    labeled_train, m_test = train_test_split(labeled_data, train_size=l_frac, random_state = 1)
                    filtered_labeled_train = labeled_train[f['filter'].predict(labeled_train,return_prob=True)  > threshold]
                    m_test['vector'] = e.predict(m_test)
                    for predictive_model in predictive_model_parms:
                        m = Predictive_model(model=predictive_model)
                        m.train(filtered_labeled_train)
                        predicted = m.predict(m_test)
                        report = classification_report(m_test['class'], predicted, output_dict = True)
                        df_eval.loc[len(df_eval)] = [f['min_count'],f['model_size'],f['alpha'],f['threshold'],embedding,author_info,predictive_model,len(unlabeled_train),len(labeled_train), f['filter_evaluation'], report]
            

evals = list(product(labeled_frac,unlabled_frac))
out = simple_map(eval_fracs,evals)
df_eval = pd.concat(out)
df_eval.to_pickle('./picklejar/eval.pickle')
