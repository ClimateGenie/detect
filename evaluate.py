from math import prod
from posixpath import altsep
from warnings import filters
from numpy import logical_xor, mod
from numpy.random.mtrand import random, sample
from pandas._libs.parsers import TextReader
from pandas.core.frame import DataFrame
from pandas.io.pytables import AppendableFrameTable
from filter import Filter
from dataset import Dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from itertools import product, starmap
from utils import *
from copy import copy
from model import Model
from embedding import Embedding
from predictive_model import Predictive_model
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import scipy.stats as st


def evaluate_filter(train,test):
    # confidence interval for 1* margin of error
    # n = z^2*p(1-p)/e^2
    min_count_parms =  [int(st.norm.ppf(x)**2 * 0.5*(1-0.5)/(0.01**2)) for x in [0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.98, 0.99, 0.999, 0.9999, 0.99999]]
    min_count_parms.sort()
    model_size_parms = [int(10**(x/4)) for x in range(4,21)] 
    model_size_parms.sort(reverse=True)
    threshold_parms = [x/100 for x in range(0,101)]
    print( f'Building Filters for {len(train)} examples')
    out = []
    ## Due to how the filter is built, it can be trained once and altered each time for parameters
    f = Filter()
    #f.train(train[train['weak_climate']]['sentence'],train[~train['weak_climate']]['sentence'])
    target_count = dict(Counter(flatten(map(word_token,train[train['weak_climate']]['sentence']))))
    general_count = dict(Counter(flatten(map(word_token,train[~train['weak_climate']]['sentence']))))

    f.total_count = {k: general_count.get(k, 0) + target_count.get(k, 0) for k in set(general_count)}
    # Can only concider words that are in the general list to avoid division by zero errors 
    normed_target = {k: target_count[k] if k in target_count.keys() else 0 for k in f.total_count.keys()}
    normed_general = {k: general_count[k] if k in general_count.keys() else 0 for k in f.total_count.keys()}
    words = set([k for k,v in f.total_count.items() if v >= max(min_count_parms)])
    ratio = {k: (normed_target[k])*sum(normed_general.values())/(normed_general[k])/sum(normed_target.values())  for k in words}
    f.norm = pd.Series({k: v/(v+1)  for k,v in ratio.items()})
    for min_count in min_count_parms:
        f.min_count = min_count
        words = set([k for k,v in f.total_count.items() if v >= min_count])
        f.norm = pd.Series({k: v/(v+1)  for k,v in ratio.items() if k in words })
        indecies = f.norm[f.norm.apply(lambda x: abs(x - 0.5)).sort_values(ascending = False).index].iloc[0:f.model_size,].index
        if len(indecies) > 0:
            f.model = f.norm.loc[indecies]
        else:
            f.model = f.norm
        for model_size in model_size_parms:
            f.model_size = model_size
            f.model = f.model.iloc[0:f.model_size]
            for i in range(len(test)):
                test[i]['prob'] = f.predict(test[i],return_prob=True, quiet=True)
            for threshold in threshold_parms:
                f.threshold = threshold
                test_report = []
                for i in range(len(test)):
                    test[i]['predicted'] = test[i]['prob'].apply(lambda x: x > threshold)
                    test_report.append(classification_report(test[i]['label'],test[i]['predicted']))
                out.append(pd.DataFrame([[min_count,model_size,threshold,len(train),test_report]], columns = ['min_count','model_size','threshold', 'training_size', 'test_report']))
    return out 

def evaluate_model(train_labeled, train_unlabeled, test, filter):
    embedding_parms = ['doc2vecdm','doc2vecdbow','tfidf','bow']
    author_info_parms = [True, False]
    predictive_model_parms = ['DecisionTree','SGDClassifier','Perceptron','RandomForestClassifier','LinearSVC','LogisticRegression', 'KNeighborsClassifier']
    weighting_parms = ['balanced', None]
    out = []
    m = Model()
    m.filter = filter
    train_unlabeled = train_unlabeled[filter.predict(train_unlabeled, quiet=True)]
    for embedding in embedding_parms:
        for author_info in author_info_parms:
            if author_info:
                m.embedding_scheme = Embedding(embedding,author_info)
                m.embedding_scheme.train(train_unlabeled)
            else:
                m.embedding_scheme.author_info = False
            train_labeled['vector'] = m.embedding_scheme.predict(train_labeled)
            for predictive_model in predictive_model_parms:
                for weights in weighting_parms:
                    m.predictive_model = Predictive_model(model = predictive_model, kwargs = {'class_weight':weights})
                    print(f'Training Model on {len(train_labeled)} with {embedding}, {predictive_model}, {author_info} author_info and {weights} weights')
                    m.predictive_model.train(train_labeled)
                    predicted = m.predict(test)['class']
                    out.append(pd.DataFrame([[embedding,author_info,predictive_model,weights, confusion_matrix(test['class'], predicted), classification_report(test['class'], predicted, output_dict = True)]]))
    return out
                    
                    
if __name__ == '__main__':

    ## Inititate dataset
    d = Dataset()
    f_val_10K = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:43546a2f-82d6-49a3-af54-69b02cff54a9/AL-10Ks.tsv%20:%203000%20(58%20positives,%202942%20negatives)%20(TSV,%20127138%20KB).tsv' , sep = '\t')
    f_val_wiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:9d139a47-878c-4d2c-b9a7-cbb982e284b9/AL-Wiki%20(train).tsv', sep = '\t')
    f_val_ALwiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:c4f6e427-6b84-41ca-a016-e66337fb283b/Wiki-Doc-Train.tsv', sep = '\t')
    f_test_10k = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:cf6dea3a-ca4f-422f-8f1c-e90d88dd56dd/10-Ks%20(2018,%20test).tsv', sep = '\t')
    f_test_wiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:8533e714-155f-49f2-b997-6b9873749303/Wikipedia%20(test).tsv', sep = '\t') 
    f_test_claims =  pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:d5e1ac74-0bf1-4d84-910f-7a9c7cd28764/Claims%20(test).tsv', sep = '\t')

    data = d.encode_labels(d.apply_labels(d.df_sentence))
    data['domain'] = d.domains(data)
    data['weak_climate'] =  data['parent'].isin(np.concatenate((d.df_climate.index,d.df_skeptics.index)))

    labeled_data, unlabeled_data = [ x for _, x in data.groupby(data['class'] == -1)]

    """
    What Questions do we want to answer?

    Basic Tuning:
     - What is the best filter
     - What is the best embedding scheme + model combo
     - Do weights inprove the model

    Is it worthwhile to gather more data:
     - Does the filter improve with more data?
     - Does the model improve with more unlabled_data
     - Does the model improve with more labeled_data
     - Does author_info improve the model?
     - Do I even need a filter -> is embedding scheme improved by more test overall
     - If inconclusive -> is there a tradeoff between the best filter and the best model

    since labeled training set has a large amount of weakly labeled_data, filter will always used to remove non climate senteces from this set

    """


    # Tuning Filter
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
    """ 
    filter_evaluation_dfs = [simple_starmap(evaluate_filter, [[unlabeled_data.sample(frac =2**(-x), random_state=1), [pd.concat([f_val_wiki, f_val_ALwiki, f_val_10K], ignore_index = True), f_test_wiki, f_test_10k, f_test_claims]] for x in range(0,15)])]
    filter_evaluation = pd.concat(flatten(flatten(filter_evaluation_dfs)))
    filter_evaluation['f1'] = filter_evaluation['test_report'].apply(lambda x: x[0]['1']['f1'])
    best_f1 = max(filter_evaluation['f1'])
    print(best_f1)

    """
    best filter is a compromise between the best f1 and the time it takes to clasify a set of documents
    time to clasify is determined by the model size
    therefore, if there exists some model which is significantly smaller than the best model with only a small sacrifice in performance, this will be the 'best model'
    """
    best_filter_parms = [x for _,x in filter_evaluation[filter_evaluation['']>=best_f1].groupby('model_size')][0].sort_values('f1', ascending=False).iloc[0]
    best_filter = Filter({
            'threshold':best_filter_parms['threshold'],
            'min_count':best_filter_parms['min_count'],
            'model_size':best_filter_parms['model_size'],
            'training_size':best_filter_parms['training_size']
        })
    best_training_data = data.sample(n=int(best_filter_parms['training_size']))
    best_filter.train(best_training_data[best_training_data['weak_climate']]['sentence'], best_training_data[~best_training_data['weak_climate']]['sentence'])
    
    raise ValueError
    #Evaluate the predictive_model
    """
    Now that the best filter has been determined, the best predicctive model can be found.
    To do this cross valisation will be used to tune four hyperparameters:

    embedding_scheme
    author_info
    predictive_model
    weights

    cross validation will be done using 4 folds, since this allows to allocate 4 cores of development machine
    test will be a sample of 0.1
    """
    train_labeled, test = train_test_split(labeled_data[best_filter.predict(labeled_data)], random_state=1)
    
    ### If we have less than 10 labels for a class we remove it
    ### Will be fixed by the active leaning scheme as these will not fit into nice categories therefore will have high entropy
    train_unlabeled = unlabeled_data[best_filter.predict(unlabeled_data)]

    train_labeled= train_labeled.groupby('class').filter(lambda x: len(x)>9 )
    splits = StratifiedShuffleSplit(n_splits=4, test_size=0.1)
    in_ls = []
    for train,test in splits.split(labeled_data, labeled_data['class']):
        in_ls.append([train_labeled.iloc[train],unlabeled_data,train_labeled.iloc[test] , best_filter])


    out = simple_starmap(evaluate_model, in_ls)
        

