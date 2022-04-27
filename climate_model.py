import numpy as np
from tqdm import tqdm
import pandas as pd
from dataset import Dataset
from multiprocessing import Pool, Manager
import gensim

"""

https://matpalm.com/semi_supervised_naive_bayes/semi_supervised_bayes.html


Model built using beys filtering

P(C|W) 

~= P(C|w1) * P(C|w2) * ... * P(C)

current sentences are tokenised as weak labels (c, !c , none) and doc length(int)

for a sentece in the dataset with label c and doc length n and parameter a:

    P(C|w  < 1/(n^1/a)
note that higher a indicates that more sentences per document will be about climate

parameters for this model -> 

 a -> controls number of senteces per traing label 
 b -> P(climate)
"""

def main(df,a):
    ## Do an initial claisifcation model using hard_climate labels
    known_df = df[df['true_climate'] == True]
    flat = [item for sublist in known_df['sentence'] for item in sublist]
    dic = {x:[] for x in set(flat)}
    tokens = known_df['sentence'].values
    probabilities =  df.apply(lambda x: convert_prob(x,a),axis = 1)
    for index, token_set in tqdm(enumerate(tokens), total=len(tokens)):
        for word in token_set:
            dic[word].append(probabilities[index])
    return dic


if __name__ == "__main__":
    d=Dataset(dev = True, download=False)
    dic = main(d.df,100000)
    neg_dic =  {}
    for key in dic.keys():
        neg_dic[key] = sum([ 1-x for x in dic[key]])
        dic[key] = sum(dic[key])
    sum_pro = sum(dic.values())
    sum_neg = sum(neg_dic.values())
    pr = {}
    for key in dic.keys():
        dic[key] = (dic[key] + 1)/(sum_pro+3)
        neg_dic[key] = (neg_dic[key]+1)/(sum_neg+3)
        pr[key] = dic[key]/(dic[key]+ neg_dic[key])
    
    
def find_sen(sen, pr):
    a = sum([pr[x] for x in gensim.utils.simple_preprocess(sen)])
    b = sum([1-pr[x] for x in gensim.utils.simple_preprocess(sen)])
    prob = a/(a+b)
    return prob

def convert_prob(row,a):
    if row['soft_climate'] == True:
        return 1/(row['doc_len']**1/a)
    elif row['soft_climate'] == False:""
