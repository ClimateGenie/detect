import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from dataset import Dataset

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
    tokens = df['sentence'].values
    probabilities =  df.apply(lambda x: 1/(x['doc_len']**1/a),axis = 1)
    dic = []
    num_pro = []
    num_neg = []
    for index, token_set in enumerate(tokens):
        print(token_set)
        for word in token_set:
            if word in dic:
                num_pro[dic.index(word)] += probabilities[index] 
                num_neg[dic.index(word)] += 1-probabilities[index] 


if __name__ == "__main__":

    d=Dataset()

    main(d.df,1)
    
    d.df.apply()

    d.df['doc_len']

