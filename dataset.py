import pickle
from nltk import text
import pandas as pd
import newspaper
import random
import numpy as np
from zipfile import ZipFile
import nltk
from newspaper import Article
from pandas.core.frame import DataFrame
from requests.models import Response
import wget
from tqdm import tqdm 
import os
import shutil
import gensim
from multiprocessing import Manager, Pool
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
import time
import re
import collections
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from itertools import repeat
import pytz


class Dataset():
    def __init__(self, download = True, rebuild = False, save = True, state = 1 , dev = False):
        try:
            self.load()
        except FileNotFoundError:
            self.timestamps=[]
            self.df_climate = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score'])
            self.df_news = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score'])
            self.df_skeptics = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score'])

        utc = pytz.UTC
        day = utc.localize(datetime(2022,4,25))
        new_timestamps = []
        while day < utc.localize(datetime.utcnow()-timedelta(days = 1)):
            if day.timestamp() in self.timestamps:
                day += timedelta(days =1)
            else:
                self.timestamps.append(day.timestamp())
                day += timedelta(days =1)
                new_timestamps.append(day.timestamp())
                

        if len(new_timestamps):
            print('Fetching for ' + ', '.join([datetime.fromtimestamp(x).strftime('%Y-%m-%d') for x in new_timestamps]))

            climate_urls = simple_starmap(self.get_links, [(date,'climate') for date in new_timestamps])
            news_urls = simple_starmap(self.get_links, [(date,'news') for date in new_timestamps])
            climateskeptics_urls = simple_starmap(self.get_links, [(date,'climateskeptics') for date in new_timestamps])
            self.df_climate = pd.concat([self.df_climate,pd.DataFrame(flatten(climate_urls))])
            self.df_news = pd.concat([self.df_climate,pd.DataFrame(flatten(news_urls))])
            self.df_skeptics = pd.concat([self.df_climate,pd.DataFrame(flatten(climateskeptics_urls))])
            self.save()


        
    def get_links(self,date_int, subreddit):
        url = 'https://api.pushshift.io/reddit/search/submission/?subreddit='+ subreddit +'&sort=desc&sort_type=score&after=' + str(int(date_int)) + '&before='+str(int(date_int + 86400)) +'&size=10000'
        res = None
        while res != 200:
            r = requests.get(url)
            res = r.status_code
        data = r.json()['data']
        return [{'author':x['author'],'timestamp': x['created_utc'],'post_url':x['full_link'], 'media_url':x['url'], 'score':x['score']} for x in data]


    def get_articles(self,url):
        try:
            a = newspaper.Article(url)
            a.download()
            a.parse()
            return a
        except newspaper.article.ArticleException:
            None


    def save(self):
        print('Pickling')
        with open(os.path.join('picklejar','dataset.pickle'), 'wb') as f:
            pickle.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling')
        with open(os.path.join('picklejar','dataset.pickle'), 'rb') as f:
            tmp_dic = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dic)


def sent_token(doc):
    if not isinstance(doc,str):
        doc = ""
    return nltk.tokenize.sent_tokenize(doc)

def word_token(sentence):
    return gensim.utils.simple_preprocess(sentence)

def flatten(ls):
    return [item for sublist in ls for item in sublist]

def simple_starmap(func, ls):
    pool = Pool()
    out = pool.starmap(func,ls)
    pool.close()
    pool.join()
    return out

if __name__ == '__main__':
    d = Dataset(dev=True, download=False)
