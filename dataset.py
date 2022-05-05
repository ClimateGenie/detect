from time import sleep
import dill
from utils import *
import numpy as np
import pandas as pd
import newspaper
from tqdm import tqdm 
from multiprocessing import Pool
from itertools import starmap
import os
import requests
import pandas as pd
from datetime import datetime, time, timedelta, date
import pytz
import warnings


class Dataset():
    def __init__(self, from_date = datetime.combine(date.today(), datetime.min.time())- timedelta(days=3)):
        warnings.filterwarnings('ignore')
        try:
            self.load()
        except FileNotFoundError:
            self.timestamps=[]
            self.df_climate = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score','comments'])
            self.df_news = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score','comments'])
            self.df_skeptics = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score','comments'])
            self.df_sentence = pd.DataFrame(columns=['parent', 'sentence'])

        utc = pytz.UTC
        day = utc.localize(from_date)
        new_timestamps = []
        while day < utc.localize(datetime.utcnow()-timedelta(days = 2)):
            if day.timestamp() in self.timestamps:
                day += timedelta(days =1)
            else:
                self.timestamps.append(day.timestamp())
                day += timedelta(days =1)
                new_timestamps.append(day.timestamp())
                

        if len(new_timestamps):
            print('Fetching for ' + ', '.join([datetime.fromtimestamp(x).strftime('%Y-%m-%d') for x in new_timestamps]))

            climate_urls = [self.get_links(date,'climate') for date in tqdm(new_timestamps,total=len(new_timestamps))]
            news_urls = flatten([ [self.get_links(date,'news') for date in tqdm(new_timestamps,total=len(new_timestamps))],  [self.get_links(date,'worldnews') for date in tqdm(new_timestamps,total=len(new_timestamps))]   ])
            climateskeptics_urls =  [self.get_links(date,'climateskeptics') for date in tqdm(new_timestamps,total=len(new_timestamps))]

            df_climate = pd.DataFrame(flatten(climate_urls),columns= ['author','timestamp','post_url','media_url','score','comments','uuid'])
            df_news = pd.DataFrame(flatten(news_urls),columns= ['author','timestamp','post_url','media_url','score','comments','uuid'])
            df_skeptics = pd.DataFrame(flatten(climateskeptics_urls),columns= ['author','timestamp','post_url','media_url','score','comments','uuid'])


            df_climate['article'] = simple_map(self.get_articles, df_climate['media_url'])
            df_news['article'] = simple_map(self.get_articles, df_news['media_url'])
            df_skeptics['article'] = simple_map(self.get_articles, df_skeptics['media_url'])


            df_climate['uuid'] = simple_map(uuid, df_climate['post_url'])
            df_news['uuid'] = simple_map(uuid, df_news['post_url'])
            df_skeptics['uuid'] = simple_map(uuid, df_skeptics['post_url'])

            df_climate.set_index(['uuid'], inplace=True)
            df_news.set_index(['uuid'], inplace=True)
            df_skeptics.set_index(['uuid'], inplace=True)

            self.df_climate = pd.concat([self.df_climate,df_climate])
            self.df_news = pd.concat([self.df_news,df_news])
            self.df_skeptics = pd.concat([self.df_skeptics,df_skeptics])

            df_sentence = pd.concat([df_climate,df_skeptics,df_news])
            df_sentence = df_sentence[df_sentence['article'] != None]
            df_sentence['sentence'] = df_sentence['article'].apply(lambda x: sent_token(x))
            df_sentence['new_ind'] = df_sentence['sentence'].apply(lambda x: [i for i in range(len(x))])
            df_sentence['parent'] = df_sentence.index
            df_sentence = df_sentence.explode(['sentence','new_ind'])
            df_sentence['uuid'] = df_sentence.apply(lambda x: uuid(x[['new_ind','parent']]), axis =1)
            df_sentence.set_index(['uuid'], inplace=True)
            self.df_sentence = pd.concat([self.df_sentence,df_sentence[['parent','sentence']]])

            

            self.save()


        
    def get_links(self,date_int, subreddit):
        url = 'https://api.pushshift.io/reddit/search/submission/?subreddit='+ subreddit +'&sort=asc&sort_type=num_comments&after=' + str(int(date_int)) + '&before='+str(int(date_int + 86400)) +'&size=50'
        res = None
        while res != 200:
            r = requests.get(url)
            res = r.status_code
            sleep(1)
        data = r.json()['data']

        return [{'author':x['author'],'timestamp': x['created_utc'],'post_url':x['full_link'], 'media_url':x['url'], 'score':x['score'], 'comments':x['num_comments']} for x in data]


    def get_articles(self,url):
        if not any(sub in url for sub in ['reddit', 'youtube'] ):
            try:
                a = newspaper.Article(url)
                a.download()
                a.parse()
                return a.text
            except newspaper.article.ArticleException:
                return None


    def save(self):
        print('Pickling')
        with open(os.path.join('picklejar','dataset.pickle'), 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling')
        with open(os.path.join('picklejar','dataset.pickle'), 'rb') as f:
            tmp_dic = dill.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dic)

    def vectorise(self, embedding_scheme):
        self.df_filtered['vector'] = simple_map(embedding_scheme.vectorise, self.df_filtered['sentence'])
        return self.df_filtered

    
    def climate_words(self):
        return flatten(flatten([simple_map(word_token,self.df_climate['article']), simple_map(word_token, self.df_skeptics['article'])]))


    def news_words(self):
        return flatten(simple_map(word_token,self.df_news['article']))


    def filter_for_climate(self, filter, threshold = 0.9):
        df = self.df_sentence.copy()
        df['prob'] = simple_map(filter.prob, df['sentence'])
        self.df_filtered = df[df['prob']>threshold]
        return self.df_filtered

        

if __name__ == '__main__':
    d = Dataset()
