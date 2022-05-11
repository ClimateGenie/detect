from time import sleep
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from scipy.stats.distributions import entropy
import dill
from pandas.core.algorithms import isin
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
import uuid as uuid_mod


class Dataset():
    def __init__(self, from_date = datetime.combine(date.today(), datetime.min.time())- timedelta(weeks = 4)):
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
            self.reload = True
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
        else:
            self.reload = False


        
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



    def save_filtered(self):
        self.df_filtered.to_pickle('./picklejar/df_filtered')

    def load_filtered(self):
        self.df_filtered = pd.read_pickle('./picklejar/df_filtered')

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
        self.df_filtered['vector'] = embedding_scheme.model.dv[self.df_filtered.index].tolist()


    
    def climate_words(self):
        return flatten(flatten([simple_map(word_token,self.df_climate['article']), simple_map(word_token, self.df_skeptics['article'])]))


    def news_words(self):
        return flatten(simple_map(word_token,self.df_news['article']))


    def filter_for_climate(self, filter, threshold = 0.9):
        self.threshold = threshold
        df = self.df_sentence.copy()
        df['prob'] = simple_map(filter.prob, df['sentence'])
        self.df_filtered = df[df['prob']>threshold]
        return self.df_filtered

    def apply_labels(self):
        if os.path.exists('labels.csv'):
            self.df_labels = pd.read_csv('labels.csv', index_col=0)
            self.df_labels.index = [uuid_mod.UUID(x) for x in self.df_labels.index]
        else:
            self.df_labels = pd.DataFrame(columns=['sub_sub_claim'])
            self.df_filtered['predicted'] =  [[0,0]] * len(self.df_filtered)
            self.get_labels()
            self.apply_labels()
        self.df_filtered[self.df_filtered['sub_sub_claim'].isna()]['sub_sub_claim'] = self.df_filtered[self.df_filtered['sub_sub_claim'].isna()].join(self.df_labels, how = 'left', lsuffix='_left', rsuffix='_right')['sub_sub_claim_right']
        self.df_filtered.loc[self.df_filtered['sub_sub_claim'].isna(), 'sub_sub_claim'] = -1

    def add_seed_data(self, embedding_scheme, filter):
        resp = urlopen("https://drive.google.com/uc?export=download&id=1QXWbovm42oDDDs4xhxiV6taLMKpXBFTS")
        df = pd.read_json(BytesIO(resp.read()), compression='zip')
        df['sentence'] = simple_map(sent_token,df['Paragraph_Text'])
        df = df.explode('sentence')
        df['prob'] = simple_map(filter.prob, df['sentence'])
        df= df[df['prob']>self.threshold]
        df['vector'] = simple_map(embedding_scheme.vectorise, df['sentence'])
        df['parent'] = None
        df.index = simple_map(uuid, [str(x) for x in df.index])
        self.df_filtered = pd.concat([self.df_filtered, df[['parent', 'sentence', 'prob', 'vector','sub_sub_claim']]])


    def encode_labels(self):
        self.encoder = LabelEncoder()
        self.df_filtered.loc[self.df_filtered['sub_sub_claim'] != -1, 'class'] = self.encoder.fit_transform(self.df_filtered.loc[self.df_filtered['sub_sub_claim'] != -1, 'sub_sub_claim'])        
        self.df_filtered.loc[self.df_filtered['class'].isna(), 'class']  = -1        

    def predict_unlabeled(self, model):
        labels = model.model.label_distributions_
        self.df_filtered['predicted'] = [*labels]

    def get_labels(self, n=10):
        self.df_filtered['entropy'] = self.df_filtered['predicted'].apply(lambda x: entropy(x))
        to_label = self.df_filtered[self.df_filtered['class'] != -1].sort_values(['entropy']).iloc[:n]
        print(to_label)
        for index, row in to_label.iterrows():
            label = input(row['sentence'] + '\n')
            self.df_labels.loc[index] = [label]
        self.df_labels.to_csv('labels.csv')
        


if __name__ == '__main__':
    d = Dataset()
