from time import sleep
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from urllib.request import urlopen
import dill
from utils import *
import pandas as pd
import newspaper
from tqdm import tqdm 
import os
import requests
import pandas as pd
from datetime import datetime,  timedelta
import pytz
import warnings
import uuid as uuid_mod


class Dataset():
    def __init__(self, from_date = datetime(2018,1,1), to_date = datetime(2022,5,31)):
        warnings.filterwarnings('ignore')
        try:
            self.load()
        except FileNotFoundError:

            os.makedirs('./picklejar', exist_ok=True)
            self.timestamps=[]
            self.df_climate = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score','comments'])
            self.df_general = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score','comments'])
            self.df_skeptics = pd.DataFrame(columns= ['author','timestamp','post_url','media_url','score','comments'])
            self.df_sentence = pd.DataFrame(columns=['parent', 'sentence'])

            resp = urlopen("https://drive.google.com/uc?export=download&id=1QXWbovm42oDDDs4xhxiV6taLMKpXBFTS")
            df = pd.read_json(BytesIO(resp.read()), compression='zip')

            df.index = df['ParagraphId'].apply(lambda x: UUID(int=x))
            self.df_seed = df

            df['sentence'] = simple_map(sent_token,df['Paragraph_Text'], 'Tokenising Seed Data') 
            df['new_ind'] = df['sentence'].apply(lambda x: [i for i in range(len(x))])
            df['parent'] = df['ParagraphId'].apply(lambda x: UUID(int=x))
            df= df.explode(['sentence','new_ind'])
            df['uuid'] = df.apply(lambda x: uuid(x[['new_ind','parent']]), axis =1)
            df.set_index(['uuid'], inplace=True)

            self.df_sentence = pd.concat([self.df_sentence, df[['parent', 'sentence']]])

        utc = pytz.UTC
        day = utc.localize(from_date)
        new_timestamps = []
        while day < utc.localize(to_date):
            if day.timestamp() in self.timestamps:
                day += timedelta(days =1)
            else:
                self.timestamps.append(day.timestamp())
                day += timedelta(days =1)
                new_timestamps.append(day.timestamp())
                

        if len(new_timestamps):
            self.reload = True
            print('Fetching for ' + ', '.join([datetime.fromtimestamp(x).strftime('%Y-%m-%d') for x in new_timestamps]))

            climate_urls = flatten([    [self.get_links(date,'climate') for date in tqdm(new_timestamps,total=len(new_timestamps))],
                                        [self.get_links(date,'environment') for date in tqdm(new_timestamps,total=len(new_timestamps))]
                                        ])
            general_urls = flatten([    [self.get_links(date,'news') for date in tqdm(new_timestamps,total=len(new_timestamps))],
                                        [self.get_links(date,'worldnews') for date in tqdm(new_timestamps,total=len(new_timestamps))]
                                        ])
            climateskeptics_urls =  [self.get_links(date,'climateskeptics') for date in tqdm(new_timestamps,total=len(new_timestamps))]

            df_climate = pd.DataFrame(flatten(climate_urls),columns= ['author','timestamp','post_url','media_url','score','comments','uuid'])
            df_general = pd.DataFrame(flatten(general_urls),columns= ['author','timestamp','post_url','media_url','score','comments','uuid'])
            df_skeptics = pd.DataFrame(flatten(climateskeptics_urls),columns= ['author','timestamp','post_url','media_url','score','comments','uuid'])


            df_climate['article'] = simple_map(self.get_articles, df_climate['media_url'], 'Fetching Climate Articles')
            df_general['article'] = simple_map(self.get_articles, df_general['media_url'], 'Fetching General Articles')
            df_skeptics['article'] = simple_map(self.get_articles, df_skeptics['media_url'], 'Fetching Skeptics Articles')


            df_climate['uuid'] = simple_map(uuid, df_climate['post_url'])
            df_general['uuid'] = simple_map(uuid, df_general['post_url'])
            df_skeptics['uuid'] = simple_map(uuid, df_skeptics['post_url'])

            df_climate.set_index(['uuid'], inplace=True)
            df_general.set_index(['uuid'], inplace=True)
            df_skeptics.set_index(['uuid'], inplace=True)

            self.df_climate = pd.concat([self.df_climate,df_climate])
            self.df_general = pd.concat([self.df_general,df_general])
            self.df_skeptics = pd.concat([self.df_skeptics,df_skeptics])

            df_sentence = pd.concat([df_climate,df_skeptics,df_general])
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
        url = 'https://api.pushshift.io/reddit/search/submission/?subreddit='+ subreddit +'&sort=asc&sort_type=num_comments&after=' + str(int(date_int)) + '&before='+str(int(date_int + 86400)) +'&size=5'
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
            except:
                return None


    def save(self):
        print('Pickling Dataset')
        with open(os.path.join('picklejar','dataset.pickle'), 'wb') as f:
            dill.dump(self.__dict__,f,2)


    def load(self):
        print('Unpickling Dataset')
        with open(os.path.join('picklejar','dataset.pickle'), 'rb') as f:
            tmp_dic = dill.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dic)

    

    def apply_labels(self, df):
        if os.path.exists('labels.csv'):
            self.df_labels = pd.read_csv('labels.csv', index_col=0)
            self.df_labels.index = [uuid_mod.UUID(x) for x in self.df_labels.index]
            self.df_labels =  self.df_labels[~self.df_labels.index.duplicated()]
        else:
            self.df_labels = pd.DataFrame(columns=['sub_sub_claim', 'timestamp'])
        df['sub_sub_claim'] = None
        
        df.loc[df['sub_sub_claim'].isna(),'sub_sub_claim'] = df[df['sub_sub_claim'].isna()].join(self.df_labels, how = 'left', rsuffix = '_y')['sub_sub_claim_y']
        
        df.loc[df['sub_sub_claim'].isna(),'sub_sub_claim'] = df[df['sub_sub_claim'].isna()].merge(self.df_seed,left_on = 'parent', right_index = True, how = 'left')['sub_sub_claim_y']


        return df


    def get_labels(self, df, n=10):
        df = df.sort_values(['entropy'], ascending = False).iloc[:n]
        for index, row in df.iterrows():

            label = input(str(index) +': ' +str(row['entropy'])+ '\n'+ row['sentence'] + '\n')
            self.df_labels.loc[index] = [label, datetime.now()]
            self.df_labels.to_csv('labels.csv')

    def domains(self, df):
        domains = df.merge(pd.concat([self.df_general, self.df_climate, self.df_skeptics]), right_index = True, left_on = 'parent', how = 'left')['media_url']
        domains.loc[~domains.isna()] = domains.loc[~domains.isna()].apply(lambda x: urlparse(x).netloc)
        return(domains)
        


if __name__ == '__main__':
    d = Dataset()

