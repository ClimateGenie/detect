import pandas as pd
import random
import numpy as np
import json
from zipfile import ZipFile
import newspaper
import wget
from tqdm import tqdm 
import os
import shutil
from multiprocessing import Manager, Pool
import pickle

# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


class Dataset():
    print('Fetching Dataset')
    def __init__(self):
        self.sources = ['https://www.climate-news-db.com/download',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2015.csv.zip',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2016.csv.zip',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2017.csv.zip',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2018.csv.zip',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2019.csv.zip',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2020.csv.zip',
                        'https://www.sustainablefinance.uzh.ch/dam/jcr:c4f6e427-6b84-41ca-a016-e66337fb283b/Wiki-Doc-Train.tsv',
                        'http://data.gdeltproject.org/blog/2020-climate-change-narrative/TelevisionNews.zip',
                        'https://drive.google.com/uc?export=download&id=1QXWbovm42oDDDs4xhxiV6taLMKpXBFTS',
                        'https://drive.google.com/uc?export=download&id=1exilBtj1fnryC7xkoHjZ6YZPaRZHBtbP',
                        'http://socialanalytics.ex.ac.uk/cards/data.zip'
                        ]
        self.dir = os.path.dirname(os.path.abspath(__file__))
        try:
            self.df = pd.read_pickle(os.path.join(self.dir,'data','dataset.pickle'))
        except FileNotFoundError:
            print('Dataset not found')
            self.df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']) 
            self.download()
            self.process()
            self.save()

    def download(self):
        try:
            os.mkdir(os.path.join(self.dir,'data'))
        except FileExistsError:
            pass

        try:
            shutil.rmtree(os.path.join(self.dir,'data','raw'))
        except FileNotFoundError:
            pass
        os.mkdir(os.path.join(self.dir,'data','raw'))
        os.chdir(os.path.join(self.dir,'data','raw'))
        for url in self.sources:
            res = None
            while res is None:
                try:
                    print(f' \nDownloading {url}')
                    wget.download(url)
                    res = 1
                except:
                    pass
        for file in os.listdir():
            if '.zip' in file:
                with ZipFile(file, 'r') as zipObj:
                          zipObj.extractall()
                os.remove(file)


    def process(self):

        os.chdir(os.path.join(self.dir,'data','raw'))
        print("Building Process Pool")
        with Manager() as manager:
            L = manager.list()

            data = []

            df = pd.read_csv('climate-news-db-dataset.csv')
            data = data + [(L,sdf,1,self.subprocess1) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            
            for year in range(2015,2021):
                df = pd.read_csv(f'WebNewsEnglishSnippets.{year}.csv',header = None)
                data = data + [(L,sdf,2+(year-2015)*0.1,self.subprocess2) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]


            df = pd.read_csv('Wiki-Doc-Train.tsv', sep='\t')
            data = data + [(L,sdf,3,self.subprocess3) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]

            for file in os.listdir('./TelevisionNews'):
                try:
                    sdf=pd.read_csv(f'./TelevisionNews/{file}')
                    
                except:
                    pass
                data = data + [(L,sdf,4,self.subprocess4)]

            df =  pd.read_json('cards_training_sub_sub_claim.json') 
            data = data + [(L,sdf,5,self.subprocess5) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]


            df = pd.read_csv('Climate_change_allyears_trim.csv')
            data = data + [(L,sdf,6,self.subprocess6) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]

            for sub_ds_id,file in enumerate(os.listdir('./data/training')):
                ds_id = 7 + 0.1*sub_ds_id
                df = pd.read_csv(f'./data/training/{file}')
                data = data + [(L,sdf,ds_id,self.subprocess7) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]

            pool=Pool()
            random.shuffle(data)
            for _ in tqdm(pool.istarmap(self.subprocess, data), total=len(data), miniters = 10, desc='Building Dataset'):
                pass
            pool.close()
            pool.join()


            print("Building DataFrame")
            self.df = pd.concat(L)
            

            print('Cleaning Feilds')
            self.df['sentence'] = self.df['sentence'].str.replace(r'[^\w\s]+','', regex=True)
    

    def save(self):
        print('Pickling')
        self.df.to_pickle(os.path.join(self.dir,'data','dataset.pickle'))




    def subprocess(self,l,df,ds_id,target):
        target(l,df,ds_id)

    def subprocess1(self,l,df,ds_id):
        # Climate-news-dbo
        # A collection of climate news
        for index, article in df.iterrows():
                url = article['article_url']
                domain = article['newspaper_url']
                publisher = article['newspaper']
                date = article['date_published']
                title = article['headline']
                try:
                    f = open("./article_body/" + article['article_id']+ ".txt", "r").read().split('.')
                    l.append(pd.DataFrame([[ds_id,url,domain,publisher,date,None,title,f,True,None,None,None,None]], columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))
                except FileNotFoundError:
                    pass

    def subprocess2(self,l,df,ds_id):
        # WebNewsEnglishSnippets
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub'])
        target_df['url'] = df[3]
        target_df['title'] = df[1]
        target_df['date'] = df[0]
        target_df['sentence'] = df[4].str.split('.')
        target_df['climate'] = True
        target_df['ds_id'] = ds_id
        l.append(target_df.explode('sentence'))


    def subprocess3(self,l,df,ds_id):
        # sustainablefinance
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub'])
        target_df['title'] = df['title']
        target_df['domain'] = 'https://en.wikipedia.org/'
        target_df['publisher'] = 'wikipedia'
        target_df['climate'] = df['label'].apply(lambda x : bool(x))
        target_df['sentence'] = df['sentence'].str.split('.')
        target_df['ds_id'] = ds_id
        l.append(target_df.explode('sentence'))

    def subprocess4(self,l,df,ds_id):
        #TelevisionNews
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub'])
        target_df['url'] = df['URL']
        target_df['date'] = df['MatchDateTime']
        target_df['publisher'] = df['Station']
        target_df['climate'] = True
        target_df['sentence'] = df['Snippet'].str.split('.')
        target_df['ds_id'] = ds_id
        l.append(target_df.explode('sentence'))


    def subprocess5(self,l,df,ds_id):
        #Labeled by trav
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub'])
        target_df['claim']=df['sub_claim'].apply(lambda x: str(x)[0])
        target_df['sub']=df['sub_claim'].apply(lambda x: str(x)[-1])
        target_df['subsub']=df['sub_sub_claim'].apply(lambda x: str(x)[-1])
        target_df['binary'] = df['sub_claim'].apply(lambda x : True if str(x)[0] != '0' else False)
        target_df['sentence'] = df['Paragraph_Text'].str.split('.')
        target_df['climate'] = True
        target_df['ds_id'] = ds_id
        l.append(target_df.explode('sentence'))


    def subprocess6(self,l,df,ds_id):
        #Miriams dataset
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub'])
        target_df['title'] = df['Headline']
        target_df['date'] = df['Date']
        target_df['publisher'] = df['LP']
        target_df['sentence'] = df['Parargraph'].str.split('.')
        target_df['climate'] = True
        target_df['ds_id'] = ds_id
        l.append(target_df.explode('sentence'))

    def subprocess7(self,l,df,ds_id):
        #Public Cards dataset
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub'])
        target_df['claim']=df['claim'].apply(lambda x: str(x)[0])
        target_df['sub']=df['claim'].apply(lambda x: str(x)[-1])
        target_df['binary'] = df['claim'].apply(lambda x : True if str(x)[0] != '0' else False)
        target_df['climate'] = True
        target_df['ds_id'] = ds_id
        l.append(target_df.explode('sentence'))

    def subset(self, cols):
        sub_df = self.df[cols].dropna()

        return sub_df

if __name__ == "__main__":
    d = Dataset()

