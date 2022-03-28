import pandas as pd
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
            self.df = pd.read_pickle(os.path.join(self.dir,'data','dataset.b2z'))
        except FileNotFoundError:
            self.df = pd.DataFrame(columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']) 
            #self.download()
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
        pbar = tqdm(total=21, desc = 'Building Jobs', miniters = 1)
        with Manager() as manager:
            L = manager.list()

            data = []

            df = pd.read_csv('climate-news-db-dataset.csv')
            data = data + [(L,sdf,1,self.subprocess1) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            pbar.update(n=1)
            
            for year in range(2015,2021):
                df = pd.read_csv(f'WebNewsEnglishSnippets.{year}.csv',header = None)
                data = data + [(L,sdf,2+(year-2015)*0.1,self.subprocess2) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            pbar.update(n=2)


            df = pd.read_csv('Wiki-Doc-Train.tsv', sep='\t')
            data = data + [(L,sdf,3,self.subprocess3) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            pbar.update(n=3)

            for file in os.listdir('./TelevisionNews'):
                try:
                    sdf=pd.read_csv(f'./TelevisionNews/{file}')
                    
                except:
                    pass
                data = data + [(L,sdf,4,self.subprocess4)]
            pbar.update(n=4)

            df =  pd.read_json('cards_training_sub_sub_claim.json') 
            data = data + [(L,sdf,5,self.subprocess5) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            pbar.update(n=5)


            df = pd.read_csv('Climate_change_allyears_trim.csv')
            data = data + [(L,sdf,6,self.subprocess6) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            pbar.update(n=6)

            for sub_ds_id,file in enumerate(os.listdir('./data/training')):
                ds_id = 7 + 0.1*sub_ds_id
                df = pd.read_csv(f'./data/training/{file}')
                data = data + [(L,sdf,ds_id,self.subprocess7) for sdf in np.split(df, np.arange(1,round(len(df)/2500))*5000)]
            pbar.update(n=7)

            pool=Pool()
            for _ in tqdm(pool.istarmap(self.subprocess, data), total=len(data), miniters = 1, desc='Building Dataset'):
                pass
            pool.join()

            self.df = pd.concat(L)
            
            self.df['sentence'] = self.df['sentence'].str.replace(r'[^\w\s]+','')
    

    def save(self):
        print('Pickling')
        self.df.to_pickle(os.path.join(self.dir,'data','dataset.bz2'),index=False)




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
                        l.append(pd.DataFrame([[ds_id,index,url,domain,publisher,date,None,title,f,True,None,None,None,None]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))
                    except FileNotFoundError:
                        pass

    def subprocess2(self,l,df,ds_id):
            # WebNewsEnglishSnippets
            for index, article in df.iterrows():
                    url = article[3]
                    title = article[1]
                    date = article[0]
                    sentence = str(article[4]).split('.')
                    l.append(pd.DataFrame([[ds_id,index,url,None,None,date,None,title,sentence,True,None,None,None,None]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))


    def subprocess3(self,l,df,ds_id):
            # sustainablefinance
            domain = 'https://en.wikipedia.org/'
            publisher = 'wikipedia'
            for index, article in df.iterrows():
                    title = article['title']
                    sentence = article['sentence']
                    if article['label'] == 1:
                        climate = True
                    elif article['label'] == 0:
                        climate = False
                    elif article['label'] == -1:
                        climate = None
                    l.append(pd.DataFrame([[ds_id,index,None,domain,publisher,None,None,title,sentence,climate,None,None,None,None]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))

    def subprocess4(self,l,df,ds_id):
            #TelevisionNews
                for index, article in df.iterrows():
                    url = article['URL']
                    date = article['MatchDateTime']
                    publisher = article['Station']
                    f = str(article['Snippet']).split('.')
                    l.append(pd.DataFrame([[ds_id,file,url,None,publisher,date,None,None,f,True,None,None,None,None]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))

    def subprocess5(self,l,df,ds_id):
            #Labeled by trav
            for index, article in df.iterrows():
                    claim = str(article['sub_claim'])[0]
                    sub = article['sub_claim']
                    subsub = article['sub_sub_claim']
                    f = str(article['Paragraph_Text']).split('.')
                    if claim == str(0):
                        binary = False
                    else:
                        binary = True
                    l.append(pd.DataFrame([[ds_id,index,None,None,None,None,None,None,f,True,binary,claim,sub,subsub]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))

    def subprocess6(self,l,df,ds_id):
            #Miriams dataset
            ds_id = 6
            df = pd.read_csv('Climate_change_allyears_trim.csv')
            for index, article in df.iterrows():
                    title = article['Headline']
                    date = article['Date']
                    publisher = article['LP']
                    f = str(article['Paragraph']).split('.')
                    l.append(pd.DataFrame([[ds_id,index,None,None,publisher,date,None,title,f,True,None,None,None,None]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))

    def subprocess7(self,l,df,ds_id):
        #Public Cards dataset
        for index, article in df.iterrows():
                claim = article['claim'][0]
                sub = article['claim'].replace('_','.')
                if claim == str(0):
                    binary = False
                else:
                    binary = True
                f = str(article['text']).split('.')
                l.append(pd.DataFrame([[ds_id,index,None,None,None,None,None,None,f,True,binary,claim,sub,None]], columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']).explode('sentence'))

d = Dataset()

