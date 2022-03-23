import pandas as pd
import json
from zipfile import ZipFile
import newspaper
import wget
from tqdm import tqdm 
import os
import shutil


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
            self.df = pd.read_csv(os.path.join(self.dir,'data','dataset.csv'))
        except FileExistsError:
            self.df = pd.DataFrame(columns = ['ds_id','ds_index','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']) 

    def download(self):
        try:
            os.mkdir(os.path.join(self.dir,'data'))
        except FileExistsError:
            pass

        try:
            shutil.rmtree(os.path.join(self.dir,'data','raw'))
        except FileExistsError:
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
        try:
            # Climate-news-dbo
            # A collection of climate news
            ds_id = 1
            df = pd.read_csv('climate-news-db-dataset.csv') 
            for index, article in tqdm(df.iterrows(),desc = 'Climate-news-dbo', total  = len(df), miniters=1):
                if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == index)]): 
                    url = article['article_url']
                    domain = article['newspaper_url']
                    publisher = article['newspaper']
                    date = article['date_published']
                    title = article['headline']
                    try:
                        f = open("./article_body/" + article['article_id']+ ".txt", "r").read().split('.')
                        for sentence in f:
                            self.df.loc[len(self.df)] = [ds_id,index,url,domain,publisher,date,None,title,sentence,True,None,None,None,None]
                    except FileNotFoundError:
                        pass



            # WebNewsEnglishSnippets
            for sub_ds_id,file in enumerate(['WebNewsEnglishSnippets.2015.csv',
                        'WebNewsEnglishSnippets.2016.csv',
                        'WebNewsEnglishSnippets.2017.csv',
                        'WebNewsEnglishSnippets.2018.csv',
                        'WebNewsEnglishSnippets.2019.csv',
                        'WebNewsEnglishSnippets.2020.csv']):
                ds_id = 2 + 0.1*sub_ds_id
                df = pd.read_csv(file,header = None)
                for index, article in tqdm(df.iterrows(),desc = file, total  = len(df), miniters=1):
                    if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == index)]):
                        url = article[3]
                        title = article[1]
                        date = article[0]
                        f = str(article[4]).split('.')
                        for sentence in f:
                            self.df.loc[len(self.df)] = [ds_id,index,url,None,None,date,None,title,sentence,True,None,None,None,None]

            # sustainablefinance
            ds_id = 3
            domain = 'https://en.wikipedia.org/'
            publisher = 'wikipedia'
            df = pd.read_csv('Wiki-Doc-Train.tsv', sep='\t')

            for index, article in tqdm(df.iterrows(),desc = 'Climatetext', total  = len(df), miniters=1):
                if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == index)]):
                    title = article['title']
                    sentence = article['sentence']
                    if article['label'] == 1:
                        climate = True
                    elif article['label'] == 0:
                        climate = False
                    elif article['label'] == -1:
                        climate = None
                    self.df.loc[len(self.df)] = [ds_id,index,None,domain,publisher,None,None,title,sentence,climate,None,None,None,None]

            #TelevisionNews
            ds_id = 4
            for file in tqdm(os.listdir('./TelevisionNews'), desc= 'TelevisionNews', total  = len(os.listdir('./TelevisionNews')), miniters=1):
                if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == file)]):
                    df=pd.read_csv(f'./TelevisionNews/{file}')
                    for index, article in df.iterrows():
                        url = article['URL']
                        date = article['MatchDateTime']
                        publisher = article['Station']
                        f = str(article['Snippet']).split('.')
                        for sentence in f:
                            self.df.loc[len(self.df)] = [ds_id,file,url,None,publisher,date,None,None,sentence,True,None,None,None,None]

            #Labeled by trav
            ds_id = 5
            df =  pd.read_json('cards_training_sub_sub_claim.json')
            for index, article in tqdm(df.iterrows(),desc = 'Subsubclaim', total  = len(df), miniters=1):
                if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == index)]):
                    claim = str(article['sub_claim'])[0]
                    sub = article['sub_claim']
                    subsub = article['sub_sub_claim']
                    f = str(article['Paragraph_Text']).split('.')
                    if claim == str(0):
                        binary = False
                    else:
                        binary = True
                    for sentence in f:
                        self.df.loc[len(self.df)] = [ds_id,index,None,None,None,None,None,None,sentence,True,binary,claim,sub,subsub]

            #Miriams dataset
            ds_id = 6
            df = pd.read_csv('Climate_change_allyears_trim.csv')
            for index, article in tqdm(df.iterrows(),desc = 'Miriams set', total  = len(df), miniters=1):
                if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == index)]):
                    title = article['Headline']
                    date = article['Date']
                    publisher = article['LP']
                    f = str(article['Paragraph']).split('.')
                    for sentence in f:
                        self.df.loc[len(self.df)] = [ds_id,index,None,None,publisher,date,None,title,sentence,True,None,None,None,None]

            #Public Cards dataset
            for sub_ds_id,file in enumerate(os.listdir('./data/training')):
                ds_id = 7 + 0.1*sub_ds_id
                df = pd.read_csv(f'./data/training/{file}')
                for index, article in tqdm(df.iterrows(),desc = f'published cards {file}', total  = len(df), miniters=1):
                    if not len(self.df.loc[(self.df['ds_id'] == ds_id) & (self.df['ds_index'] == index)]):
                        claim = article['claim'][0]
                        sub = article['claim'].replace('_','.')
                        if claim == str(0):
                            binary = False
                        else:
                            binary = True
                        f = str(article['text']).split('.')
                        for sentence in f:
                            self.df.loc[len(self.df)] = [ds_id,None,None,None,None,None,None,sentence,True,binary,claim,sub,None]

        except KeyboardInterrupt:
            self.save()

    def inflate(self):
        pass

    def save(self):
        self.df.to_csv(os.path.join(self.dir,'data','dataset.csv'),index=False)

d = Dataset()
#d.download()
d.process()
d.save()
