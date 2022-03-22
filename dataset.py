import pandas as pd
from zipfile import ZipFile
import newspaper
import wget
import os


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
                        'https://drive.google.com/uc?export=download&id=1exilBtj1fnryC7xkoHjZ6YZPaRZHBtbP'
                        ]

        self.df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','climate','binary','claim','sub','subsub']) 

    def download(self):
        try:
            os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'))
        except FileExistsError:
            pass

        try:
            os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','raw'))
        except FileExistsError:
            pass
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','raw'))
        for file in os.listdir():
            os.remove(file)
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
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','raw'))

        # Climate-news-dbo
        # A collection of climate news
        ds_id = 1
        df = pd.read_csv('climate-news-db-dataset.csv') 
        for index, article in df.iterrows():
            print(self.df)
            url = article['article_url']
            domain = article['newspaper_url']
            publisher = article['newspaper']
            date = article['date_published']
            title = article['headline']
            try:
                f = open("./article_body/" + article['article_id']+ ".txt", "r").read().split('.')
                for sentence in f:
                    self.df.loc[len(self.df)] = [ds_id,url,domain,publisher,date,None,title,sentence,True,None,None,None,None]
            except FileNotFoundError:
                pass
            
        # WebNewsEnglishSnippets
        ds_id = 2

        # sustainablefinance

        # WebNewsEnglishSnippets

        #TelevisionNews

        #Labeled by trav

        #Miriams dataset
    

    def inflate(self):
        pass
Dataset().process()
