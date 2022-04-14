import pandas as pd
import random
import numpy as np
from zipfile import ZipFile
import nltk
from newspaper import Article
import wget
from tqdm import tqdm 
import os
import shutil
import gensim
from multiprocessing import Manager, Pool
from kaggle.api.kaggle_api_extended import KaggleApi
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
    def __init__(self, rebuild = False, state = 1 ):
        self.state = state
        self.dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(self.dir,'data','dataset.pickle')) and not rebuild:
            self.df = pd.read_pickle(os.path.join(self.dir,'data','dataset.pickle'))
        else:
            print('Dataset not found')
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
        self.sources = ['https://www.climate-news-db.com/download', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2015.csv.zip', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2016.csv.zip', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2017.csv.zip', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2018.csv.zip', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2019.csv.zip', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/WebNewsEnglishSnippets.2020.csv.zip', 'https://www.sustainablefinance.uzh.ch/dam/jcr:c4f6e427-6b84-41ca-a016-e66337fb283b/Wiki-Doc-Train.tsv', 'http://data.gdeltproject.org/blog/2020-climate-change-narrative/TelevisionNews.zip', 'https://drive.google.com/uc?export=download&id=1QXWbovm42oDDDs4xhxiV6taLMKpXBFTS', 'https://drive.google.com/uc?export=download&id=1exilBtj1fnryC7xkoHjZ6YZPaRZHBtbP', 'http://socialanalytics.ex.ac.uk/cards/data.zip']
        for url in self.sources:
            res = None
            while res is None:
                try:
                    print(f' \nDownloading {url}')
                    wget.download(url)
                    res = 1
                except:
                    pass

        api = KaggleApi()
        api.authenticate()

        self.kaggle_sources = [{ "name": 'News Category Dataset', "path":'rmisra/news-category-dataset'},  { "name": 'Fake and real news dataset', "path":'clmentbisaillon/fake-and-real-news-dataset'},  { "name": 'Getting Real about Fake News', "path":'mrisdal/fake-news'}, { "name": 'All the News', "path":'snapcrack/all-the-news'}, {"path":"ruchi798/source-based-news-classification"}]
        for source in self.kaggle_sources:
            api.dataset_download_files(source["path"], quiet = False)

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
            data = data + [(L,sdf,1,self.subprocess1) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            for year in range(2015,2021):
                df = pd.read_csv(f'WebNewsEnglishSnippets.{year}.csv',header = None)
                data = data + [(L,sdf,2+(year-2015)*0.1,self.subprocess2) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]


            df = pd.read_csv('Wiki-Doc-Train.tsv', sep='\t')
            data = data + [(L,sdf,3,self.subprocess3) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            for file in os.listdir('./TelevisionNews'):
                try:
                    sdf=pd.read_csv(f'./TelevisionNews/{file}')
                    
                except:
                    pass
                data = data + [(L,sdf,4,self.subprocess4)]

            df =  pd.read_json('cards_training_sub_sub_claim.json') 
            data = data + [(L,sdf,5,self.subprocess5) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            df = pd.read_csv('Climate_change_allyears_trim.csv')
            data = data + [(L,sdf,6,self.subprocess6) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            for sub_ds_id,file in enumerate(os.listdir('./data/training')):
                ds_id = 7 + 0.1*sub_ds_id
                df = pd.read_csv(f'./data/training/{file}')
                data = data + [(L,sdf,ds_id,self.subprocess7) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            df = pd.read_csv('True.csv')
            data = data + [(L,sdf,8.0,self.subprocess8) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            df = pd.read_csv('Fake.csv')
            data = data + [(L,sdf,8.1,self.subprocess8) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            df = pd.read_csv('fake.csv')
            data = data + [(L,sdf,10,self.subprocess10) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            df = pd.concat([pd.read_csv('articles1.csv'),pd.read_csv('articles2.csv'),pd.read_csv('articles3.csv')])
            data = data + [(L,sdf,11,self.subprocess11) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            df = pd.read_csv('news_articles.csv')
            data = data + [(L,sdf,12,self.subprocess12) for sdf in np.split(df, np.arange(1,round(len(df)/5000))*5000)]

            pool=Pool()
            random.shuffle(data)
            for _ in tqdm(pool.istarmap(self.subprocess, data), total=len(data), miniters = 5000, desc='Building Dataset'):
                pass
            pool.close()
            pool.join()

            print("Building DataFrame")
            self.df = pd.concat(L).apply(lambda x: self.apply_token(x), axis=1)
            self.df.reset_index(inplace = True, drop = True)
            

    def save(self):
        print('Pickling')
        self.df.to_pickle(os.path.join(self.dir,'data','dataset.pickle'))


    def sent_token(self, doc):
        if not isinstance(doc,str):
            doc = ""
        return nltk.tokenize.sent_tokenize(doc)

    def word_token(self, sentence):
        return gensim.utils.simple_preprocess(sentence)


    def subprocess(self,l,df,ds_id,target):
        target(l,df,ds_id)


    def apply_token(self,df):
        df.reset_index(inplace=True, drop = True)
        df = df[df['sentence'].apply(lambda x: isinstance(x, str))]
        tokens = df['sentence'].apply(lambda x: self.word_token(x))
        df = df[tokens.str.len() > 1]
        df = df.assign(sentence = tokens)    
        return df

    def subprocess1(self,l,df,ds_id):
        # Climate-news-dbo
        # A collection of climate news
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['url'] = df['article_url']
        target_df['domain'] = df['newspaper_url']
        target_df['publisher'] = df['newspaper']
        target_df['date'] = df['date_published']
        target_df['title'] = df['headline']
        target_df['soft_climate'] = True
        target_df['ds_id'] = ds_id
        target_df['sentence']= df['article_id']
        for index, article in target_df.iterrows():
            if os.path.exists("./article_body/" + article['sentence']+ ".txt"):
                f = self.sent_token(open("./article_body/" + article['sentence']+ ".txt", "r").read())
                if len(f) != 0:
                    target_df.at[index,'sentence']= [f]
                    target_df['doc_len'] = target_df['sentence'].str.len()
                else:
                    target_df.loc[index,'sentence']= None
            else:
                target_df.loc[index,'sentence']= None

        l.append(target_df[target_df['sentence'] != None].explode('sentence'))

    def subprocess2(self,l,df,ds_id):
        # WebNewsEnglishSnippets
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['url'] = df[3]
        target_df['title'] = df[1]
        target_df['date'] = df[0]
        target_df['sentence'] = df[4].apply(self.sent_token)
        target_df = target_df[target_df['sentence'].str.len() > 2]
        target_df['sentence'] =  target_df['sentence'][1:-1]
        target_df['soft_climate'] = True
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))


    def subprocess3(self,l,df,ds_id):
        df = df[~df.isnull()]
        # sustainablefinance
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['title'] = df['title']
        target_df['domain'] = 'https://en.wikipedia.org/'
        target_df['publisher'] = 'wikipedia'
        target_df['soft_climate'] = df['label'].apply(lambda x : bool(x))
        target_df['sentence'] = df['sentence'].apply(self.sent_token)
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))

    def subprocess4(self,l,df,ds_id):
        df = df[~df.isnull()]
        #TelevisionNews
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['url'] = df['URL']
        target_df['date'] = df['MatchDateTime']
        target_df['publisher'] = df['Station']
        target_df['soft_climate'] = True
        target_df['sentence'] = df['Snippet'].apply(self.sent_token)
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))


    def subprocess5(self,l,df,ds_id):
        df = df[~df.isnull()]
        #Labeled by trav
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['claim']=df['sub_claim'].apply(lambda x: str(x)[0])
        target_df['sub']=df['sub_claim'].apply(lambda x: str(x)[-1])
        target_df['subsub']=df['sub_sub_claim'].apply(lambda x: str(x)[-1])
        target_df['binary'] = df['sub_claim'].apply(lambda x : True if str(x)[0] != '0' else False)
        target_df['sentence'] = df['Paragraph_Text'].apply(self.sent_token)
        target_df['soft_climate'] = True
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))


    def subprocess6(self,l,df,ds_id):
        df = df[~df.isnull()]
        #Miriams dataset
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['title'] = df['Headline']
        target_df['date'] = df['Date']
        target_df['publisher'] = df['LP']
        target_df['sentence'] = df['Paragraph'].apply(self.sent_token)
        target_df['soft_climate'] = True
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))

    def subprocess7(self,l,df,ds_id):
        df = df[~df.isnull()]
        #Public Cards dataset
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['claim']=df['claim'].apply(lambda x: str(x)[0])
        target_df['sub']=df['claim'].apply(lambda x: str(x)[-1])
        target_df['binary'] = df['claim'].apply(lambda x : True if str(x)[0] != '0' else False)
        target_df['sentence'] = df['text'].apply(self.sent_token)
        target_df['soft_climate'] = True
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))


    def subprocess8(self,l,df,ds_id):
        #Fake_real_news https://www.uvic.ca/ecs/ece/isot/assets/docs/ISOT_Fake_News_Dataset_ReadMe.pdf
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['binary'] = bool(int(10*(ds_id - 7)))
        target_df['title'] = df['title']
        target_df['sentence'] = df['text'].apply(self.sent_token)
        target_df['date'] = df['date']
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))


    def subprocess9(self,l,df,ds_id):
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        try:
            article = Article(df['link'].values[0].replace('https://www.huffingtonpost.comhttp','http'))
            article.download()
            article.parse()
            target_df['sentence'] = self.sent_token(article.text)
            target_df['url'] = df['link']
            target_df['date'] = df['date']
            target_df['author'] = df['authors']
            target_df['ds_id'] = ds_id
            target_df['soft_climate'] = df['category'].apply(lambda x : True if x == 'ENVIROMENT' else False)
            target_df['doc_len'] = target_df['sentence'].str.len()
            l.append(target_df.explode('sentence'))
        except:
            pass

    def subprocess10(self,l,df,ds_id):
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['sentence'] = df['text'].apply(self.sent_token)
        target_df['publisher'] = df['author']
        target_df['title'] = df['title']
        target_df['date'] = df['published']
        target_df['domain'] = df['site_url']
        target_df['publisher'] = df['author']
        target_df['binary'] = True
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))

    def subprocess11(self,l,df,ds_id):
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['sentence'] = df['content'].apply(self.sent_token)
        target_df['title'] = df['title']
        target_df['publisher'] = df['publication']
        target_df['author'] = df['author']
        target_df['date'] = df['date']
        target_df['url'] = df['url']
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))

    def subprocess12(self,l,df,ds_id):
        target_df = pd.DataFrame(columns = ['ds_id','url','domain','publisher','date','author','title','sentence','doc_len','soft_climate','true_climate','binary','claim','sub','subsub'])
        target_df['sentence'] = df['text'].apply(self.sent_token)
        target_df['publisher'] = df['author']
        target_df['title'] = df['title']
        target_df['date'] = df['published']
        target_df['publisher'] = df['author']
        target_df['domain'] = df['site_url']
        target_df['binary'] = df['label'].apply(lambda x : True if x == 'Fake' else False)
        target_df['ds_id'] = ds_id
        target_df['doc_len'] = target_df['sentence'].str.len()
        l.append(target_df.explode('sentence'))
        

    def subset(self, cols):
        sub_df = self.df[cols].dropna()

        return sub_df


    def gather_embedding_training(self,ds_array = [1,2.0,2.1,2.2,2.3,2.4,3,4,5,6,7.0,7.1,7.2,8.0,8.1], sample = None):
        print('Gathering Climate Claims')
        train =  self.subset(['ds_id','sentence','climate'])
        train = train[(train['climate'] == True) & (train['ds_id'].isin(ds_array))]
        if sample: 
            train = train.sample(frac = sample, random_state = self.state).reset_index()
        print(f'Processing Embedding Training Data, {len(train)} rows')
        self.embedding_train = train.apply(lambda x: gensim.models.doc2vec.TaggedDocument(x['sentence'],[x.name]), axis = 1).values


if __name__ == "__main__":
    d = Dataset(rebuild = True)
    print(d.df.groupby(['ds_id']).count())
