import nltk
import gensim
from multiprocessing import Pool
from tdqm import tdqm



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

def simple_map(func, ls):
    pool = Pool()
    out = [x for x in tqdm(pool.imap(func,ls), total= len(ls))]
    pool.close()
    pool.join()
    return out

