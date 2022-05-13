import time
import hashlib
import collections
import nltk
import gensim
from multiprocessing import Pool
from tqdm import tqdm
from uuid  import UUID



def sent_token(doc):
    if not isinstance(doc,str):
        doc = ""
    return nltk.tokenize.sent_tokenize(doc)

def word_token(sentence):
    if not isinstance(sentence,str):
        sentence = ""
    return gensim.utils.simple_preprocess(sentence)


def flatten(ls):
    return [item for sublist in ls for item in sublist]

def simple_starmap(func, ls):
    pool = Pool()
    out = pool.starmap(func,ls)
    pool.close()
    pool.join()
    return out

def simple_map(func, ls, desc = None):
    pool = Pool()
    out = [x for x in tqdm(pool.imap(func,ls), total= len(ls),desc=desc)]
    pool.close()
    pool.join()
    return out

def uuid(values):
    values = ''.join([str(x) for x in values])
    seed = int(hashlib.sha256(values.encode('utf-8')).hexdigest(), 16) % 2**127
    return UUID(int=seed)


