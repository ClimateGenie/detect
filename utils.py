import hashlib
import nltk
import gensim
from multiprocessing import Pool
from tqdm import tqdm
from uuid  import UUID



def sent_token(doc):
    if not isinstance(doc,str):
        doc = ""
    return flatten([ x.splitlines() for x in nltk.tokenize.sent_tokenize(doc)])

def word_token(sentence):
    if not isinstance(sentence,str):
        sentence = ""
    words =  gensim.utils.simple_preprocess(sentence)
    return words

def clean_words(words):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    eng_words = set(nltk.corpus.words.words())
    words =  [w for w in words if w in eng_words]
    words =  [w for w in words if not w in stop_words]
    return words

def mult_word_token(ls_sentence):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    eng_words = set(nltk.corpus.words.words())
    out = []
    for sentence in ls_sentence:
        if not isinstance(sentence,str):
            sentence = ""
        words =  gensim.utils.simple_preprocess(sentence)
        words =  [w for w in words if w in eng_words]
        words =  [w for w in words if not w in stop_words]
        out.append(words)
    return(out)

def flatten(ls):
    return [item for sublist in ls for item in sublist]

def simple_starmap(func, ls, size = 8):
    pool = Pool(size)
    out = pool.starmap(func,ls)
    pool.close()
    pool.join()
    return out


def simple_map(func, ls, desc = None):
    pool = Pool()
    if desc:
        out = [x for x in tqdm(pool.imap(func,ls), total= len(ls),desc=desc)]
    else:

        out = pool.map(func,ls)
    pool.close()
    pool.join()
    return out

def uuid(values):
    values = ''.join([str(x) for x in values])
    seed = int(hashlib.sha256(values.encode('utf-8')).hexdigest(), 16) % 2**127
    return UUID(int=seed)
