import gc
from filter import Filter
from dataset import Dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils import *
from model import Model
from embedding import Embedding
from predictive_model import Predictive_model
import numpy as np
from collections import Counter
import scipy.stats as st


def evaluate_filter(train,test):
    # confidence interval for 1* margin of error
    # n = z^2*p(1-p)/e^2
    min_count_parms =  [int(st.norm.ppf(x)**2 * 0.5*(1-0.5)/(0.01**2)) for x in  [x*0.005 for x in range(100,200)] + [0.999, 0.9999, 0.99999]]
    model_size_parms = [int(10**(x/10)) for x in range(10,101)] 
    threshold_parms = [x/100 for x in range(0,101)]
    out = []
    ## Due to how the filter is built, it can be trained once and altered each time for parameters
    f = Filter()
    #f.train(train[train['weak_climate']]['sentence'],train[~train['weak_climate']]['sentence'])
    target_corpus = word_token(' '.join(train[train['weak_climate']]['sentence'].dropna()))
    general_corpus = word_token(' '.join(train[~train['weak_climate']]['sentence'].dropna()))
    target_count = dict(Counter(clean_words(target_corpus)))
    general_count = dict(Counter(clean_words(general_corpus)))
    print( f'Building Filters for {len(train)} examples')
    f.total_count = {k: general_count.get(k, 0) + target_count.get(k, 0) for k in set(general_count).union(set(target_count))}
    # Can only concider words that are in the general list to avoid division by zero errors 
    normed_target = {k: target_count[k] if k in target_count.keys() else 0 for k in f.total_count.keys()}
    normed_general = {k: general_count[k] if k in general_count.keys() else 0 for k in f.total_count.keys()}
    ratio = {k: (normed_target[k])*sum(normed_general.values())/(normed_general[k])/sum(normed_target.values())  for k in set(f.total_count.keys()).intersection(set(general_count))}
    # The assume the next sample for words not in general corpus is in the general cormus 
    for word in set(f.total_count.keys()) - (set(general_count)):
        ratio[word] = target_count[word]
    f.norm = pd.Series({k: v/(v+1)  for k,v in ratio.items()})
    for min_count in min_count_parms:
        f.min_count = min_count
        words = set([k for k,v in f.total_count.items() if v >= min_count])
        f.norm = pd.Series({k: ratio[k]/(ratio[k]+1)  for k in words })
        score = pd.Series([abs(0.5-x) * general_count.get(i,0) for i,x in f.norm.iteritems()], index=f.norm.index)
        f.norm =  f.norm.loc[score.index]
        for model_size in [x for x in  [len(f.norm)] + model_size_parms if x <= len(f.norm)]:
            f.model_size = model_size
            f.model = f.norm.iloc[0:f.model_size]
            for i in range(len(test)):
                test[i]['prob'] = f.predict(test[i],return_prob=True, quiet=True)
            for threshold in threshold_parms:
                f.threshold = threshold
                test_report = []
                confusion = []
                for i in range(len(test)):
                    test[i]['predicted'] = test[i]['prob'].apply(lambda x: x >= threshold)
                    test_report.append(classification_report(test[i]['label'],test[i]['predicted'], output_dict = True))
                    confusion.append(confusion_matrix(test[i]['label'],test[i]['predicted']))
                out.append(pd.DataFrame([[min_count,model_size,threshold,len(train),test_report, confusion]], columns = ['min_count','model_size','threshold', 'training_size', 'test_report','confusion_matrix']))
    return out 

def evaluate_model(train_labeled, train_unlabeled, test, filter,embedding,type):
    predictive_model_parms = ['DecisionTree','SGDClassifier','RandomForestClassifier','LinearSVC','LogisticRegression', 'KNeighborsClassifier']
    weighting_parms = ['balanced', None]
    out = []
    m = Model()
    m.filter = filter
    author_info = True
    m.embedding_scheme = Embedding(embedding,author_info)
    m.embedding_scheme.train(train_unlabeled)
    train_labeled['vector'] = m.embedding_scheme.predict(train_labeled)
    for predictive_model in predictive_model_parms:
        for weights in weighting_parms:
            m.predictive_model = Predictive_model(model = predictive_model, kwargs = {'class_weight':weights})
            print(f'Training Model on {len(train_labeled)} with {embedding}({len(train_unlabeled)}), {predictive_model}, {author_info} author_info and {weights} weights')
            m.predictive_model.train(train_labeled)
            predicted = m.predict(test)['predicted']
            out.append(pd.DataFrame([[embedding,author_info,predictive_model,weights,len(train_labeled),len(train_unlabeled), confusion_matrix(test['class'], predicted), classification_report(test['class'], predicted, output_dict = True), type]]))
    return out
                    
                    
if __name__ == '__main__':

    ## Inititate dataset
    d = Dataset()
    f_val_AL10K = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:43546a2f-82d6-49a3-af54-69b02cff54a9/AL-10Ks.tsv%20:%203000%20(58%20positives,%202942%20negatives)%20(TSV,%20127138%20KB).tsv' , sep = '\t')
    f_val_ALwiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:9d139a47-878c-4d2c-b9a7-cbb982e284b9/AL-Wiki%20(train).tsv', sep = '\t')
    f_val_wiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:c4f6e427-6b84-41ca-a016-e66337fb283b/Wiki-Doc-Train.tsv', sep = '\t')
    f_test_10k = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:cf6dea3a-ca4f-422f-8f1c-e90d88dd56dd/10-Ks%20(2018,%20test).tsv', sep = '\t')
    f_test_wiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:8533e714-155f-49f2-b997-6b9873749303/Wikipedia%20(test).tsv', sep = '\t') 
    f_test_claims =  pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:d5e1ac74-0bf1-4d84-910f-7a9c7cd28764/Claims%20(test).tsv', sep = '\t')

    data = d.apply_labels(d.df_sentence)
    data['class'] = data['sub_sub_claim'].apply(lambda x: str(x))
    data['domain'] = d.domains(data)
    data['weak_climate'] =  data['parent'].isin(np.concatenate((d.df_climate.index,d.df_skeptics.index)))

    labeled_data, unlabeled_data = [ x for _, x in data.groupby(data['sub_sub_claim'].isna())]

    """
    What Questions do we want to answer?

    Basic Tuning:
     - What is the best filter
     - What is the best embedding scheme + model combo
     - Do weights inprove the model

    Is it worthwhile to gather more data:
     - Does the filter improve with more data?
     - Does the model improve with more unlabled_data
     - Does the model improve with more labeled_data
     - Does author_info improve the model?
     - Do I even need a filter -> is embedding scheme improved by more test overall
     - If inconclusive -> is there a tradeoff between the best filter and the best model

    since labeled training set has a large amount of weakly labeled_data, filter will always used to remove non climate senteces from this set

    """


    # Tuning Filter
    """
    There will exist some optimal kwargs that will maximise any given evaluator for a model

    Due to the nature of the pipline, this evaluation can be done in two stages:

        1. Evaluate the Filter
        2. Evaluate the Embedding Scheme + the Predictive_model

    The filter can be evaluated seperatley as its goal is to determine if a sentence is about climate change or not.

    The parameters for the filter is as follows:
        Min count -> provides a minumum number of aoccurances, higher means more confidence in pr(word) being correct in the population
        Threshold -> pr for true vs false
        Alpha -> Smoothing cooef
        model_size -> number of words taken into account in final model, smaller model = greater speed.
    """ 
    filter_evaluation_dfs = [simple_starmap(evaluate_filter, [[unlabeled_data.sample(frac =2**(-x), random_state=1), [pd.concat([ f_val_ALwiki, f_val_AL10K], ignore_index = True), f_test_wiki, f_test_10k, f_test_claims]] for x in range(0,15)])]
    filter_evaluation = pd.concat(flatten(flatten(filter_evaluation_dfs)))
    filter_evaluation['a'] = filter_evaluation['confusion_matrix'].apply(lambda x: (x[0][0][0] + x[0][1][1]) / (sum(flatten(x[0]))))
    for t in ['0','1','macro avg','weighted avg']:
        filter_evaluation[f'p({t})'] = filter_evaluation['test_report'].apply(lambda x: x[0][t]['precision'])
        filter_evaluation[f'r({t})'] = filter_evaluation['test_report'].apply(lambda x: x[0][t]['recall'])
        for beta in ['0.125','0.25','0.5', '1', '2','4','8']:
            filter_evaluation[f'f{beta}({t})'] = filter_evaluation.apply(lambda x: (1+float(beta)**2) * x[f'p({t})']*x[f'r({t})'] /((float(beta)**2)*x[f'p({t})'] + x[f'r({t})']) if x[f'r({t})'] != 0 else np.NaN, axis=1)
    filter_evaluation.to_pickle('./picklejar/filter_eval.pickle')
    filter_evaluation = pd.read_pickle('./picklejar/filter_eval.pickle')
    """
    best filter is a compromise between the best f1 and the time it takes to clasify a set of documents
    time to clasify is determined by the model size
    therefore, if there exists some model which is significantly smaller than the best model with only a small sacrifice in performance, this will be the 'best model'
    """
    best_filter_parms = filter_evaluation[filter_evaluation.training_size == max(filter_evaluation.training_size)].sort_values('f1(1)', ascending=False).iloc[0]
    print(best_filter_parms)
    ## God save my ram
    del filter_evaluation
    gc.collect()
    best_filter = Filter({
            'threshold':best_filter_parms['threshold'],
            'min_count':best_filter_parms['min_count'],
            'model_size':best_filter_parms['model_size'],
            'training_size':best_filter_parms['training_size']
        })
    best_training_data = unlabeled_data
    best_filter.train(best_training_data[best_training_data['weak_climate']]['sentence'].dropna(), best_training_data[~best_training_data['weak_climate']]['sentence'].dropna())

    """
    Now that the best filter has been determined, the best predicctive model can be found.
    To do this cross valisation will be used to tune four hyperparameters:

    embedding_scheme
    author_info
    predictive_model
    weights

    cross validation will be done using 4 folds, since this allows to allocate 4 cores of development machine
    test will be a sample of 0.1
    """
    filtered = labeled_data[best_filter.predict(labeled_data)]
    train_labeled, test = train_test_split(filtered, test_size=0.1, random_state=1, stratify = filtered['class'])
    train_labeled = train_labeled.groupby('class').filter(lambda x: len(x) >1)
    train_unlabeled = unlabeled_data[best_filter.predict(unlabeled_data)]

    fin_test = pd.concat([test,labeled_data.loc[(~labeled_data.index.isin(filtered.index)) & (labeled_data.index.isin(d.df_labels.index))]])

    fin_test.to_pickle('./picklejar/test.pickle')

    ### If we have less than 10 labels for a class we remove it
    ### Will be fixed by the active leaning scheme as these will not fit into nice categories therefore will have high entropy

    ## Go through range of unlabed data
    in_ls = []
    for x in range(0,6):
        train, validate = train_test_split(train_labeled, train_size = 2**(-x-0.1),stratify = train_labeled['class'], random_state =1)
        print(len(train), len(train_unlabeled))
        for embedding_scheme in ['doc2vecdm','doc2vecdbow','tfidf','bow']:
            in_ls.append([train.copy(),train_unlabeled.copy(),validate.copy(), best_filter,embedding_scheme,'subsubclaim'])

    for x in range(1,10):
        train = train_unlabeled.sample(frac = 2**(-x), random_state =1)
        train_l, validate = train_test_split(train_labeled, train_size = 2**(-0.1),stratify = train_labeled['class'], random_state = 1)
        print(len(train_l), len(train))
        for embedding_scheme in ['doc2vecdm','doc2vecdbow','tfidf','bow']:
            in_ls.append([train_l.copy(),train.copy(),validate.copy(), best_filter,embedding_scheme,'subsubclaim'])

        
    sub_claim = train_labeled.copy()
    sub_claim['class'] = sub_claim.sub_sub_claim.apply(lambda x: str(x)[0:3]) 
    train, validate = train_test_split(sub_claim, train_size = 2**(-0.1),stratify = sub_claim['class'], random_state =1)
    for embedding_scheme in ['doc2vecdm','doc2vecdbow','tfidf','bow']:
            in_ls.append([train.copy(),train_unlabeled.copy(),validate.copy(), best_filter,embedding_scheme, 'subclaim'])

    binary  = train_labeled.copy()
    binary['class'] = binary.sub_sub_claim.apply(lambda x: '0' if str(x)[0] == '0' else '1') 
    train, validate = train_test_split(binary, train_size = 2**(-0.1),stratify = binary['class'], random_state =1)
    for embedding_scheme in ['doc2vecdm','doc2vecdbow','tfidf','bow']:
            in_ls.append([train.copy(),train_unlabeled.copy(),validate.copy(), best_filter,embedding_scheme,'binary'])

    out = simple_starmap(evaluate_model, in_ls, size = 5)
    model_eval = pd.concat(flatten(out))
    model_eval.columns = ['embedding','author_info','model','weights','labeled','unlabeled', 'confusion_matrix', 'test_report','type']
    model_eval['p'] =  model_eval['test_report'].apply(lambda x: x['macro avg']['precision'])
    model_eval['r'] =  model_eval['test_report'].apply(lambda x: x['macro avg']['recall'])
    model_eval['f1'] =  model_eval['test_report'].apply(lambda x: x['macro avg']['f1-score'])
    model_eval['a'] =  model_eval['test_report'].apply(lambda x: x['accuracy'])
    model_eval.to_pickle('./picklejar/model_eval.pickle')

    levels = ['subsubclaim','subclaim','binary']
    for level in levels:
        best_model_parms = model_eval[(model_eval.type == level) & (model_eval.labeled == max(model_eval.labeled)) & (model_eval.unlabeled == max(model_eval.unlabeled))].sort_values('f1', ascending = False).iloc[0]
        best_model =  Model({
            'filter':{
                'model_size':best_filter_parms['model_size'],
                'threshold':best_filter_parms['threshold'],
                'min_count':best_filter_parms['min_count']
                },
            'embedding': {
                'model_type':best_model_parms['embedding'],
                'author_info':best_model_parms['author_info'],
                'args': {}
                },
            'predictive_model': {
                'model_type':best_model_parms['model'],
                'args':{'class_weight':best_model_parms['weights']}
                }
                })

        if level == 'subsubclaim':
            fin_test['class'] = fin_test.sub_sub_claim.apply(lambda x: str(x))
            data['class'] = data.sub_sub_claim.apply(lambda x: str(x))
            best_model.train(data.loc[~data.index.isin(test.index)])
            predicted = best_model.predict(fin_test)['predicted'].apply(lambda x: "0.0.0" if x == '0' else x)
        elif level == 'subclaim':
            fin_test['class'] = fin_test.sub_sub_claim.apply(lambda x: str(x)[0:3]) 
            data['class'] = data.sub_sub_claim.apply(lambda x: str(x)[0:3]) 
            best_model.train(data.loc[~data.index.isin(test.index)])
            predicted = best_model.predict(fin_test)['predicted'].apply(lambda x: "0.0" if x == '0' else x)
        elif level == 'binary':
            fin_test['class'] = fin_test.sub_sub_claim.apply(lambda x: '0' if str(x)[0] == '0' else '1') 
            data['class'] = data.sub_sub_claim.apply(lambda x: '0' if str(x)[0] == '0' else '1') 
            best_model.train(data.loc[~data.index.isin(test.index)])
            predicted = best_model.predict(fin_test)['predicted']



        rep = classification_report(fin_test['class'],predicted, output_dict = True) 
        rep_df = pd.DataFrame(rep).T
        conf = pd.DataFrame(confusion_matrix(fin_test['class'],predicted, labels = [x for x in rep_df.index if not x in ['macro avg', 'accuracy','weighted avg']]))
        conf.columns =  [x for x in rep_df.index if not x in ['macro avg', 'accuracy','weighted avg']]
        conf.index =  [x for x in rep_df.index if not x in ['macro avg', 'accuracy','weighted avg']]

        rep_df.to_pickle(f'./picklejar/{level}_report.pickle')
        conf.to_pickle(f'./picklejar/{level}_conf.pickle')

    



        

