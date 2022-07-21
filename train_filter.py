import wandb
import numpy as np
import pandas as pd
from modules.filter import Filter, dataset_to_xy
from modules.dataset import Dataset
from sklearn.metrics import classification_report

def main():

    f_val_AL10K = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:43546a2f-82d6-49a3-af54-69b02cff54a9/AL-10Ks.tsv%20:%203000%20(58%20positives,%202942%20negatives)%20(TSV,%20127138%20KB).tsv' , sep = '\t')
    f_val_ALwiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:9d139a47-878c-4d2c-b9a7-cbb982e284b9/AL-Wiki%20(train).tsv', sep = '\t')
    f_test_10k = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:cf6dea3a-ca4f-422f-8f1c-e90d88dd56dd/10-Ks%20(2018,%20test).tsv', sep = '\t')
    f_test_wiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:8533e714-155f-49f2-b997-6b9873749303/Wikipedia%20(test).tsv', sep = '\t') 
    f_test_claims =  pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:d5e1ac74-0bf1-4d84-910f-7a9c7cd28764/Claims%20(test).tsv', sep = '\t')
  
    val =  pd.concat([f_val_ALwiki, f_val_AL10K,f_test_10k,f_test_wiki,f_test_claims])

    X_val = np.array([val.sentence.values]).T

    y_val = val.label.values.astype(bool)

    run = wandb.init( reinit=True)

    d = Dataset(run)
    d.load('genie/defaults:latest')

    X,y = dataset_to_xy(d)

    filter = Filter(run, model_size=10000)

    filter.fit(X,y)
    
    pred_val = filter.predict(X_val)
   
    report = classification_report(y_val, pred_val,output_dict=True)


    run.config.update(filter.get_params())
    run.log(report["True"])
    run.log({'accuracy':report["accuracy"]})
    filter.save('potential_filter') 
    run.finish()

    return filter





if __name__ == '__main__':
    f = main()
