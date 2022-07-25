import wandb
import numpy as np
import pandas as pd
from modules.filter import Filter, dataset_to_xy
from modules.dataset import Dataset
from sklearn.metrics import classification_report
import os




def main():
    sweep_configuration = {
	"name": "filter_grid_sweep_bayes",
	"metric": {"name": "f1-score", "goal": "maximize"},
	"method": "bayes",
	"parameters": {
	    "min_count": {
                "min": 0,
                "max":100000
	    },
	    "threshold": {
                "min":0.8,
                "max":1.0
	    },
	    "model_size": {
                "min":1,
                "max":100000
	    },
	    "rank_score": {
                "min":0.0,
                "max":1.0
	    }
	}
    }
    sweep_id = wandb.sweep(sweep_configuration)
    wandb.agent(sweep_id,function=train_filter)





def construct_artifacts():

    run = wandb.init(project='genie')
    f_val_AL10K = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:43546a2f-82d6-49a3-af54-69b02cff54a9/AL-10Ks.tsv%20:%203000%20(58%20positives,%202942%20negatives)%20(TSV,%20127138%20KB).tsv' , sep = '\t')
    f_val_ALwiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:9d139a47-878c-4d2c-b9a7-cbb982e284b9/AL-Wiki%20(train).tsv', sep = '\t')
    f_test_10k = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:cf6dea3a-ca4f-422f-8f1c-e90d88dd56dd/10-Ks%20(2018,%20test).tsv', sep = '\t')
    f_test_wiki = pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:8533e714-155f-49f2-b997-6b9873749303/Wikipedia%20(test).tsv', sep = '\t') 
    f_test_claims =  pd.read_csv('https://www.sustainablefinance.uzh.ch/dam/jcr:d5e1ac74-0bf1-4d84-910f-7a9c7cd28764/Claims%20(test).tsv', sep = '\t')
  
    val =  pd.concat([f_val_ALwiki, f_val_AL10K,f_test_10k,f_test_wiki,f_test_claims])

    X_val = np.array([val.sentence.values]).T
    y_val = val.label.values.astype(bool)

    
    artifact = wandb.Artifact('Filter-Validation', type = 'ValidationData')
    with artifact.new_file('X_val.npy', 'wb') as f:
        np.save(f,X_val)
    with artifact.new_file('y_val.npy', 'wb') as f:
        np.save(f,y_val)
    run.log_artifact(artifact)
    
    d = Dataset(run)
    d.load('genie/defaults:latest')
    X,y = dataset_to_xy(d) 

    artifact = wandb.Artifact('Filter-Train', type = 'FilterData')
    with artifact.new_file('X.npy', 'wb') as f:
        np.save(f,X)
    with artifact.new_file('y.npy', 'wb') as f:
        np.save(f,y)
    run.log_artifact(artifact)

    run.finish()

def train_filter():
    with wandb.init() as run:
        
        artifact = run.use_artifact('genie/Filter-Validation:latest')
        path = artifact.download()
        X_val = np.load(os.path.join(path,'X_val.npy'),allow_pickle=True)
        y_val = np.load(os.path.join(path,'y_val.npy'))
        
        artifact = run.use_artifact('genie/Filter-Train:latest')
        path = artifact.download()
        X = np.load(os.path.join(path,'X.npy'),allow_pickle=True)
        y = np.load(os.path.join(path,'y.npy'))

        filter = Filter(run)
        filter.set_params(**run.config)

        filter.fit(X,y)

        
        pred_val = filter.predict(X_val)

        report = classification_report(y_val, pred_val,output_dict=True,zero_division=0)

        run.log(report["True"])
        run.log({'accuracy':report["accuracy"]})
    

if __name__ == '__main__':
    #construct_artifacts()
    main()
