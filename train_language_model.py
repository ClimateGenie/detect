import wandb
from simpletransformers.language_modeling import LanguageModelingModel

def train():
    api = wandb.Api()
    train_doc = api.artifact('genie/Language-Model-Training-Data:latest').download()+ '/sentence.txt'
    eval_doc = api.artifact('genie/Language-Model-Test-Data:latest').download() + '/sentence.txt'


    model =  LanguageModelingModel('roberta','roberta-large', args={'wandb_project':'genie'})
    model.train_model(train_doc, './lm')
    model.eval_model(eval_doc)

    




if __name__ == "__main__":
    train()

