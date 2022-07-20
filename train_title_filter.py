import wandb
from modules.filter import Filter, dataset_to_xy_sentence
from modules.dataset import Dataset

def main():
    run = wandb.init()

    d = Dataset(run)
    d.load('genie/dev:latest')

    X,y = dataset_to_xy_sentence(d)

    filter = Filter(run,min_count=100, threshold=0.9, model_size=500, rank_score=1)

    filter.fit(X,y)

    test 

if __name__ == '__main__':
    f = main()
