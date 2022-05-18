from model import Model
from scipy.stats.distributions import entropy as get_entropy
from predictive_model import Predictive_model
import pandas as pd

model = Model(kwargs={
        'filter':{
            'min_count': 5000
            },
        'embedding': {
            'model_type':'doc2vecdm',
            'args': {}
            },
        'predictive_model': {
            'model_type':'semi_supervised',
            'args': {'kernel': 'knn'}
            }
    })

model.train()

while True:
    entropy = pd.Series([ x for x in model.m.model.label_distributions_])
    entropy.index = model.m.X_train.index
    entropy = entropy.apply(lambda x: get_entropy(x))

    predicted = pd.Series(model.m.model.transduction_).apply(lambda x: int(x))
    predicted.index = model.m.X_train.index
    predicted = pd.concat([predicted, model.m.Y_test])



    model.training_data= model.training_data.join(entropy.rename('entropy'), how = 'left')
    model.training_data = model.training_data.join(predicted.rename('predicted'), how = 'left')
    model.training_data.loc[~model.training_data.predicted.isna(),'predicted'] = model.d.encoder.inverse_transform( model.training_data.loc[~model.training_data.predicted.isna(),'predicted'].apply(lambda x: int(x)) )

    model.d.get_labels(model.training_data, n = 50)
    model.training_data =model.d.encode_labels(model.d.apply_labels(model.training_data))

    
    model.m = Predictive_model(model.training_data[model.training_data['climate'] == True], model=model.args['predictive_model']['model_type'],  kwargs=model.args['predictive_model']['args'])
    model.m.train()
