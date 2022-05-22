from model import Model
from scipy.stats.distributions import entropy as get_entropy
from predictive_model import Predictive_model
from scipy.sparse import vstack
import pandas as pd

model = Model()

model.train()

while True:
    to_label = model.training_data[model.training_data['sub_sub_claim'].isna()][model.training_data['climate']]
    print(to_label)
    to_label['distribution'] =  model.m.model.predict_proba(vstack(to_label['vector'].values))
    to_label['entropy'] = to_label['distribution'].apply(lambda x: get_entropy(x))

    model.d.get_labels(to_label, n = 50)
    model.training_data =model.d.encode_labels(model.d.apply_labels(model.training_data))

    
    model.m = Predictive_model(model.training_data[model.training_data['climate'] == True], model=model.args['predictive_model']['model_type'],  kwargs=model.args['predictive_model']['args'])
    model.m.train()
