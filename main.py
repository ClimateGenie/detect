from model import Model
from scipy.stats.distributions import entropy as get_entropy
from predictive_model import Predictive_model
from scipy.sparse import vstack
import pandas as pd
from evaluate import evaluate_model


model = Model()
d = Dataset()    
training_data = d.apply_labels(d.df_sentence)
training_data['domain'] = d.domains(training_data)
training_data = training_data[~training_data['sentence'].isna()]
training_data['weak_climate'] =  training_data['parent'].isin(np.concatenate((d.df_seed.index,d.df_climate.index,d.df_skeptics.index)))
model.train(training_data)


while True:
    unlabeled = training_data[training_data['sub_sub_claim'].isna()]
    to_label = unlabeled[model.filter.predict(unlabeled)]
    to_label['distribution'] = [*model.m.model.predict_proba(vstack(to_label['vector'].value
    to_label['entropy'] = to_label['distribution'].apply(lambda x: get_entropy(x))

    model.d.get_labels(to_label, n = 50)
    model.training_data =model.d.encode_labels(model.d.apply_labels(model.training_data))

    
    model.m = Predictive_model(model.training_data[model.training_data['climate'] == True], model=model.args['predictive_model']['model_type'],  kwargs=model.args['predictive_model']['args'])
    model.m.train()
