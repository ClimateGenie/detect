import pandas as pd
import numpy as np
from scipy.sparse import vstack
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model  import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model  import PassiveAggressiveClassifier    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.ensemble  import BaggingClassifier
from sklearn.ensemble  import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

class Predictive_model():
    def __init__(self, model = 'ExtraTreeClassifier', kwargs = {}):

        self.kwargs = kwargs
        if model == 'ExtraTree':
            self.model_class = ExtraTreeClassifier
        elif model == 'DecisionTree':
            self.model_class = DecisionTreeClassifier
        elif model == 'SGDClassifier':
            self.model_class = SGDClassifier
        elif model == 'RidgeClassifier':
            self.model_class = RidgeClassifier
        elif model == 'PassiveAggressiveClassifier':
            self.model_class = PassiveAggressiveClassifier
        elif model == 'AdaBoostClassifier':
            self.model_class = AdaBoostClassifier
        elif model == 'GradientBoostingClassifier':
            self.model_class = GradientBoostingClassifier
        elif model == 'BaggingClassifier':
            self.model_class = BaggingClassifier
        elif model == 'ExtraTreeClassifier':
            self.model_class = ExtraTreesClassifier
        elif model == 'RandomForestClassifier':
            self.model_class = RandomForestClassifier
        elif model == 'BernoulliNB':
            self.model_class = BernoulliNB
        elif model == 'LinearSVC':
            self.model_class = LinearSVC
        elif model == 'LogisticRegression':
            self.model_class = LogisticRegression
        elif model == 'LogisticRegressionCV':
            self.model_class = LogisticRegressionCV
        elif model == 'MultinomialNB':
            self.model_class = MultinomialNB
        elif model == 'NearestCentroid':
            self.model_class = NearestCentroid
        elif model == 'NuSVC':
            self.model_class = NuSVC
        elif model == 'Perceptron':
            self.model_class = Perceptron
        elif model == 'SVC':
            self.model_class = SVC
            


        
    
    def train(self, training_data):
        self.X_train = training_data['vector'].apply(lambda x: np.array(x))
        self.Y_train = training_data['class'].apply(lambda x: int(x))
        print(self.model_class,self.X_train[0].A)
        self.model = self.model_class(**self.kwargs)
        self.model.fit(vstack(self.X_train), self.Y_train)

    def predict(self, df):
        return self.model.predict(vstack(df['vector']))

