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
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
        elif model == 'ExtraTreeClassifier':
            self.model_class = ExtraTreesClassifier
        elif model == 'RandomForestClassifier':
            self.model_class = RandomForestClassifier
        elif model == 'LinearSVC':
            self.model_class = LinearSVC
        elif model == 'LogisticRegression':
            self.model_class = LogisticRegression
        elif model == 'Perceptron':
            self.model_class = Perceptron
        elif model == 'SVC':
            self.model_class = SVC
        elif model == 'KNeighborsClassifier':
            self.model_class = KNeighborsClassifier
            self.kwargs.pop('class_weight', None)
            


        
    
    def train(self, training_data):
        self.X_train = training_data['vector'].apply(lambda x: np.array(x))
        self.Y_train = training_data['class'].apply(lambda x: str(x))
        self.model = self.model_class(**self.kwargs)
        self.model.fit(vstack(self.X_train), self.Y_train)

    def predict(self, df):
        return self.model.predict(vstack(df['vector']))

