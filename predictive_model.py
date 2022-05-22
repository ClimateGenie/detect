from nltk.util import pr
import pandas as pd
from sklearn import semi_supervised
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.sparse import vstack

class Predictive_model():
    def __init__(self,data, model = 'semi_supervised', kwargs = {}):

        self.kwargs = kwargs
        if model == 'semi_supervised':
            self.model_class = LabelPropagation
        elif model == 'n_neighbors':
            self.model_class = KNeighborsClassifier
        elif model == 'rbf_svm':
            self.model_class = SVC
        elif model == 'gaussian_process':
            self.model_class = GaussianProcessClassifier
        elif model == 'decision_tree':
            self.model_class = DecisionTreeClassifier
        elif model == 'random_forrest':
            self.model_class = RandomForestClassifier
        elif model == 'nn':
            self.model_class = MLPClassifier
        elif model == 'adaboost':
            self.model_class = AdaBoostClassifier
        elif model == 'bayes':
            self.model_class = GaussianNB
        elif model == 'qda':
            self.model_class = QuadraticDiscriminantAnalysis


        labeled = data[~data['sub_sub_claim'].isna()]
        
        
        self.X_train = labeled['vector'].apply(lambda x: np.array(x))
        self.Y_train = labeled['class'].apply(lambda x: int(x))

        
    
    def train(self):
        print('Training Model')
        self.model = self.model_class(**self.kwargs)
        self.model.fit(vstack(self.X_train), self.Y_train)

    def predict(self, df):
        return self.model.predict(vstack(df['vector']))

