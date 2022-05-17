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

class Predictive_model():
    def __init__(self,data, model, kwargs):

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
        elif model == 'lin_svm':
            self.model_class = SVC


        labeled, unlabeled = [x for _,x in data.groupby(data['class'] == -1)]
        
        
        X = labeled['vector'].apply(lambda x: np.array(x))
        Y = labeled['class'].apply(lambda x: int(x))


        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y)
        
        print(self.X_train)
        print(self.Y_train)
        
        if model = 'semi_supervised':
            self.X_train = pd.concat([self.X_train, unlabeled['vector'].apply(lambda x: np.array(x))])
            self.Y_train = pd.concat([self.Y_train, unlabeled['class']]).apply(lambda x: int(x))

    
    def train(self):
        print('Training Model')
        self.model = self.model_class(**self.kwargs)
        self.model.fit(np.stack(self.X_train.values), self.Y_train)

    def predict(self, vector):
        return self.model.predict([vector])[0]

    def evaluate(self):
        y_hat = self.model.predict(np.stack(self.X_test))
        cm = confusion_matrix(self.Y_test, y_hat, labels=self.model.classes_)
        print(classification_report(self.Y_test, y_hat))
        disp = ConfusionMatrixDisplay(confusion_matrix= cm)
        disp.plot() 
        plt.show()
