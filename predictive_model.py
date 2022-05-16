from nltk.util import pr
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Predictive_model():
    def __init__(self,data):

        labeled, unlabeled = [x for _,x in data.groupby(data['class'] == -1)]
        
        
        X = labeled['vector'].apply(lambda x: np.array(x))
        Y = labeled['class'].apply(lambda x: int(x))


        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y)
        print(self.X_train)
        print(self.Y_train)
        
        self.X_train = pd.concat([self.X_train, unlabeled['vector'].apply(lambda x: np.array(x))])
        self.Y_train = pd.concat([self.Y_train, unlabeled['class']]).apply(lambda x: int(x))

        self.train()
    
    def train(self, model_class = LabelPropagation, kernel='knn', gamma=20, n_neighbors=7, max_iter=30, tol=0.001, n_jobs=-1):
        print('Training Model')
        model = model_class(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors,max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        self.model = model.fit(np.stack(self.X_train.values), self.Y_train)

    def evaluate(self):
        y_hat = self.model.predict(np.stack(self.X_test))
        cm = confusion_matrix(self.Y_test, y_hat, labels=self.model.classes_)
        print(classification_report(self.Y_test, y_hat))
        disp = ConfusionMatrixDisplay(confusion_matrix= cm)
        disp.plot() 
        plt.show()
