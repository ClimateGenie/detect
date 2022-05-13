from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # for showing confusion matrix
from sklearn.preprocessing import MinMaxScaler # for feature scaling
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import numpy as np

class Model():
    def __init__(self,training_data):
        self.X = np.array([np.array(x) for x in training_data['vector']])
        self.Y = np.array(training_data['class'].values)
        self.train()
    
    def train(self, model_class = LabelPropagation, kernel='knn', gamma=20, n_neighbors=7, max_iter=30, tol=0.001, n_jobs=-1):
        model = model_class(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors,max_iter=max_iter, tol=tol, n_jobs=n_jobs)

        self.model = model.fit(self.X, self.Y)
