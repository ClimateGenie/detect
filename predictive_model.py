from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # for showing confusion matrix
from sklearn.preprocessing import MinMaxScaler # for feature scaling
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

class Model():
    def __init__(self,training_data):
        self.X = training_data['vector'].values
        self.Y = training_data['class'].values
        print(self.X.shape)
        print(self.Y.shape)
        self.train()
    
    def train(self, model_class = LabelPropagation, kernel='rbf', gamma=20, n_neighbors=7, max_iter=30, tol=0.001, n_jobs=-1):
        model = model_class(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors,max_iter=max_iter, tol=tol, n_jobs=n_jobs)

        self.model = model.fit(self.X, self.Y)
