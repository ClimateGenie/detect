from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical 



class Model():
    def __init__(self,training_data):
        self.X = training_data['vector'].values
        self.Y = training_data['class'].values
        print(self.X.shape)
        print(self.Y.shape)
        self.train()
    
    def train(self):
        self.model = Sequential()
        self.model.add(Dense(500, activation='relu', input_dim=self.X.shape[1]))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(Y.shape[1], activation='softmax'))
    
        self.model.fit(self.X, Y, epochs=20)

    def infer_df(self,df):
        df['prediction'] = self.model.predict(df['vector'])
        return df
