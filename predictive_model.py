import keras
from dataset import Dataset
from embedding import Embedding
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 


class Model():
    def __init__(training_data):
        X = df[]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)


if __name__ == "__main__":
    d = Dataset()
    e = Embedding()
    labeled_data = d.fetch_labeled_data(e)
    


