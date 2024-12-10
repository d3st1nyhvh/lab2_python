import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
from sklearn.model_selection import train_test_split 

data = pd.read_csv('Laba_2_Python\Airpollution.csv') 
data = data.drop('City', axis=1) # Задание 2
y = data[['AQI Value']] # Задание 3
X = data.drop('AQI Value', axis=1) # Задание 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345) # Задание 5