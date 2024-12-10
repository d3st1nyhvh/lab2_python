import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
from IPython.display import display_html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Laba_2_Python\Airpollution.csv') 
data = data.drop('City', axis=1)
y = data[['AQI Value']]
X = data.drop('AQI Value', axis=1) # сброс ключевого стоблца для X выборки

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc.fit(X_train.select_dtypes(exclude=['object'])) # Проводим fit

X_train_object = enc.transform(X_train.select_dtypes(exclude=['object'])) # Проводим transform
X_test_object = enc.transform(X_test.select_dtypes(exclude=['object'])) # Проводим transform
X_traint_OHE = pd.DataFrame(X_train_object, columns=enc.get_feature_names_out())
X_test_OHE = pd.DataFrame(X_test_object, columns=enc.get_feature_names_out())
print(X_traint_OHE)
print(X_test_OHE)