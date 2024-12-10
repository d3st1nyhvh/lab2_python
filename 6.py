import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Laba_2_Python\Airpollution.csv') 
data = data.drop('City', axis=1)
y = data[['AQI Value']]
X = data.drop('AQI Value', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)
scaler = StandardScaler() # Задание 6

# скейлим данные с fit_transform т.к. он нужен для обучающих данных (чтобы одновременно и учить и преобразовывать данные)
X_traint_scaled = scaler.fit_transform(X_train.select_dtypes(exclude=['object']))
# скейлим данные с transform т.к. нужен для тестовых данных (применяет параметры и т.д.)
X_test_scaled = scaler.transform(X_test.select_dtypes(exclude=['object'])) 

# дальнейшие действия - вывод таблиц в html для её настройки отображения
df_X_traint_scaled= pd.DataFrame(X_traint_scaled)
df_X_test_scaled= pd.DataFrame(X_test_scaled)

html = f"""
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style = "flex: 1; padding: 10px; height: 300px"">{df_X_traint_scaled.to_html(index=False)}</div>
    <div style = "flex: 1; padding: 10px; height: 300px">{df_X_test_scaled.to_html(index=False)}</div>
</div>
"""
display(HTML(html))