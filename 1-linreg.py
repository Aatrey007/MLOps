import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('Practicals\wine-quality.csv')
print(data)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.3)
mod = LinearRegression()
mod.fit(xtr,ytr)
ypr = mod.predict(xts)
ma = mean_squared_error(yts,ypr)
print("mean squared error :",ma)
ms = mean_absolute_error(yts,ypr)
print("mean absolute error :",ms)
 