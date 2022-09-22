
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Salary_Data.csv')
print(data)

x=data.iloc[:,[0]].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split=train_test_split(x,y,test_size=0.2)

model=LinearRegression()
lm=model.fit(x_train,y_train)
pre=lm.predict(x_test)

print(pre[0:5])

r2_score =lm.score(x_test,y_test)
print(r2_score*100,'%')

import pickle

pickle.dump(lm,open('sal.pkl', 'wb'))
