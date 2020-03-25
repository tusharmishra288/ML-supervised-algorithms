import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as k
from sklearn.linear_model import LinearRegression
r=pd.read_csv('bank_contacts.csv')
x=pd.DataFrame(r.iloc[:,:-1])
y=pd.DataFrame(r.iloc[:,-1])
x_train,x_test,y_train,y_test=k.train_test_split(x,y,test_size=0.2,random_state=42)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
pred=regressor.predict(x_test)
act=x_test.iloc[:,1]
plt.plot(act,y_test,color='r')
plt.scatter(pred,y_test,color='b')
plt.show()
import sklearn.metrics as mt
import numpy as np
print(mt.mean_squared_error(y_test,pred))
print(mt.mean_absolute_error(y_test,pred))
print(np.sqrt(mt.mean_squared_error(y_test,pred)))
print(mt.roc_auc_score(y_test,pred))
print('accuracy is :',regressor.score(x_test,y_test))







