import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.model_selection as k 
r=pd.read_csv('bank_contacts.csv')
x=r.drop('credit_application',axis=1)
y=r['credit_application']
train_x,test_x,train_y,test_y=k.train_test_split(x,y,test_size=0.2)
from sklearn.svm import SVC
t=SVC(kernel='linear',C=1.0)
t.fit(train_x,train_y)
y_pred=t.predict(test_x)
print('accuracy: %d',t.score(test_x,test_y))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y,y_pred))
from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(test_y,y_pred))
w =t.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx -t.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(test_x.iloc[:, 0],test_x.iloc[:, 1], c=y_pred, s=50, cmap='autumn')
plt.show()

    



