
import pandas as pd
import sklearn.model_selection as k
r=pd.read_csv('bank_contacts.csv')
x=pd.DataFrame(r.iloc[:,:-1])
y=pd.DataFrame(r.iloc[:,-1])
train_x,test_x,train_y,test_y=k.train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
t=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
u=t.fit(train_x,train_y)
y_pred=t.predict(test_x)
import sklearn.metrics as mt
print('Accuracy will be:',t.score(test_x,test_y))
print(mt.classification_report(test_y,y_pred))
print(mt.roc_auc_score(test_y,y_pred))
print(mt.confusion_matrix(test_y,y_pred))
import matplotlib.pyplot as plt
actual_x=test_x.iloc[:,1]
plt.scatter(actual_x,test_y,c='blue',marker='o')
plt.scatter(y_pred,test_y,c='red')
plt.show()
