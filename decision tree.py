
import pandas as pd
r=pd.read_csv('bank_contacts.csv')
x=pd.DataFrame(r.iloc[:,:-1])
y=pd.DataFrame(r.iloc[:,-1])
import sklearn.model_selection as k
x_train,x_test,y_train,y_test=k.train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
t=DecisionTreeClassifier(criterion='gini',min_samples_split=1000,random_state=42)
g=DecisionTreeClassifier(criterion='entropy',min_samples_split=1000,random_state=42)
t_pred=t.fit(x_train,y_train)
g_pred=g.fit(x_train,y_train)
y_pred=t.predict(x_test)
print('predicted values of gini:',y_pred)
h_pred=g.predict(x_test)
print('predicted values of entropy:',h_pred)
print('accuracy of gini: %d',t.score(x_test,y_test))
print('accuracy of entropy: %d',g.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(confusion_matrix(y_test,h_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(classification_report(y_test,h_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_pred))
print(roc_auc_score(y_test,h_pred))

