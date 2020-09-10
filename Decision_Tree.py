import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
data = pd.read_csv('diabetes.csv')

data.info()
array = data.values
array

X = array[:,0:8] # ivs for train
y = array[:,8] # dv
                                                                                                                                                                                                  
test_size = 0.33
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
prediction

outcome = y_test
outcome

print(metrics.accuracy_score(outcome,prediction))
#print(metrics.confusion_matrix(y_test,prediction))
print("    ")

model.feature_importances_

expected = y_test
predicted = prediction
conf = metrics.confusion_matrix(expected, predicted)


label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)

decision_tree = tree.DecisionTreeClassifier(max_depth = 4)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
cm_df = pd.DataFrame(confusion_matrix(y_test,y_pred).T, index=decision_tree.classes_,columns=decision_tree.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
print(cm_df)
print(conf)

acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
acc_decision_tree

decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
cm_df = pd.DataFrame(confusion_matrix(y_test,y_pred).T, index=decision_tree.classes_,
columns=decision_tree.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
print(cm_df)

acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
acc_decision_tree

decision_tree = tree.DecisionTreeClassifier(max_depth = 2)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
cm_df = pd.DataFrame(confusion_matrix(y_test,y_pred).T, index=decision_tree.classes_,
columns=decision_tree.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
print(cm_df)
#print(classification_report(y_test, y_pred))
acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
acc_decision_tree