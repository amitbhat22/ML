import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier,plot_tree 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix,accuracy_score
df=pd.read_csv("zoo_data.csv",header=None)
print(df.dtypes)
print(df[16].unique())
y=df[16]
X=df.drop(16, axis=1)
print(X.head())
print(y.unique())
for i in range(len(y)):
    if y[i] <= 4:
        y[i]=0
    else:
        y[i]=1
print(y.unique())    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)
clf_dt = DecisionTreeClassifier(criterion='entropy',random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)
y_pred=clf_dt.predict(X_test)
print(accuracy_score(y_pred,y_test)*100)
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(clf_dt, X_test, y_test,)
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt,filled=True,rounded=True,class_names=["No", "Yes"],feature_names=X.columns)