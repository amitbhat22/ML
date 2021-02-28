from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
iris = datasets.load_iris()
print(iris.target_names)
print(iris.feature_names)
print(iris.data[0:5])
print(iris.target)
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
print(data.head())
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
#print("CM:",metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

print(clf.predict([[3, 2, 3, 2]]))