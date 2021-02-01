import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import svm #Import svm model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

data = pd.read_csv("heart.csv")
## The data looks like this

##	age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
##0	63	1	3	145	233	1	0	150	0	2.3	0	0	1	1
##1	37	1	2	130	250	0	1	187	0	3.5	0	0	2	1
##2	41	0	1	130	204	0	0	172	0	1.4	2	0	2	1
##3	56	1	1	120	236	0	1	178	0	0.8	2	0	2	1
##4	57	0	0	120	354	0	1	163	1	0.6	2	0	2	1



#Separate the data -- last column 'target' is removed from the input feature set x
x = data.drop('target',axis = 1) 
y = data.target

#split the test set and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109) # 70% training and 30% test



#Create a svm Classifier
ml = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
ml.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = ml.predict(x_test)


# Model Accuracy: how often is the classifier correct?
#print(ml.score(x_test,y_test))
cm = confusion_matrix(y_test,y_pred)
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
                
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print('False Positives\n {}'.format(FP))
print('False Negetives\n {}'.format(FN))
print('True Positives\n {}'.format(TP))
print('True Negetives\n {}'.format(TN))
TPR = TP/(TP+FN)
print('Sensitivity \n {}'.format(TPR))
TNR = TN/(TN+FP)
print('Specificity \n {}'.format(TNR))
Precision = TP/(TP+FP)
print('Precision \n {}'.format(Precision))
Recall = TP/(TP+FN)
print('Recall \n {}'.format(Recall))
Acc = (TP+TN)/(TP+TN+FP+FN)
print('Áccuracy \n{}'.format(Acc))
Fscore = 2*(Precision*Recall)/(Precision+Recall)
print('FScore \n{}'.format(Fscore))

