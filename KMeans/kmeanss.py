import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('kmeans.csv')
darray=data.to_numpy()
data
plt.scatter(data.X1,data.X2)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(data)
clusters=kmeans.cluster_centers_
clusters
y=kmeans.fit_predict(data)
y
count=0
for i in y:
    if i==0:
        plt.scatter(darray[count][0],darray[count][1],s=100,color="green")
    elif i==1:
        plt.scatter(darray[count][0],darray[count][1],s=100,color="blue")
    else:
        plt.scatter(darray[count][0],darray[count][1],s=100,color="red")
    count += 1
    
plt.scatter(clusters[0][0],clusters[0][1],s=100,color="black",marker="*")
plt.scatter(clusters[1][0],clusters[1][1],s=100,color="black",marker="*")
plt.scatter(clusters[2][0],clusters[2][1],s=100,color="black",marker="*")