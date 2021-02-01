import numpy as np
import pandas as pd
from copy import deepcopy


def euclidean(a,b):
    return np.linalg.norm(a-b)


def main():
    X = pd.read_csv('hier_cluster_data.csv',index_col=False)
    print(X)

    x1 = X['X1'].values
    x2 = X['X2'].values
    X = np.array(list(zip(x1, x2)))

    farthest = 0
    closest = 10
    sum = 0
    for i in range(len(X)):
        for j in range(len(X)):
            #print(X[i])
            distance = round(euclidean(X[i], X[j]),4)
            if distance>farthest:
                farthest = distance
            if (distance > 0 and distance < closest):
                closest = distance
            if distance != 0:
                sum = sum +distance
            print(distance)
    average = sum / (len(X) * len(X))
    print("Distance betwwen farthest points",farthest)
    print("Distance between closest points",closest)
    print("Average distance = ", average)

    

if __name__ == "__main__": 
    main()


