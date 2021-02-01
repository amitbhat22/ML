import csv
import pandas as pd
import numpy as np
with open('Enjoy-sport.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
print(data,"\n")
print("\n")

d = np.array(data)[:,:-1]
print(" The attributes are: \n",d)


h = ['0', '0', '0', '0', '0', '0']

for row in data:
    if row[-1] == 'yes':
        j = 0
        
        for col in row:
            if col != 'yes':
                if col != h[j] and h[j] == '0':
                    h[j] = col
                elif col != h[j] and h[j] != '0':
                    h[j] = '?'
                    
            j = j + 1
print("\n")
print('Maximally Specific Hypothesis: ', h)