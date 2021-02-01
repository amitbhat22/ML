import numpy as np


data = [['A1', 28], ['A2', 32], ['A3', 1], ['A4', 0],
        ['A5', 10], ['A6', 22], ['A7', 30], ['A8', 19],
		['B1', 145], ['B2', 27], ['B3', 36], ['B4', 25],
		['B5', 9], ['B6', 38], ['B7', 21], ['B8', 12],
		['C1', 122], ['C2', 87], ['C3', 36], ['C4', 3],
		['D1', 0], ['D2', 5], ['D3', 55], ['D4', 62],
		['D5', 98], ['D6', 32]]

print(data)

print("The Nnmber of sites sampled by the field crew ")
print(len(data))

print("Number of birds seen in the 7th site")
print(data[6][1])

print("Number of birds seen in the last site")
print(data[-1][1])

print("Total number of birds across all sites")
colsum = sum(row[1] for row in data)
print(colsum)


print("The average number of birds seen on a site")
print(int(colsum/len(data)))
