import csv
import random
import math
import numpy as np
from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix
def loadCsv(filename):
        lines = csv.reader(open(filename, 'r'))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset

def splitData(dataset, sRatio):
        trainSize = int(len(dataset) * sRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
                index = random.randrange(len(copy))
                trainSet.append(copy.pop(index))
        return [trainSet, copy]

def ClassData(dataset):
        classdivision = {}
        for i in range(len(dataset)):
                vector = dataset[i]
                if (vector[-1] not in classdivision):
                        classdivision[vector[-1]] = []
                classdivision[vector[-1]].append(vector)
        print(classdivision)
        return classdivision

def mean(numbers):
        return sum(numbers)/float(len(numbers))

def stdev(numbers):
        avg = mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

def process(dataset):
        foreveryclass=[]
        for attribute in zip(*dataset):
                x = mean(attribute)
                y = stdev(attribute)
                foreveryclass.append([x,y])
        del foreveryclass[-1]
        return foreveryclass

def summarizeByClass(dataset):
        divided = ClassData(dataset)
        #print(separated)
        ProcessValues = {}  # a dictionary to store mean stdev of all attributes classwise
        for classValue, instances in divided.items(): #returns a list of key, value pairs for tuples
                ProcessValues[classValue] = process(instances)
        #print(ProcessValues)
        return ProcessValues

def calculateProbability(x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(ProcessValues, inputVector):
        probabilities = {}
        for classValue, classSummaries in ProcessValues.items():
                probabilities[classValue] = 1
                for i in range(len(classSummaries)):
                        mean, stdev = classSummaries[i]
                        x = inputVector[i]
                        probabilities[classValue] *= calculateProbability(x, mean, stdev)
        #print(probabilities)
        return probabilities

def predict(ProcessValues, inputVector):
        probabilities = calculateClassProbabilities(ProcessValues, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
                if bestLabel is None or probability > bestProb:
                        bestProb = probability
                        bestLabel = classValue
        return bestLabel

def getPredictions(ProcessValues, testSet):
        predictions = []
        y_true = []
        for i in range(len(testSet)):
                result = predict(ProcessValues, testSet[i])
                predictions.append(result)
        #print(predictions)
        for i in range(len(testSet)):
                vector=testSet[i]
                y_true.append(vector[-1])
        #print(y_true)
        return [y_true, predictions]

def getAccuracy(testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
                if testSet[i][-1] == predictions[i]:
                        correct += 1
        return (correct/float(len(testSet))) * 100.0

def main():
        filename = 'data.csv'
        file = 'Databalancedtest.csv'
        sRatio = 0.80
        dataset = loadCsv(filename)
        trainingSet, testSet = splitData(dataset, sRatio)
        #print('Split {} rows into train={} and test={} rows'.format(len(dataset), len(trainingSet), len(testSet)))
        # prepare model
        ProcessValues = summarizeByClass(trainingSet)
        # test model
        y_true, predictions = getPredictions(ProcessValues, testSet)
        #print('True Classes of test dataset: {}\n'.format(y_true))
        #print('\nPredicted Classes : {}\n'.format(y_true))
        cm = confusion_matrix(y_true, predictions)
        #for i in range(6):
                #for j in range(6):
                        #print('{:4}'.format(cm[i][j])),
                #print
        print('\n\n Confusion Matrix \n')
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
        #confusionmatrix = np.matrix(cm)
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
        #accuracy = getAccuracy(testSet, predictions)
        #print('Accuracy: {}%'.format(accuracy))

main()
