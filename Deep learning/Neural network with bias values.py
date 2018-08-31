# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

X,Y = [],[]
npoints=50

# class 0
X.append(np.random.uniform(low=-2.5, high=2.3, size=(npoints,)))
Y.append(np.random.uniform(low=-1.7, high=2.8, size=(npoints,)))
# class 1
X.append(np.random.uniform(low=-7.2, high=-4.4, size=(npoints,)))
Y.append(np.random.uniform(low=3, high=6.5, size=(npoints,)))

learnset=[]
for i in range(2):
    #adding points of class i to learnset
    points = zip(X[i],Y[i])
    for x in points:
        learnset.append((x,i))
colours=["b","r"]
for i in range(2):
    plt.scatter(X[i],Y[i],c=colours[i])


class Perceptron:
    
    def __init__(self,input_length,weights=None):
        if weights == None:
            # input_length + 1 because bias needs a weights as well
            self.weights=np.random.random((input_length)+1) * 2 - 1
        self.learning_rate=0.05
        self.bias=1
    
    @staticmethod
    def sigmoid_function(x):
        res = 1 / (1+np.power(np.e,-x))
        if res < 0.5:
            return 0
        else:
            return 1
    
    def __call__(self,input_data):
        weighted_input=self.weights[:-1] * input_data
        weighted_sum=weighted_input.sum() + self.bias * self.weights[-1]
        return Perceptron.sigmoid_function(weighted_sum)
    
    def adjust(self,
               target_result,
               calculated_result,
               input_data):
        error = target_result - calculated_result
        for i in range(len(input_data)):
            correction = error * input_data[i] * self.learning_rate
            print("weights :",self.weights)
            print(target_result,calculated_result,input_data,error,correction)
            self.weights[i]+=correction
        #correction of the bias
        correction = error * self.bias * self.learning_rate
        self.weights[-1]+=correction
    
p=Perceptron(2)
for point,label in learnset:
    p.adjust(label,
             p(point),
             point)
evaluation=Counter()
for point,label in learnset:
    if p(point) == label:
        evaluation["correct"]+=1
    else:
        evaluation["wrong"]+=1
print(evaluation.most_common())

colours=['b','r']
for i in range(2):
    plt.scatter(X[i],Y[i],c=colours[i])
XR=np.arange(-8,4)
m=-p.weights[0]/p.weights[1]
print(m)
plt.plot(XR,m*XR,label="decision boundary")
plt.legend()
plt.show()












        
        






