# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

npoints = 50
X, Y = [], []
# class 0
X.append(np.random.uniform(low=-2.5, high=2.3, size=(npoints,)) )
Y.append(np.random.uniform(low=-1.7, high=2.8, size=(npoints,)))
# class 1
X.append(np.random.uniform(low=-7.2, high=-4.4, size=(npoints,)) )
Y.append(np.random.uniform(low=3, high=6.5, size=(npoints,)))
learnset = []

for i in range(2):
    # adding points of class i to learnset
    points = zip(X[i],Y[i])
    for p in points:
        learnset.append((p,i)) #gather points in a tuple according to its class
    colours=["b","r"]
    for i in range(2):
        plt.scatter(X[i],Y[i],c=colours[i])
    
    
    

class Perceptron:
    
    def __init__(self,input_length,weights=None):
        if weights==None:
            #we add +1 for the bias
            self.weights=np.random.random((input_length + 1))*2 - 1
        self.learning_rate=0.05
        self.bias=1
    
    @staticmethod
    def sigmoid_function(x):
        res = 1/(1+np.power(np.e,-x))
        if res < 0.5:
            return 0
        else:
            return 1
        
    def __call__(self,in_data): #when an argument is filled with the perceptron class variable call
        weighted_input = self.weights[:-1] * in_data
        weighted_sum=weighted_input.sum() + self.bias * self.weights[-1]
        return Perceptron.sigmoid_function(weighted_sum)
    
    def adjust(self,
               target_result,
               calculated_result,
               in_data):
        error=target_result - calculated_result
        for i in range(len(in_data)):
            correction=error*in_data[i]*self.learning_rate
            self.weights[i]+=correction
        correction=error*self.bias*self.learning_rate
        self.weights[-1] += correction
        
p = Perceptron(2)
for point, label in learnset:
    p.adjust(label,
             p(point),
             point)
evaluation=Counter()
for point, label in learnset:
    if p(point) == label:
        evaluation["correct"]+=1
    else:
        evaluation["wrong"]+=1

print(evaluation.most_common())

colours=["b","r"]
for i in range(2):
    plt.scatter(X[i],Y[i],c=colours[i])
XR = np.arange(-8,4)
m=-p.weights[0]/p.weights[1]
print(m)
plt.plot(XR,m*XR,label="decision boundary")
plt.legend()
plt.show()
  
            