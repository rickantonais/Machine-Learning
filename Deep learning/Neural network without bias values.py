# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
npoints=50
X,Y=[],[]

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


from collections import Counter
class Perceptron:
    
    def __init__(self,input_length,weights=None):
        if weights == None:
            self.weights = np.random.random(input_length) * 2 - 1
        self.learning_rate=0.1
    
    @staticmethod
    def unit_step_function(x):
        if x < 0:
            return 0
        return 1
    
    def __call__(self,input_data):
        weighted_input=self.weights * input_data
        weighted_sum=weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)
    
    def adjust(self,
               target_result,
               calculated_result,
               input_data):
        error = target_result - calculated_result
        for i in range(len(input_data)):
            correction = error * input_data[i] * self.learning_rate
            self.weights[i]+=correction

p=Perceptron(2)
for point, label in learnset:
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

colours=["r","b"]
for i in range(2):
    plt.scatter(X[i],Y[i],c=colours[i])

XR=np.arange(-8,4)
m=-p.weights[0]/p.weights[1]
print(m)

plt.plot(XR, m*XR, label="decision boundary")
plt.legend()
plt.show()



















