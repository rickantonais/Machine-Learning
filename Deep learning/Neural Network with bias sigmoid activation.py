# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

@np.vectorize
def sigmoid(x):
    return 1/(1+np.exp(-x))

activation_function=sigmoid

def truncated_normal(mean,sd,low,upp):
    return truncnorm((low-mean)/sd,(upp-mean)/sd,loc=mean,scale=sd)

class NeuralNetwork:
    
    def __init__(self,
                 no_of_input_nodes,
                 no_of_output_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None):
        self.no_of_input_nodes = no_of_input_nodes
        self.no_of_output_nodes = no_of_output_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()
        
    def create_weight_matrices(self): #initialize weights for our neural network
        if self.bias:
            bias_node = 1
        else:
            bias_node = 0
        
        rad = 1/np.sqrt(self.no_of_input_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weight_input_to_hidden = X.rvs((self.no_of_hidden_nodes,self.no_of_input_nodes + bias_node))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0,sd=1,low=-rad,upp=rad)
        self.weight_hidden_to_output = X.rvs((self.no_of_output_nodes,self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        if self.bias:
            bias_node = 1
        else:
            bias_node = 0
        
        if self.bias:
            #add the bias at the end of the vector
            input_vector = np.concatenate((input_vector,[self.bias]))
        
        input_vector = np.array(input_vector,ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        #===================== hidden layer================================
        output_vector1 = np.dot(self.weight_input_to_hidden,input_vector)
        output_vector_hidden = activation_function(output_vector1)
        
        if self.bias:
            output_vector_hidden = np.concatenate(output_vector_hidden,[self.bias])
            
        #===================== output layer================================
        output_vector1 = np.dot(self.weight_hidden_to_output,output_vector_hidden)
        output_vector_final = activation_function(output_vector1)
        
        #===================== Error function===============================
        output_errors = target_vector - output_vector_final
        
        #===================== Backpropagation and weights update===========
        #temporary variable backpropagation for HIDDEN - OUTPUT weights
        temp = output_errors * output_vector_final * (1 - output_vector_final)
        temp = self.learning_rate * np.dot(temp, output_vector_hidden.T)
        self.weight_hidden_to_output += temp
        
        #updating hidden layer's error
        hidden_errors = np.dot(self.weight_hidden_to_output.T, output_errors)
        
        #temporary variable backpropagation for INPUT - HIDDEN weights
        temp = hidden_errors * output_vector_hidden * (1 - output_vector_hidden)
        if self.bias:
            x = np.dot(temp, input_vector.T)[:-1,:]
        else:
            x = np.dot(temp, input_vector.T)
        self.weight_input_to_hidden += self.learning_rate * x
        
    def run(self, input_vector):
        if self.bias:
            #adding bias node at the end of the input_vector
            input_vector = np.concatenate((input_vector,[1]))
        input_vector = np.array(input_vector,ndmin=2).T
        output_vector = np.dot(self.weight_input_to_hidden,input_vector)
        output_vector = activation_function(output_vector)
        
        #hidden layer
        if self.bias:
            output_vector = np.concatenate(output_vector,[1])
        output_vector = np.dot(self.weight_hidden_to_output,output_vector)
        output_vector = activation_function(output_vector)
        return output_vector
    
    
class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3) ] 
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6), 
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6), 
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8) ]

labeled_data = []
for el in class1:
    labeled_data.append([el,[1,0]])
for el in class2:
    labeled_data.append([el,[0,1]])
    
np.random.shuffle(labeled_data)
data,labels = zip(*labeled_data)
labels = np.array(labels)
data = np.array(data)

simple_network = NeuralNetwork(no_of_input_nodes=2,
                               no_of_output_nodes=2,
                               no_of_hidden_nodes=10,
                               learning_rate=0.1,
                               bias=None)

#training with a 20 epoch
for _ in range(20):
    for i in range(len(data)):
        simple_network.train(input_vector=data[i],target_vector=labels[i])

for i in range(len(data)):
    print(labels[i])
    print(simple_network.run(input_vector=data[i]))









