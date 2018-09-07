# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

image_size=28
no_of_different_labels=10
image_pixels = image_size * image_size
lr = np.arange(10)

train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist_test.csv", delimiter = ",")

fac = 255 * 0.99 +0.01
train_imgs = np.asfarray(train_data[:,1:]) / fac
test_imgs = np.asfarray(test_data[:,1:]) / fac

train_labels = np.asfarray(train_data[:,:1])
test_labels = np.asfarray(test_data[:,:1])

train_labels_one_hot=(train_labels==lr).astype(np.float)
train_labels_one_hot[train_labels_one_hot==0]=0.01
train_labels_one_hot[train_labels_one_hot==1]=0.99

test_labels_one_hot = (test_labels == lr).astype(np.float)
test_labels_one_hot[test_labels_one_hot == 0]=0.01
test_labels_one_hot[test_labels_one_hot == 1]=0.99

#for i in range(10):
#    img = train_imgs[i].reshape((28,28))
#    plt.imshow(img,cmap="Greys")
#    plt.show()
    
# Dumping data for faster reload
with open("pickled_mnist.pkl","bw") as fh:
    data = (train_imgs,
            test_imgs,
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data,fh)
    
with open("pickled_mnist.pkl","br") as fh:
    data = pickle.load(fh)
    
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

#=================Classifying the data =================

@np.vectorize
def sigmoid(x):
    return 1/(1+np.e ** -x)

activation_function=sigmoid

from scipy.stats import truncnorm

def truncated_normal(mean=0,sd=1,low=0,upp=10):
    return truncnorm((low - mean)/sd,
                     (upp - mean)/sd,
                     loc=mean,
                     scale=sd)

class NeuralNetwork:
    def __init__(self,
                 no_of_input_nodes,
                 no_of_output_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_input_nodes=no_of_input_nodes
        self.no_of_output_nodes=no_of_output_nodes
        self.no_of_hidden_nodes=no_of_hidden_nodes
        self.learning_rate=learning_rate
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        rad = 1/np.sqrt(self.no_of_input_nodes)
        X=truncated_normal(mean=0,
                           sd=1,
                           low=-rad,
                           upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_input_nodes))
        
        rad = 1/np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0,sd=1,low=-rad,upp=rad)
        self.who = X.rvs((self.no_of_output_nodes,self.no_of_hidden_nodes))
        
    def train(self, input_vector, target_vector):
        #=====================Training - Forward propagation===================
        input_vector = np.array(input_vector, ndmin=2)
        target_vector = np.array(target_vector, ndmin=2)
        
        output_vector1 = np.dot(self.wih,input_vector.T)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.who,output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector.T - output_network
        
        #=========================Backpropagation==============================
        #update weights
        temp = output_errors * output_network * (1 - output_network)
        temp = self.learning_rate * np.dot(temp,output_hidden.T)
        self.who += temp
        
        hidden_errors = np.dot(self.who.T,output_errors)
        
        tempBis = hidden_errors * output_hidden * (1 - output_hidden)
        tempBis = self.learning_rate * np.dot(tempBis,input_vector)
        self.wih += tempBis
        
    def run(self, input_vector):
        input_vector = np.array(input_vector,ndmin=2)
        output_vector = np.dot(self.wih,input_vector.T)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who,output_vector)
        output_vector = activation_function(output_vector)
        
        return output_vector
    
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10,10),int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax() #
            target = labels[i][0]
            cm[res_max, int(target)]+=1
        return cm
    
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:,label]
        return confusion_matrix[label,label] / col.sum()
    
    def recall(self, label,confusion_matrix):
        row = confusion_matrix[label,:]
        return confusion_matrix[label,label] / row.sum()
    
    def evaluate(self,data,labels):
        corrects,wrongs=0,0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects,wrongs
    
ANN = NeuralNetwork(no_of_input_nodes=image_pixels,
                    no_of_output_nodes=10,
                    no_of_hidden_nodes=100,
                    learning_rate=0.1)

for i in range(len(train_imgs)):
    ANN.train(train_imgs[i],train_labels_one_hot[i]) #entrainement du réseaux de neuronnes
for i in range(20):
    res = ANN.run(test_imgs[i])
    print(test_labels[i],np.argmax(res),np.max(res))


    
    
    
        
        
        
        
        
        
        
        
        















