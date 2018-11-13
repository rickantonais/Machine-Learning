# -*- coding: utf-8 -*-

import numpy as np

#lignes de X : valeurs des variables explicatives pour une variable cible Y
#colonnes de X : feature/variable explicative

X = np.random.rand(100,1)
Y = np.linspace(0.1,5,X.shape[0])
learning_rate = 0.01
iterations = 400

#h_theta = np.dot(theta,X)
#loss_function = np.power(h_theta - Y,2)
#cost_function = np.multiply(np.divide(1,2*m),np.sum(loss_function,keepdims=True))

#============================Gradient descent==================================
def gradientDescentLinearRegression(X,Y, iterations,learning_rate):
    m = len(Y)
    X_bias = np.ones((len(X),1))
    X = np.append(X_bias,X,axis=1)
    theta = np.random.rand(X.shape[0],X.shape[1]-1)
    theta_bias = np.ones((len(theta),1))
    theta = np.append(theta_bias,theta,axis=1)
    for i in range(iterations):
        h_theta = np.dot(theta,X.T) #Linear regression : h_theta = theta0 + theta1 * X_1 + ... + thetaN * X_N
        derived_loss = h_theta - Y
        sum_derived_cost = np.dot(derived_loss,X)
        temp = learning_rate/m * sum_derived_cost
        theta = theta - temp

        print('theta at step {} : {}'.format(i,theta))
    return theta

theta = gradientDescentLinearRegression(X,Y,iterations,learning_rate)
