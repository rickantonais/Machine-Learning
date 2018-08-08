# -*- coding: utf-8 -*-


import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True) #most algorithm will recognize -99999 as an outlier data
df.drop(['id'],1,inplace=True)


#=================================Feature and label definitions =======================
X=np.array(df.drop(['class'],1)) #take all columns except the class
y=np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#===================================K-nearest neighbors model training=================
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)


example_measures = np.array([[4,2,1,1,1,2,3,2,1]]) #add a [] bracket so we can use the len() function
example_measures = example_measures.reshape(len(example_measures),-1) #need to reshape the numpy array in order to feed the scikit-learn model


prediction = clf.predict(example_measures)
print(prediction)



