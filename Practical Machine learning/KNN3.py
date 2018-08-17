# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
from sklearn import preprocessing, cross_validation, neighbors


def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less that total voting groups !!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]] #we create a list of groups up to the k-th distances
    vote_result = Counter(votes).most_common(1)[0][0] #Counter() allows to count occurrences without using regex - most_common(1) returns the 1st most common element
    confidence = Counter(votes).most_common(1)[0][1] / k
#    print(vote_result,confidence)
    
    return vote_result,confidence

accuracies=[]

for i in range(25):
    df = pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?', -99999, inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data = df.astype(float).values.tolist()
    #shuffle the data
    random.shuffle(full_data)
    
    test_size = 0.2
    train_set = {2:[], 4:[]} #2 column empty list and 4 column empty list
    test_set = {2:[],4:[]}
    train_data = full_data[:-int(test_size*len(full_data))] # everything up to the last 20% of the data
    test_data = full_data[-int(test_size*len(full_data)):] #last 20% of the data
    
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) #if the last number is 2 => append in the 2 column; else if the last is 4 => 4 column
        
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    
    correct = 0
    total = 0
    
    for group in test_set:
        for data in test_set[group]:
            vote,confidence=k_nearest_neighbors(train_set, data, k=4)
            if group == vote:
                correct+=1
#            else:
#                print(confidence)
            total+=1
#    print('Accuracy:',correct/total)
    accuracies.append(correct/total)
    
print(sum(accuracies)/len(accuracies))