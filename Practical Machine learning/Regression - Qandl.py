# -*- coding: utf-8 -*-

#practical machine learning - regression

import pandas as pd
import numpy as np
import quandl
import math
import datetime
from sklearn import preprocessing #in order to scale our data between -1 and +1
from sklearn import cross_validation #to create our training and testing examples and shuffle our data
from sklearn import svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

#we take adjusted prices taking account of the stock splits
df1=quandl.get('WIKI/GOOGL',authtoken='g7XBjuazMArxdD6ULdxU')

#========================Defining our features ====================================
df = df1.loc[:,['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']] #.loc[row_indexer,col_indexer] = value

#high - low percent : percent of volatility
df.loc[:,'HL_PCT'] = (df.loc[:,'Adj. High'] - df.loc[:,'Adj. Close']) / (df.loc[:,'Adj. Close']) * 100
#daily percent change
df.loc[:,'PCT_change'] = (df.loc[:,'Adj. Close'] - df.loc[:,'Adj. Open']) / (df.loc[:,'Adj. Open']) * 100
df = df.loc[:,['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#==========================Defining our label======================================
forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)

#we're trying to predict out 10% of the dataframe
#take 10% of the length of our dataframe
forecast_out = int(math.ceil(0.01*len(df))) 

#we want the label column for each row to be adjusted close price 10% of total days in the future
df['label']=df[forecast_col].shift(-forecast_out) #shift the rows up of 35 days

#============================Training and testing===================================
#define our features X
X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X) #we're scaling X before we feed it in the classifier - !!! Make sure X and y are the same size !!!
X_lately = X[-forecast_out:] #take the last 35 days of features X
X=X[:-forecast_out]


df.dropna(inplace=True)
#define our label y
y=np.array(df['label']) 
#now we're creating our training and testing set
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2) #shuffle the data and takes 20% of our data
#choose a classifier
clf = LinearRegression(n_jobs=-1) #n_jobs is about how many parrallel threads launched to do the calculation
#clf = svm.SVR(kernel='poly') #with support vector regression - SVR
clf.fit(X_train,y_train)

#================================Save in pickle and re-use saved data===============
#saving in a pickle file - practical for scaling : save your prediction file in a pickle file to dump in a cloud server and make predictions for cheap money
with open('linearRegression.pickle','wb') as f:
    pickle.dump(clf,f)

#to use the classifier :
pickle_in = open('linearRegression.pickle','rb')
clf = pickle.load(pickle_in)
#================================Save in pickle and re-use saved data===============


#check how good the prediction is made on test set
accuracy=clf.score(X_test,y_test)
#print(accuracy)


#==============================Forecasting and predicting============================
forecast_set = clf.predict(X_lately) #pass an array of values to predict (here : X_lately) - calculate and predict the next 35 days of stock prices
print(forecast_set, accuracy, forecast_out)

df['Forecast']=np.nan

last_date=df.iloc[-1].name #grab the last date index
last_unix=last_date.timestamp() #convert date in seconds
one_day=86400 #86400 seconds in a day
next_unix= last_unix + one_day #next day in seconds

for i in forecast_set: #add forecast infos at the end of our array
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] + [i] 
    #creates new rows of dates with nan values for all columns features except for "Forecast" column


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()








