#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import zeros, newaxis
import keras
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
get_ipython().system('pip install seaborn')
import seaborn as sns
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow_addons.layers import MultiHeadAttention
from keras.layers import Bidirectional
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from keras.layers import Dense, Dropout, Flatten
import time


# In[ ]:


#load dataset
df=pd.read_csv("expanded-lbl.csv",sep=",",header=None)


# In[ ]:


for x in range(40):
    df.at[x,0]=0
    df.at[x,1]=0


# In[ ]:


arr = [0 for j in range(8420)]
cnt=0
j=0


# In[ ]:


#create sums for selected inputs / columns (5: traffic , 7: number of requests) - each sum corresponds to one time-step
for x in range(0,336760):
    if((x)%40==0):
        arr[j]=cnt
        cnt=0
        j=j+1
    
    cnt=cnt+df.iloc[x,5]


# In[ ]:


# split dataset for train - test
def split_dataset(data):
    train, test = data[0:5892], data[5892:8418]

    return train, test


# In[ ]:


# create sequences
def split_sequence(seq, steps, out):
    X, Y = list(), list()
    for i in range(len(seq)):
        end = i + steps
        outi = end + out
        if outi > len(seq)-1:
            break
        seqx, seqy = seq[i:end], seq[end:outi]
        X.append(seqx)
        Y.append(seqy)
    return np.array(X), np.array(Y)


# In[ ]:


train, test = split_dataset(arr)

# number of time steps -input
steps = 3
# number of time steps -output
out = 5
features=1

# split into samples
X_train, Y_train = split_sequence(train, steps, out)
X_test, Y_test = split_sequence(test, steps, out)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))
Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], 1))


# In[ ]:


model = Sequential()
model.add(Bidirectional(LSTM(180, activation='relu', input_shape=(steps, features))))
model.add(RepeatVector(out))
model.add(Bidirectional(LSTM(180, activation='relu',return_sequences=True)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(TimeDistributed(Dense(1)))
adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)


# In[ ]:


# fit model
model.fit(X_train, Y_train, epochs=66, verbose=0)


# In[ ]:


# produce predictions
preds=[]

for i in range(2518):
    tmp = np.array(X_test[i])
    tmp = prox.reshape((1, 3, n_features))
    
    yhat = model.predict(tmp, verbose=0)
    preds.append(yhat)


# In[ ]:


# resize predictions to be compliant with the format
preds=np.array(preds)
preds=preds.reshape(2518,5)
Y_test=np.array(Y_test)
Y_test.reshape(2518,5)


# In[ ]:


# produce time-step specific predictions
def column(matrix, i):
    return [row[i] for row in matrix]
# set the specific time-step to be examined (0 : 4)
a_preds=column(preds,0)
a_test=column(y_test,0)


# In[ ]:


#print RMSE 
rms = mean_squared_error(preds, Y_test, squared=False)
print(rms)
#print MAE
mea=mean_absolute_error(preds, Y_test)
print(mea)


# In[ ]:


#print time-step specific RMSE
rms = mean_squared_error(a_preds, a_test, squared=False)
print(rms)

#print time-step specific MAE
mea=mean_absolute_error(a_preds, a_test)
print(mea)

