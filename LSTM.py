#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


df=pd.read_csv("expanded-lbl.csv",sep=",",header=None)


# In[5]:


for x in range(40):
    df.at[x,0]=0
    df.at[x,1]=0


# In[6]:


arr = [0 for j in range(8420)]
cnt=0
j=0


# In[7]:


for x in range(0,336760):
    if((x)%40==0):
        arr[j]=cnt
        cnt=0
        j=j+1
    
    cnt=cnt+df.iloc[x,5]


# In[8]:


def split_dataset(data):
    train, test = data[0:5892], data[5892:8418]

    return train, test


# In[11]:


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


# In[12]:


train, test = split_dataset(arr)

# number of time steps
steps = 3
out = 5
features=1

# split into samples
X_train, Y_train = split_sequence(train, steps, out)
X_test, Y_test = split_sequence(test, steps, out)
X_train = X_train.reshape((Y_train.shape[0], X_train.shape[1], features))


# In[13]:


# define model
model = Sequential()
model.add(LSTM(180, activation='relu', input_shape=(steps, features)))
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse')


# In[14]:


# fit model
model.fit(X_train, Y_train, epochs=66, verbose=0)


# In[ ]:


preds=[]

for i in range(2518):
    tmp = np.array(X_test[i])
    tmp = prox.reshape((1, 3, n_features))
    
    yhat = model.predict(tmp, verbose=0)
    preds.append(yhat)


# In[ ]:


preds=np.array(preds)
preds=preds.reshape(2518,5)
Y_test=np.array(y_test)
Y_test.reshape(2518,5)


# In[ ]:


def column(matrix, i):
    return [row[i] for row in matrix]

a_preds=column(preds,4)
a_test=column(y_test,4)


# In[ ]:


rms = mean_squared_error(a_preds, a_test, squared=False)
print(rms)
mea=mean_absolute_error(a_preds, a_test)
print(mea)


# In[ ]:




