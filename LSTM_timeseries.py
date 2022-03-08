#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf


# In[2]:


df=pd.read_csv(r'D:\Assignments\Extra\Prac\A_1\temperature.csv')


# In[3]:


df.head()


# In[ ]:


#df.plot()


# In[7]:


import numpy as np
def sample(dataset,step):
    x,y=list(),list()
    for i in range(len(dataset)):
        a=i+step
        if a> len(dataset)-1:
            break
        x1,y1=dataset[i:a],dataset[a]
        x.append(x1)
        y.append(y1)
        
    return np.array(x),np.array(y)


# In[8]:


xdata,ydata=sample(df['Min temp.'].tolist(),3)


# In[9]:


for i in range(len(df)):
    print(xdata[i],ydata[i])


# In[10]:


xdata


# In[11]:


ydata 


# In[12]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dense


# In[13]:


model=Sequential()
model.add(LSTM(50,activation='relu',input_shape=(3,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')


# In[14]:


xdata.shape


# In[15]:


xdata=xdata.reshape(362,3,1) #batchsize,timestamp,no.of features
#in cnn batchsize,length,width,no.of channels


# In[16]:


model.fit(xdata,ydata,epochs=500)


# In[17]:


xtest=np.array([[3.051210,2.74258137,2.63091493],[2.74258137, 2.63091493, 2.60292435]])


# In[18]:


xtest=xtest.reshape(2,3,1)


# In[19]:


model.predict(xtest)

