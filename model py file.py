#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import Python libraries for data manipuation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

# Import the Python machine learning libraries we need
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[3]:


# Load the data set
dataset = pd.read_excel("data.xlsx")


# In[4]:


# Inspect first few rows
dataset.head(12)


# In[5]:


# Inspect data shape
dataset.shape


# In[6]:


# Inspect descriptive stats
dataset.describe()


# In[7]:


X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1:].values


# In[9]:


print(X)


# In[10]:


print(Y)


# In[12]:


X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.25, random_state=0)


# In[13]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


X_test


# In[16]:


Y_train


# In[17]:


Y_test


# In[18]:


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# In[19]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[33]:


print(y_pred)


# In[34]:


y_prob = classifier.predict_proba(X_test)[:, 1]


# In[35]:


print(y_prob)


# In[39]:


print("Accuracy is ",classifier.score(X_test,Y_test)*100,'%')


# In[ ]:




