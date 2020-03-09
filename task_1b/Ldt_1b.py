#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold


# In[3]:


dat = pd.read_csv("./data/train.csv", header = 0)
y = dat['y']
X = dat.iloc[:,2:7]


# In[4]:


fm2 = np.cos(X)
fm3 = np.multiply(X, X)
fm4 = np.exp(X)
fm5 = pd.DataFrame(np.ones(X.shape[0]))

frames = [X, fm3, fm4, fm2, fm5]
dat = pd.concat(frames, axis = 1)


# In[5]:


num = list(range(1,22))
res = list(map(str,num)) 
res = np.core.defchararray.add("V", res)
dat.columns = res


# In[23]:


ridgr_parameter = [0.001, 0.01, 0.1, 1, 1.1, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 70, 100]
coefs = []
score = []
max_score = -np.Inf
bestregressor = []

for i in range(len(frames)):
#     quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
#     X_scaled = quantile_transformer.fit_transform(frames[i])
#     min_max_scaler = preprocessing.MinMaxScaler()
#     X_scaled = min_max_scaler.fit_transform(frames[i])
    X_scaled = preprocessing.scale(frames[i])
    
    rkf = RepeatedKFold(n_splits= 10, n_repeats= 10, random_state= 1234)
    max_score = -np.Inf
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tmpRidge = RidgeCV(alphas= ridgr_parameter, fit_intercept= False, normalize= True).fit(X_train, y_train)
        tmp_score = tmpRidge.score(X_test, y_test)
        score.append(tmpRidge.score(X_test, y_test))
        if(max_score < tmp_score):
            bestregressor = tmpRidge
        
    
    coefs.extend(bestregressor.coef_)


# In[24]:


print(len(coefs))
list(coefs)
#np.savetxt("results.csv", coefs, delimiter=",")

