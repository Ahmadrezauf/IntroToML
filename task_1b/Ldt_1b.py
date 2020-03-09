#!/usr/bin/env python
# coding: utf-8

# In[175]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[4]:


dat = pd.read_csv("./data/train.csv", header = 0)
y = dat['y']
X = dat.iloc[:,2:7]


# In[167]:


fm2 = np.cos(X)
fm3 = np.multiply(X, X)
fm4 = np.exp(X)
fm5 = pd.DataFrame(np.ones(X.shape[0]))

frames = [X, fm3, fm4, fm2, fm5]
dat = pd.concat(frames, axis = 1)


# In[168]:


# X.shape
# fm2.shape

# fm3.shape

# fm4.shape

# fm5.shape
num = list(range(1,22))
res = list(map(str,num)) 
res = np.core.defchararray.add("V", res)
dat.columns = res


# In[177]:





# In[178]:


ridgr_parameter = [0.001, 0.01, 0.1, 1, 1.1, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 70, 100]
coefs = []
for i in range(len(frames)):
    X_scaled = preprocessing.scale(frames[i])
    
    tmpRidge = RidgeCV(alphas= ridgr_parameter, fit_intercept= False, normalize= True, cv = 10).fit(X_scaled, y)
    print(tmpRidge.score(X_scaled, y))
    print(tmpRidge.alpha_)
    
    coefs.extend(tmpRidge.coef_)
    print(coefs)


# In[180]:


#np.savetxt("results.csv", tmpRidge.coef_, delimiter=",")
list(coefs)
np.savetxt("results.csv", coefs, delimiter=",")


# In[159]:


ridgr_parameter = [0.001, 0.01, 0.1, 1, 1.1, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 10,15]
tmpRidge = RidgeCV(alphas= ridgr_parameter, fit_intercept= False, normalize= True, cv = 10).fit(dat, y)
tmpRidge.score(dat, y)
tmpRidge.coef_
#tmp = linear_model.Ridge(alpha = ridgr_parameter[3], normalize= True, fit_intercept = False,
#                                     max_iter=100000, tol = 1e-8, solver = 'auto'))

np.savetxt("results.csv", coefs, delimiter=",")


# In[160]:


print(tmpRidge.alpha_)
tmpRidge.score(dat, y)
tmpRidge.coef_

