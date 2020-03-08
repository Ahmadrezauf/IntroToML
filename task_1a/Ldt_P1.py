#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


# In[8]:


dat = pd.read_csv("./data/train.csv", header = 0)
y = dat['y']
X = dat.iloc[:,2:15]

ridgr_parameter = [0.01, 0.1, 1, 10, 100]


# In[33]:


print(X)


# In[38]:


results = []
for i in range(0 , len(ridgr_parameter)):
    print(ridgr_parameter[i])
    ridge_ridgr = linear_model.Ridge(alpha = ridgr_parameter[i], normalize= False, fit_intercept = True, max_iter=100000)

    cv_results = cross_validate(ridge_ridgr, X, y, cv = 10, scoring = 'neg_mean_squared_error')
    results.append(np.sqrt(np.mean(cv_results['test_score']) * (-1)))
print(results)

np.savetxt("results.csv", results, delimiter=",")

