#!/usr/bin/env python
# coding: utf-8

# # Support vector machines

# In[36]:


# import libraries

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn import model_selection

from sklearn.impute import KNNImputer


# ## Data pre-processing

# In[3]:


# load training data

# load data from csv file
df_train_features = pd.read_csv ('train_features.csv')
df_train_labels = pd.read_csv('train_labels.csv')

# Load test data
df_test_features = pd.read_csv ('test_features.csv')


#  ### Histogram of the output labels 

# We should check for class imbalance.

# In[3]:


df_train_labels.hist()

with PdfPages("./Results/Labels_histogram.pdf") as export_pdf:
    for i in list(df_train_labels)[1:]:
        df_train_labels.hist(column = i, bins = 100)
        export_pdf.savefig()


# One can see the class imbalance problem here. Other observations:
#   * Heartrate, RRate, ABPm,  distribution is similar to a normal distribution
#   * SpO2 is like a censored normal distribution. 
#   * For all of the other features, class imbalance is an obvious problem.

# A basic strategy that could be used here: Upsample both classes! Do the upsampling efficiently, not just replicating the datapoints

# ### Train Data pre-processing

# In[4]:


# data inspection: 
#############################################
# range of the provided data?
print(df_train_features.agg([min, max]))

# Boxplotting the data
# fig2, ax2 = plt.subplots()
# ax2.set_title('BUN')
# ax2.boxplot(df_train_features.iloc[:,5], notch=True)

plt.figure(figsize=(16, 16))
ax = sns.boxplot(data = df_train_features.iloc[:,1:])
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);

# with PdfPages("./Results/Train_columns_boxplot.pdf") as export_pdf:
#     for i in list(df_train_labels)[1:]:
#         df_train_labels.hist(column = i, bins = 100)
#         export_pdf.savefig()


# In[24]:


# calculate the correlation matrix
corr = df_train_features.corr()

# plot the heatmap
plt.figure(figsize=(16, 16))
ax = sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, 
        vmin=-1, vmax=1, center=0, 
           cmap=sns.diverging_palette(20, 220, n=200))
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# ### Visualizing pattern of missing values

# In[14]:


# how much missing data? 
print("Percentage of missing values:")
print(df_train_features.isnull().sum(axis=0) / len(df_train_features))

msno.matrix(df_train_features)

# Plotting the correlation between the missing values
msno.heatmap(df_train_features)


# ### Train data pre-processing

# In[115]:


# Patient by patient pre-processing for imputation and feature generation

# get variables from train features
variables = df_train_features.columns[2:]

# get all pids, to impute and pre-process individually
train_pids = df_train_features['pid'].unique()

for pid in train_pids:
    # dataframe for this pid
    df = df_train_features.loc[df_train_features['pid'] == pid]
    
    for var in variables:
        # how many NaN are in there for this variable
        data = df[var]
        num_nan = data.isnull().sum(axis=0)
        if num_nan == 12:
            continue
        
        data = data.to_numpy()
           
        imp = SimpleImputer(missing_values = np.nan)
        train_imputed = imp.fit_transform(data.reshape(-1, 1))
    
        df_train_features.loc[df_train_features['pid'] == pid,var] = train_imputed


# In[116]:


# aggregate the time series
data_array = np.empty([len(train_pids), len(variables)])
i = 0
j = 0

for pid in train_pids:
    
    df = df_train_features.loc[df_train_features['pid'] == pid]
    j = 0
    
    for var in variables:
        # how many NaN are in there for this variable
        data = df[var]
        num_nan = data.isnull().sum(axis=0)
        if num_nan == 12:
            data_array[i, j] = np.nan
            j = j + 1
            continue
        data = data.to_numpy()
        data_array[i, j] = np.mean(data)
        j = j + 1 
        
    i = i +1


# In[117]:


# impute missing data points
#imp = SimpleImputer(strategy="mean")
imputer = KNNImputer(n_neighbors=12)
df_train_agg_imputed_features = imputer.fit_transform(data_array)
#print(df_train_agg_imputed_features)


# In[118]:


# scale the data
min_max_scaler = preprocessing.StandardScaler()
# standard_scalar = preprocessing.StandardScaler()

data_train_scaled = min_max_scaler.fit_transform(df_train_agg_imputed_features)


# In[119]:


# Visualizing the training data after imputing and aggregating

plt.figure(figsize=(16, 16))
ax = sns.boxplot(data = pd.DataFrame(data_train_scaled))
ax.set_xticklabels(
    list(df_train_features),
    rotation=90,
    horizontalalignment='right'
);


# In[120]:


# What is the correlation between the 
pd.DataFrame(data_train_scaled).corrwith(other = pd.DataFrame(df_train_agg_imputed_features), method = "spearman").transpose()


# ### Test Data pre-processing

# In[121]:


# data inspection: 
#############################################
# range of the provided data?
print(df_test_features.agg([min, max]))

# how much missing data? 
print("number of missing values:")
print(df_test_features.isnull().sum(axis=0))


# In[122]:


# aggregate data for each pid
df_test_aggregate_features = df_test_features.groupby('pid').agg('median')

#print(df_test_aggregate_features)

# collect all test pids
test_pids = list(set(df_test_features.pid))


# In[123]:


# remove time from data frame 
df_test_agg_features = df_test_aggregate_features.drop(['Time'], axis = 1)
print(df_test_agg_features)


# In[128]:


# impute missing data points
# should we impute it with the same imputer that we've used for train?
df_test_agg_imputed_features = imputer.transform(df_test_agg_features[2:])


# In[134]:


# scale test data
data_test_scaled = min_max_scaler.transform(df_test_agg_imputed_features)


# ## Fit a model & Predictdf_test_agg_imputed_features

# ### predict with support vector machine classification and use probabilities

# In[142]:


# first for the labels that have an output [0,1]

columns_1 = [test_pids]

for i in range(1, 12):
    clf = SVC(kernel = 'poly', degree = 5, class_weight = 'balanced', verbose = True)
    clf.fit(data_train_scaled, df_train_labels.iloc[:,i])
    # pred = clf.predict(df_test_agg_imputed_features)
    # columns_1.append(pred)
     
    # compute probabilites as opposed to predictions
    dual_coefficients = clf.dual_coef_    # do we have to normalize with norm of this vector ?
    distance_hyperplane = clf.decision_function(data_test_scaled)
    probability = np.empty(len(distance_hyperplane))
    for j in range(0, len(probability)):
        if distance_hyperplane[j] < 0:
            probability[j] = 1 - 1/(1 + math.exp(distance_hyperplane[j]))
        else:
            probability[j] = 1/(1 + math.exp(-distance_hyperplane[j]))
    columns_1.append(probability)


    
    distance_hyperplace_train = clf.decision_function(data_train_scaled)
    probability = np.empty(len(distance_hyperplace_train))
    for j in range(0, len(probability)):
        if distance_hyperplace_train[j] < 0:
            probability[j] = 1 - 1/(1 + math.exp(distance_hyperplace_train[j]))
        else:
            probability[j] = 1/(1 + math.exp(-distance_hyperplace_train[j]))
      
    tmp = roc_auc_score(y_score= probability, y_true= df_train_labels.iloc[:,i])
    print("ROC AUC for feature", list(df_train_labels)[i] , " : ", tmp)
    


# In[148]:


# labels that have a real value
columns_2 = []

for i in range(12, 16):
    clf_w = SVR('rbf', degree = 3)
    parameters = {'C':np.linspace(1,10, 4)}
    clf = model_selection.GridSearchCV(estimator= clf_w, param_grid = parameters, cv = 5,
                                       refit = True, scoring = 'r2', verbose = 1)
    clf.fit(data_train_scaled, df_train_labels.iloc[:,i])
    pred_train = clf.predict(data_train_scaled)
    tmp = r2_score(y_pred= pred_train, y_true=df_train_labels.iloc[:,i])
    print("R2 for feature", list(df_train_labels)[i] , " : ", tmp)
    
    pred = clf.predict(data_test_scaled)
    columns_2.append(pred)
    


# In[155]:


columns_final = columns_1 + columns_2


# ### predict with Support vector regression and then compute sigmoid function

# In[59]:


# first for the labels that have an output [0,1]

columns_1 = [test_pids]

for i in range(1,12):
    
    clf = SVR(kernel = 'poly', degree = 3, max_iter = 10000)
    clf.fit(data_train_scaled, df_train_labels.iloc[:,i])
    pred = clf.predict(data_test_scaled)
    prob = np.empty(len(pred))
    for j in range(0, len(pred)):
        prob[j] = 1 / (1 + math.exp(-pred[j]))
    columns_1.append(prob)
    
    pred_train = clf.predict(data_train_scaled)
    prob_train = np.empty(len(pred_train))
    for j in range(0, len(pred_train)):
        prob_train[j] = 1 / (1 + math.exp(-pred_train[j]))    
    tmp = roc_auc_score(y_score= prob_train, y_true= df_train_labels.iloc[:,i])
    print("ROC AUC for feature", list(df_train_labels)[i] , " : ", tmp)


# In[81]:


# labels that have a real value

columns_2 = []

for i in range(12, 16):
    clf_w = LinearSVR()
    parameters = {'C':np.linspace(0.1,10, 20)}
    clf = model_selection.GridSearchCV(estimator= clf_w, param_grid = parameters, cv = 5,
                                       refit = True, scoring = 'r2', verbose = 1, n_jobs=6)
    
    clf.fit(data_train_scaled, df_train_labels.iloc[:,i])
    print(clf.cv_results_)
    pred = clf.predict(data_test_scaled)
    columns_2.append(pred)
    
    pred_train = clf.predict(data_train_scaled)
    tmp = r2_score(y_pred= pred_train, y_true=df_train_labels.iloc[:,i])
    print("R2 for feature", list(df_train_labels)[i] , " : ", tmp)


# In[82]:


columns_final = columns_1 + columns_2


# ## Save predictions

# In[161]:


print(np.shape(columns_final))
result = pd.DataFrame(columns_final).transpose()
result.columns = list(df_train_labels)
result.to_csv('./Results/prediction.zip', index=False, float_format='%.3f', compression='zip')


# In[157]:


result.to_csv('./Results/prediction.csv', index=False, float_format='%.3f')


# In[ ]:




