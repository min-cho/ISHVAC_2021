# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:26:55 2021
python 3.8.5.
@author: MR004CHM
"""

import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols


os.chdir('C:\\Users\\MR004CHM\\Desktop\\TFcode\\2021-fdd')


######################################################################################################################
#%% Data Preparation

error_mat = np.loadtxt("RMSLE_matrix_0907_1954.csv",delimiter=",",skiprows=1)
# NOTE: since pd.DataFrame doesn't consider the row and col as same characterstic, it is not proper to handle the matrix.
# So import the CSV via Numpy but merge to pd.DataFrame for OLS regression

######################################################################################################################
#%% SLoW evaluation metric of Each model : row-wise slicing

len_all = 52
len_slice = 30
n_slice = len_all-(len_slice-1)

bucket     = pd.DataFrame({'week': range(1,len_slice+1,1), 'error': np.full((len_slice),9)})
slope_list = pd.DataFrame({'slope':  np.full((n_slice),9), 'p-value': np.full((n_slice),99)})

for i in range(n_slice):     # i = 0 to 9, when n_slice=10
    bucket['error']       =  error_mat[i,i:(i+len_slice)]
    model_fit             =  ols('error ~ week', data=bucket).fit()
    slope_list.iloc[i,0]  =  model_fit.params.week
    slope_list.iloc[i,1]  =  model_fit.pvalues.week
    if model_fit.pvalues.week > 0.1:
        print(("{}th model doesn't show linear trend".format(i+1)))
        
#plt.scatter(bucket['week'],bucket['error'])
#print(model_fit.pvalues.week)

# NOTE: Due to the local saddle point, regression doesn't work well in certrain condition
# we can solve this problem by finding the optimal length of slice (len_slice)
# even it has trade-off to n_slice , which means the number of stastical samples

### Plot the  bar-graph : slope values
plt.figure(dpi=1000, figsize=(8,5))
plt.bar(range(1,n_slice+1,1), (slope_list.iloc[:,0])*(10**3), color='grey', width=0.5)
plt.xlabel('Index of Detector Model',fontsize=14)
plt.ylabel('Trend of Error Regression(E-3)',fontsize=14)
plt.xticks(range(1,n_slice+1,1), range(1,n_slice+1,1));  # Set text labels and properties.

######################################################################################################################
#%% SLoW evaluation metric of Each Test-set : column-wise slicing

len_all = 52
len_slice = 30
n_slice = len_all-(len_slice-1)

bucket     = pd.DataFrame({'model':  range(1,len_slice+1,1), 'error': np.full((len_slice),9)})
slope_list = pd.DataFrame({'slope':  np.full((n_slice),9), 'p-value': np.full((n_slice),99)})

# NOTE: column-wise slicing compares the error of certain test-set from various detector models (= len_slice)
# and slice can't start from i=0 at Error Matrix , beacuse the matrix is upper triangular.


for j in range(n_slice):     # j = 0 to 9, when n_slice=10
    bucket['error']       =  error_mat[j:(j+len_slice), len_slice-1]
    model_fit             =  ols('error ~ model', data=bucket).fit()
    slope_list.iloc[j,0]  =  model_fit.params.model
    slope_list.iloc[j,1]  =  model_fit.pvalues.model
    if model_fit.pvalues.model > 0.1:
        print(("{}th test-week doesn't show linear trend".format(j+1)))
        
#plt.scatter(bucket['model'],bucket['error'])
#print(model_fit.pvalues.model)

# Again, length of slice should be determined with avoiding the local minima

### Plot the  bar-graph : slope values
plt.figure(dpi=1000, figsize=(8,5))
plt.bar(range(len_slice,len_all+1,1), -(slope_list.iloc[:,0])*(10**3), color='grey', width=0.5)
plt.xlabel('Test Set(Week)',fontsize=14)
plt.ylabel('Trend of Error Regression(E-3)',fontsize=14)
plt.xticks(range(len_slice,len_all+1,1), range(len_slice,len_all+1,1));  # Set text labels and properties.


######################################################################################################################
#%% MATRIX HEATMAP

heatdata = pd.read_csv("RMSLE_matrix_0907_1954.csv")
plt.figure(dpi=500)
sns.heatmap(heatdata)
plt.figure(figsize=(20, 15))

### VANILLA MODEL - Diagonal - 
RMSE_mat = np.loadtxt("RMSE_matrix_0907_1954.csv",delimiter=",",skiprows=1)
RMSLE_mat = np.loadtxt("RMSLE_matrix_0907_1954.csv",delimiter=",",skiprows=1)
diag_RMSE_mat = np.diag(RMSE_mat)
diag_RMSLE_mat = np.diag(RMSLE_mat)
np.savetxt("diag_RMSE.csv",diag_RMSE_mat,delimiter=",")
np.savetxt("diag_RMSLE.csv",diag_RMSLE_mat,delimiter=",")




######################################################################################################################
#%% Additional Figures - when fouling occurs

energy_data= pd.read_csv('edata_210616.csv')
unfaulted_HK= energy_data['unfaulted_HK']
unfaulted_MAC= energy_data['unfaulted_MAC']
fault_MAC= energy_data['fouling_MAC']

unfaulted_tot   = np.hstack([unfaulted_HK, unfaulted_MAC])
fault_tot       = np.hstack([unfaulted_HK, fault_MAC])

plt.figure(dpi=1000, figsize=(12,4)) 
plt.plot(unfaulted_tot, color='blue')
plt.plot(fault_tot, color='red')
plt.plot(unfaulted_HK, color='black')
plt.xticks([1,129600,260640,393120,525600,655200,786240,918720,1051200], ['Jan','Apr','Jul','Oct','Jan','Apr','Jul','Oct',"Dec"],rotation=20)  # Set text labels and properties.
plt.xlabel('Time (month)')
plt.ylabel('Energy (MJ)')
plt.legend(['2nd year_unfaulted','2nd year_faulted','1st year'], loc=[0.08, -0.3],ncol=3, fontsize=14)


######################################################################################################################
#%% Additional Figures - when fouling occurs
