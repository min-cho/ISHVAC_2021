# -*- coding: utf-8 -*-
"""
Created on Sep 06 - dev ver4.0
@author: MR004CHM
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn import linear_model

tf.random.set_seed(777)
os.chdir('C:\\Users\\MR004CHM\\Desktop\\TFcode\\2021-fdd')

######################################################################################################################
#%% DEFINE FUNCTIONS ####

def filter_optime(rawdata, week_info_csv):
    # weekdays and weekends (ref: 2017y)
    week_info   = pd.read_csv(week_info_csv)
    weekday     = week_info[['week']].to_numpy(dtype='int32')
    weekdaymins = np.repeat(weekday,1440)
    #operation hours: 07-18
    workhour     = np.concatenate((np.zeros(60*7,dtype='int32'),np.ones(60*11,dtype='int32'),np.zeros(60*6,dtype='int32')),axis=None)
    workhourmins = np.tile(workhour, 365)
    #filter : Zeros during deactivated
    optime = weekdaymins*workhourmins
    # APPLY filter
    # CHECK: in 2017, weekdays are 260 out of 365
    data = rawdata
    data['operation']=optime
    data = data[data['operation'] != 0] 
    return(data)

def filter_startup(rawdata, cutoff):
    #cut data at start-up period (IT IS NOISY DATA)
    ndays         = len(rawdata)//(60*11)
    startuptime   = np.concatenate((np.zeros(cutoff,dtype='int32'),np.ones(60*11-cutoff,dtype='int32')),axis=None)
    startupfilter = np.tile(startuptime, ndays)
    # APPLY filter
    # CHECK: in 2017, weekdays are 260 out of 365
    data = rawdata
    data['startups']= startupfilter
    data = data[data['startups'] != 0] 
    return(data)

def scaler_set(data_ref):
    global scalerX
    scalerX = MinMaxScaler(feature_range=(0.001,1))     ## to avoid ZERO inputs
    scalerX.fit(data_ref[:,0:(n_input)])                ## to cover the WHOLE SEASONS
 
def train_NN(data_ready, n_epoch):
    global records_train, nn_hist
    data_train = data_ready[train_start_day*daymin:test_start_day*daymin,0:(n_input+1)]
    X_var = data_train[:,0:(n_input)]
    Y_var = data_train[:,(n_input)]/1000000
    trainX, valX, trainY, valY = train_test_split(X_var, Y_var, test_size=0.1, shuffle=True, random_state=777)
    trainX = scalerX.transform(trainX)
    valX   = scalerX.transform(valX)

    model = keras.Sequential()
    model.add(layers.Dense(32, input_dim=(n_input), activation="sigmoid", name="layer1"))
    model.add(layers.Dense(32,activation="sigmoid", name="layer2"))
    model.add(layers.Dense(1, name="output"))
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    nn_hist = model.fit(trainX, trainY, epochs=n_epoch, batch_size=1000)        #### BATCH SIZE IS TUNED HYPER-PARAMETER
    model.reset_states()
    return(model)


def predict_NN_RMSE(NN_MODEL,data_ready):
    global RMSE_mat
    model = NN_MODEL
    data_test  = data_ready[test_start_day*daymin:(test_start_day+test_len)*daymin , 0:(n_input+1)]
    testX = data_test[:,0:n_input]
    testY = data_test[:,n_input]/1000000
    testX = scalerX.transform(testX)
    y_pred  = model.predict(testX)
    y_pred[y_pred < 0] = 0                  # non-negative prediction
    y_true  = testY.reshape(-1,1)
    
    result_RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
    result_RMSE = round(result_RMSE,3)
    RMSE_mat.iloc[i//5,j//5] = result_RMSE  # matrix index start from 0

def predict_NN_RMSLE(NN_MODEL,data_ready):
    global RMSLE_mat
    model = NN_MODEL
    data_test  = data_ready[test_start_day*daymin:(test_start_day+test_len)*daymin , 0:(n_input+1)]
    testX = data_test[:,0:n_input]
    testY = data_test[:,n_input]/1000000
    testX = scalerX.transform(testX)
    y_pred  = model.predict(testX)
    y_pred[y_pred < 0] = 0                   # non-negative prediction
    y_true  = testY.reshape(-1,1)
    
    result_RMSLE = np.sqrt(mean_squared_log_error(y_true,y_pred))
    result_RMSLE = round(result_RMSLE,3)
    RMSLE_mat.iloc[i//5,j//5] = result_RMSLE  # matrix index start from 0


def records_save(table):
    now = datetime.datetime.now()
    run_date = now.strftime('%m%d_%H%M')
    filename  = "RMSLE_matrix_" + run_date +".csv"
    table.to_csv(filename, index=False)
    
def records_save_RMSE(table):
    now = datetime.datetime.now()
    run_date = now.strftime('%m%d_%H%M')
    filename   = "RMSE_matrix_" + run_date +".csv"
    table.to_csv(filename, index=False)
    
def records_save_baseline(table):
    now = datetime.datetime.now()
    run_date = now.strftime('%m%d_%H%M')
    filename   = "RMSLE_matrix_" + "baseline_" + run_date +".csv"  ### test for normal condition data
    table.to_csv(filename, index=False)
    
def records_save_baseline_RMSE(table):
    now = datetime.datetime.now()
    run_date = now.strftime('%m%d_%H%M')
    filename   = "RMSE_matrix_" + "baseline_" + run_date +".csv"  ### test for normal condition data
    table.to_csv(filename, index=False)

######################################################################################################################
#%% Variable Setting ####

# INPUT VARIABLES
startup   = 30                # filter for start-up period to remove the simulation noise
daymin    = 60*11-startup     # revised daily time length
n_input   = 4                 # number of input variables

# Length of sliding-window _ for test
test_len= 5
n_iter  = 260//test_len

# Record Initialization
#records_train = pd.DataFrame({'train_score':[0], 'validation_score':[0], 'train_date':[0]})
#records_test  = pd.DataFrame({'test_score':[0],'train_date':[0], 'test_date':[0]})
RMSE_mat  = pd.DataFrame(np.zeros((n_iter, n_iter)))
RMSLE_mat = pd.DataFrame(np.zeros((n_iter, n_iter)))

######################################################################################################################
#%% FAULT_CASE : FOULING FAULT in 2nd Year

### Data Preparation

DATA_NAME = 'chiller_unfaulted_HK.csv'
# Data Import
data_from_csv = pd.read_csv(DATA_NAME)
data_from_csv.set_axis(['OA_temp','inlet_flowrate','inlet_temp','outlet_temp','electric_energy'],axis='columns', inplace=True)
# Data Filter : weekdays & operation hour(07-18)
data_ref = filter_optime(data_from_csv, 'weekend.csv')
data_ref = filter_startup(data_ref, startup)
data_ref = data_ref.to_numpy()
#optimesheet  = ("filtered_" + DATA_NAME)       #TO EXPORT .CSV
#data_optime.to_csv(optimesheet, index=False)   #TO EXPORT .CSV

DATA_NAME = 'chiller_fouling_MAC.csv'
# Data Import
data_from_csv = pd.read_csv(DATA_NAME)
data_from_csv.set_axis(['OA_temp','inlet_flowrate','inlet_temp','outlet_temp','electric_energy'],axis='columns', inplace=True)
# Data Filter : weekdays & operation hour(07-18)
data_foul = filter_optime(data_from_csv, 'weekend.csv')
data_foul = filter_startup(data_foul, startup)
data_foul = data_foul.to_numpy()
#optimesheet  = ("filtered_" + DATA_NAME)       #TO EXPORT .CSV
#data_optime.to_csv(optimesheet, index=False)   #TO EXPORT .CSV


data_tot   = np.vstack([data_ref, data_foul])
scaler_set(data_tot)


### Train and Prediction of Neural Network

for i in range(0,260,5):
    train_start_day = i
    test_start_day = train_start_day+260   # number of workdays in the calendar is 260
    nn_model_HKMAC = train_NN(data_tot, 250)
    #nn_model.save('model_normal_HK.h5')
    for j in range(i,260,5):
        test_start_day = 260 + j
        predict_NN_RMSE(nn_model_HKMAC,data_tot)
        predict_NN_RMSLE(nn_model_HKMAC,data_tot)
        

records_save(RMSLE_mat)
records_save_RMSE(RMSE_mat)

#######################################################
