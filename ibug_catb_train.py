##### Script for Muon Energy Reconstruction in the water tank
#import Store
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
#import seaborn as sns
import catboost
from catboost import CatBoostRegressor, Pool
from ibug import IBUGWrapper

#-------- File with events for reconstruction:
#--- evts for training:
infile = "vars_Ereco_train.csv"
#----------------

folder = 'visualisations/'
num_trees = 800

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

E_threshold = 2.
E_low=0
E_high=2000
div=100
bins = int((E_high-E_low)/div)
print('bins: ', bins)

print( "--- opening file with input variables!") 
#--- events for training ---
filein = open(str(infile))
print("evts for training in: ",filein)
df00=pd.read_csv(filein)
df0=df00[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','TrueTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
#dfsel=df0.loc[df0['neutrinoE'] < E_threshold]
dfsel=df0.dropna()
print("df0.head(): ", df0.head())

#print to check:
print("check training sample: \n",dfsel.head())
#   print(dfsel.iloc[5:10,0:5])
#check fr NaN values:
print("The dimensions of training sample ",dfsel.shape)
assert(dfsel.isnull().any().any()==False)

#--- normalisation-training sample:
#dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['TrueTrackLengthInMrd']/200., dfsel['diffDirAbs'], dfsel['recoDWallR']/152.4, dfsel['recoDWallZ']/198., dfsel['totalLAPPDs']/1000., dfsel['totalPMTs']/1000., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['TrueTrackLengthInMrd'], dfsel['diffDirAbs'], dfsel['recoDWallR'], dfsel['recoDWallZ'], dfsel['totalLAPPDs']/200., dfsel['totalPMTs']/200., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
print("check normalisation: ", dfsel_n.head())

#--- prepare training & test sample for CATB:
arr_hi_E0 = np.array(dfsel_n[['DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']])
arr3_hi_E0 = np.array(dfsel[['trueKE']])
 
#---- random split of events ----
rnd_indices = np.random.rand(len(arr_hi_E0)) < 1. #< 0.50
print(rnd_indices[0:5])
#--- select events for training/test:
arr_hi_E0B = arr_hi_E0[rnd_indices]
print(arr_hi_E0B[0:5])
arr2_hi_E_n = arr_hi_E0B #.reshape(arr_hi_E0B.shape + (-1,))
arr3_hi_E = arr3_hi_E0[rnd_indices]

#printing..
print('events for training: ',len(arr3_hi_E)) #,' events for predicting: ',len(test_data_trueKE_hi_E)) 
print('initial train shape: ',arr3_hi_E.shape) #," predict: ",test_data_trueKE_hi_E.shape)

num_trees = 1000
########### CatBoost ############
params = {
    'num_trees': num_trees,
    'depth': 2,
    'learning_rate': 0.025,
    'border_count': 128,
    #'custom_metric': 'RMSE',
    'l2_leaf_reg': 3,
    'subsample': 0.8
    #'loss_function': 'MAE',  # Use 'MAE' for mean absolute error
    #best {'border_count': 64, 'custom_metric': 'RMSE', 'depth': 4, 'iterations': 800, 'l2_leaf_reg': 1, 'learning_rate': 0.01, 'subsample': 0.8}
}

print("arr2_hi_E_n.shape: ",arr2_hi_E_n.shape)
#--- select 70% of sample for training and 30% for testing:
offset = int(arr2_hi_E_n.shape[0] * 0.7) 
arr2_hi_E_train, arr3_hi_E_train = arr2_hi_E_n[:offset], arr3_hi_E[:offset].reshape(-1)  # train sample
arr2_hi_E_test, arr3_hi_E_test   = arr2_hi_E_n[offset:], arr3_hi_E[offset:].reshape(-1)  # test sample
 
print("train shape: ", arr2_hi_E_train.shape," label: ",arr3_hi_E_train.shape)
print("test shape: ", arr2_hi_E_test.shape," label: ",arr3_hi_E_test.shape)
    
eval_set = [(arr2_hi_E_train, arr3_hi_E_train), (arr2_hi_E_test, arr3_hi_E_test)]

eval_dataset = Pool(arr2_hi_E_test,
                    arr3_hi_E_test)

print("training CatBoost...")
#model = xgb.XGBRegressor().fit(arr2_hi_E_train, arr3_hi_E_train)
model = CatBoostRegressor(**params,  verbose=False, random_seed=0).fit(arr2_hi_E_train, arr3_hi_E_train)

# extend GBRT model into a probabilistic estimator
prob_model = IBUGWrapper().fit(model, arr2_hi_E_train, arr3_hi_E_train, X_val=arr2_hi_E_test, y_val=arr3_hi_E_test)

# Save the model to disk
filename = 'models/ibug_catboost.sav'
pickle.dump(prob_model, open(filename, 'wb'))


# Calculate Mean Squared Error
mse = mean_squared_error(arr3_hi_E_test, model.predict(arr2_hi_E_test)) #[:,0]
print("MSE: %.4f" % mse)
print("events at training & test samples: ", len(arr_hi_E0))
print("events at train sample: ", len(arr2_hi_E_train))
print("events at test sample: ", len(arr2_hi_E_test))
