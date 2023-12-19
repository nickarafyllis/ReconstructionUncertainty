##### Script for Muon Energy Reconstruction in the water tank
#import Store
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import uncertainty_toolbox as uct

#-------- File with events for reconstruction:
#--- evts for prediction:
infile2 = "vars_Ereco_pred.csv"
#----------------

#folder = 'XGBoost_ibug/'

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

#--- events for predicting ---
filein2 = open(str(infile2))
print(filein2)
df00b = pd.read_csv(filein2)
df0b=df00b[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','TrueTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
#dfsel_pred=df0b.loc[df0b['neutrinoE'] < E_threshold]
dfsel_pred=df0b
#print to check:
print("check predicting sample: ",dfsel_pred.shape," ",dfsel_pred.head())
#   print(dfsel_pred.iloc[5:10,0:5])
#check fr NaN values:
assert(dfsel_pred.isnull().any().any()==False)

#--- normalisation-sample for prediction:
#dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd']/200., dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR']/152.4, dfsel_pred['recoDWallZ']/198., dfsel_pred['totalLAPPDs']/1000., dfsel_pred['totalPMTs']/1000., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T
dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd'], dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR'], dfsel_pred['recoDWallZ'], dfsel_pred['totalLAPPDs']/200., dfsel_pred['totalPMTs']/200., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T

#prepare events for predicting 
evts_to_predict_n= np.array(dfsel_pred_n[['DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs', 'vtxX','vtxY','vtxZ']])
test_data_trueKE_hi_E = np.array(dfsel_pred[['trueKE']])

# load the model from disk
filename = 'models/ibug_xgboost.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

# predict mean and variance for unseen instances
location, scale = loaded_model.pred_dist(evts_to_predict_n)

# return k highest-affinity neighbors for more flexible posterior modeling
location, scale, train_idxs, train_vals = loaded_model.pred_dist(evts_to_predict_n, return_kneighbors=True)
print(location[:3])
print(scale[:3])

df1 = pd.DataFrame(test_data_trueKE_hi_E,columns=['MuonEnergy'])
df2 = pd.DataFrame(location,columns=['RecoE'])
df3 = pd.DataFrame(scale,columns=['Uncertainty'])
df_final = pd.concat([df1,df2,df3],axis=1)

#df_final.to_csv(folder + "Ereco_results.csv", float_format = '%.3f')

Etrue = df_final['MuonEnergy']
Ereco = df_final['RecoE']
Uncertainty = df_final['Uncertainty']

# TotalUncertainty = np.sqrt(DataUncertainty + KnowledgeUncertainty)
pred_mean = np.array(Ereco)
pred_std = np.array(Uncertainty)
y =  np.array(Etrue)

# Compute all uncertainty metrics metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)
metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)