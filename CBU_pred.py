##### Script for Muon Energy Reconstruction in the water tank
#import Store
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import uncertainty_toolbox as uct

#-------- File with events for reconstruction:
#--- evts for training:
infile1 = "vars_Ereco_train.csv"
#--- evts for prediction:
infile2 = "vars_Ereco_pred.csv"
#----------------

folder = 'cbu/'

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
filein = open(str(infile1))
print("evts for training in: ",filein)
df00=pd.read_csv(filein)
df0=df00[['totalPMTs','totalLAPPDs','TrueTrackLengthInWater','trueKE','diffDirAbs','TrueTrackLengthInMrd','recoDWallR','recoDWallZ','dirX','dirY','dirZ','vtxX','vtxY','vtxZ','DNNRecoLength']]
dfsel=df0.dropna()

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
scaler = StandardScaler()
dfsel_scaled = scaler.fit(dfsel)
dfsel_pred_n = scaler.transform(dfsel_pred)
dfsel_pred_n  = pd.DataFrame(dfsel_pred_n, columns=dfsel.columns)

#dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd']/200., dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR']/152.4, dfsel_pred['recoDWallZ']/198., dfsel_pred['totalLAPPDs']/1000., dfsel_pred['totalPMTs']/1000., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T
#dfsel_pred_n = pd.DataFrame([ dfsel_pred['DNNRecoLength']/600., dfsel_pred['TrueTrackLengthInMrd'], dfsel_pred['diffDirAbs'], dfsel_pred['recoDWallR'], dfsel_pred['recoDWallZ'], dfsel_pred['totalLAPPDs']/200., dfsel_pred['totalPMTs']/200., dfsel_pred['vtxX']/150., dfsel_pred['vtxY']/200., dfsel_pred['vtxZ']/150. ]).T

#prepare events for predicting 
evts_to_predict_n= np.array(dfsel_pred_n[['DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs', 'vtxX','vtxY','vtxZ']])
test_data_trueKE_hi_E = np.array(dfsel_pred[['trueKE']])

# load the model from disk
filename = 'models/finalized_CatBoost_model_forMuonEnergy_trueTrackLengthInMrd.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

#predicting...
print("events for energy reco: ", len(evts_to_predict_n)) 
BDTGoutput_E = loaded_model.virtual_ensembles_predict(evts_to_predict_n, prediction_type='TotalUncertainty', 
                                        virtual_ensembles_count=10,  thread_count=-1)

###
mean_preds = BDTGoutput_E[:,0] # mean values predicted by a virtual ensemble
knowledge_un = BDTGoutput_E[:,1] # knowledge uncertainty predicted by a virtual ensemble
data_un = BDTGoutput_E[:,2] # average estimated data uncertainty

print()
print('Mean:', mean_preds[:5])
print('Knowledge Uncertainty:',knowledge_un[:5])
print('Data Uncertainty:',data_un[:5])
print()
###

Y=[0 for j in range (0,len(test_data_trueKE_hi_E))]
for i in range(len(test_data_trueKE_hi_E)):
    Y[i] = 100.*(test_data_trueKE_hi_E[i]-mean_preds[i])/(1.*test_data_trueKE_hi_E[i])
#   print("MC Energy: ", test_data_trueKE_hi_E[i]," Reco Energy: ",mean_preds[i]," DE/E[%]: ",Y[i])

df1 = pd.DataFrame(test_data_trueKE_hi_E,columns=['MuonEnergy'])
df2 = pd.DataFrame(mean_preds,columns=['RecoE'])
df3 = pd.DataFrame(knowledge_un,columns=['KnowledgeUncertainty'])
df4 = pd.DataFrame(data_un,columns=['DataUncertainty'])
df_final = pd.concat([df1,df2,df3,df4],axis=1)
 
#-logical tests:
print("checking..."," df0.shape[0]: ",df1.shape[0]," len(y_predicted): ", len(mean_preds)) 
assert(df1.shape[0]==len(mean_preds))
assert(df_final.shape[0]==df2.shape[0])

#save results to .csv:  
df_final.to_csv(folder + "Ereco_results.csv", float_format = '%.3f')

Etrue = df_final['MuonEnergy']
Ereco = df_final['RecoE']
KnowledgeUncertainty = df_final['KnowledgeUncertainty']
DataUncertainty = df_final['DataUncertainty']

#uncertainties are calculated as variances, we need std (sigma) instead.

TotalUncertainty = np.sqrt(DataUncertainty + KnowledgeUncertainty)
pred_mean = np.array(Ereco)
#pred_std = np.array(Uncertainty)
pred_std = np.array(TotalUncertainty)
y =  np.array(Etrue)

# Compute all uncertainty metrics metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)
metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)

nbins=np.arange(-100,100,2)
fig,ax0=plt.subplots(ncols=1, sharey=True)#, figsize=(8, 6))
cmap = sns.light_palette('b',as_cmap=True)
f=ax0.hist(np.array(Y), nbins, histtype='step', fill=True, color='gold',alpha=0.75)
ax0.set_xlim(-100.,100.)
ax0.set_xlabel('$\Delta E/E$ [%]')
ax0.set_ylabel('Number of Entries')
ax0.xaxis.set_label_coords(0.95, -0.08)
ax0.yaxis.set_label_coords(-0.1, 0.71)
title = "mean = %.2f, std = %.2f " % (np.array(Y).mean(), np.array(Y).std())
plt.title(title)
#plt.savefig(folder + "DE_E.png")