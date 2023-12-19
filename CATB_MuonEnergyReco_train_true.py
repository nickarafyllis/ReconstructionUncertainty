##### Script for Muon Energy Reconstruction in the water tank
#import Store
import numpy as np
import pandas as pd
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
#import shap
from sklearn.model_selection import train_test_split


#-------- File with events for reconstruction:
#--- evts for training:
infile = "vars_Ereco_train.csv"
#----------------

folder = 'visualisations/'
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
scaler = StandardScaler()
dfsel_scaled = scaler.fit_transform(dfsel)
dfsel_n = pd.DataFrame(dfsel_scaled, columns=dfsel.columns)

#dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['TrueTrackLengthInMrd']/200., dfsel['diffDirAbs'], dfsel['recoDWallR']/152.4, dfsel['recoDWallZ']/198., dfsel['totalLAPPDs']/1000., dfsel['totalPMTs']/1000., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
#dfsel_n = pd.DataFrame([ dfsel['DNNRecoLength']/600., dfsel['TrueTrackLengthInMrd'], dfsel['diffDirAbs'], dfsel['recoDWallR'], dfsel['recoDWallZ'], dfsel['totalLAPPDs']/200., dfsel['totalPMTs']/200., dfsel['vtxX']/150., dfsel['vtxY']/200., dfsel['vtxZ']/150. ]).T
print("check normalisation: ", dfsel_n.head())

#--- prepare training & test sample for CATB:
X = dfsel_n[['DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ']]
#X = dfsel_n[['DNNRecoLength','TrueTrackLengthInMrd','totalLAPPDs','totalPMTs','vtxZ']]
y = dfsel[['trueKE']]
 
#test print histogram of TrueTrackLengthInMrd after standard scaler
# Replace 'dfsel_n' with the appropriate DataFrame name
data = dfsel_n['TrueTrackLengthInMrd'] 
plt.hist(data, bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
plt.xlabel('True Track Length in MRD')
plt.ylabel('Frequency')
plt.title('Histogram of True Track Length in MRD')
plt.grid(True)
#plt.show()
plt.close()

#before scaling
data = dfsel['TrueTrackLengthInMrd']
plt.hist(data, bins=20, color='skyblue', edgecolor='black')  # Adjust bins as needed
plt.xlabel('True Track Length in MRD')
plt.ylabel('Frequency')
plt.title('Histogram of True Track Length in MRD')
plt.grid(True)
#plt.show()
plt.close()

########### CatBoost ############
params = {
    'iterations': 800,
    'depth': 4,
    'learning_rate': 0.01,
    'border_count': 64, 
    'custom_metric': 'RMSE',
    'l2_leaf_reg': 1,
    'subsample': 0.8
    #'loss_function': 'MAE',  # Use 'MAE' for mean absolute error
    #best {'border_count': 64, 'custom_metric': 'RMSE', 'depth': 4, 'iterations': 800, 'l2_leaf_reg': 1, 'learning_rate': 0.01, 'subsample': 0.8}
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
 
print("train shape: ", X_train.shape," label: ",y_train.shape)
print("test shape: ", X_test.shape," label: ",y_test.shape)
    
# Training CatBoost model
print("Training CatBoost model...")
net_hi_E = CatBoostRegressor(**params)
model = net_hi_E.fit(X_train, y_train)

# Save the model to disk
filename = 'models/finalized_CatBoost_model_forMuonEnergy_trueTrackLengthInMrd.sav'
pickle.dump(model, open(filename, 'wb'))

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, model.predict(X_test))
print("MSE: %.4f" % mse)
print("events at training & test samples: ", len(dfsel_n))
print("events at train sample: ", len(X_train))
print("events at test sample: ", len(X_test))

# uncomment if you need to reproduce shap value explanations
'''
#SHAP values
explainer = shap.TreeExplainer(model, feature_names=('DNNRecoLength','TrueTrackLengthInMrd','diffDirAbs','recoDWallR','recoDWallZ','totalLAPPDs','totalPMTs','vtxX','vtxY','vtxZ'))
shap_values = explainer(X_test)

#SHAP Importance
shap.plots.bar(shap_values)
#plt.savefig(folder+'shap_importances.png', bbox_inches='tight')
plt.close()

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
#plt.savefig(folder+'shap_test.png', bbox_inches='tight')
plt.close()
#shap.plots.waterfall(shap_values[0][0])

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)
#plt.savefig(folder+'shap_effects.png')
plt.close()

#plot permutation importances
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=17, n_jobs=-1)
sorted_idx = perm_importance.importances_mean.argsort()

# Print permutation importance values for each feature
for idx in sorted_idx:
    print(f"{X_test.columns[idx]}: {perm_importance.importances_mean[idx]}")
    
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Permutation Importance')
#plt.savefig(folder+'permutations_importances.png', bbox_inches='tight')
plt.close()

#log
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=17, n_jobs=-1)
sorted_idx = perm_importance.importances_mean.argsort()
# Apply logarithmic transformation to the importance values
log_importances = np.log1p(perm_importance.importances_mean)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), log_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Permutation Importance (Log Scale)')
#plt.savefig(folder + 'permutations_importances.png', bbox_inches='tight')
plt.close()
'''