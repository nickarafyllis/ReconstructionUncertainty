##### Script to average CBU and IBUG on CatBoosy
#import Store
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import uncertainty_toolbox as uct

folder = "Ibug_cbu/"

infileI = 'CatBoost_ibug/' + "Ereco_results.csv"
infileC = 'cbu/' + "Ereco_results.csv"

dfI = pd.read_csv(infileI)
dfC = pd.read_csv(infileC)

df1 = dfI['MuonEnergy']
df2 = np.average([dfI['RecoE'],dfC['RecoE']],axis=0)
df3 = np.average([dfI['Uncertainty'], np.sqrt(dfC['KnowledgeUncertainty']+dfC['DataUncertainty'])],axis=0)

# Create DataFrames from the individual values
df2 = pd.DataFrame({'RecoE': df2})
df3 = pd.DataFrame({'Uncertainty': df3})
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