#!/bin/bash

# train and predict track length to be used in all models
python DNNFindTrackLengthInWater_Keras_train.py
python DNNFindTrackLengthInWater_Keras_predict.py

# train point-prediction models to reproduce table 1
# results will be in results folder
python BDT_MuonEnergyReco_train_true.py
python BDT_MuonEnergyReco_pred_true.py
python XGBoost_MuonEnergyReco_train_true.py
python XGBoost_MuonEnergyReco_pred_true.py
python LGBM_MuonEnergyReco_train_true.py
python LGBM_MuonEnergyReco_pred_true.py
python NGBoost_MuonEnergyReco_train_true.py
python NGBoost_MuonEnergyReco_pred_true.py
python CATB_MuonEnergyReco_train_true.py
python CATB_MuonEnergyReco_pred_true.py

# train uncertainty models to reproduce tables 2 and 3
python ibug_catb_train.py 
python ibug_xgb_train.py 
python CBU_train.py

# then
# python ibug_catb_test.py 
# python ibug_xgb_test.py 
# python CBU_pred.py
# python ibug_cbu.py