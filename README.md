Predictive Uncertainty in Gradient-Boosted Regression Trees : A Muon Energy Reconstruction Case Study
--

This repository hosts the code base for the report that I wrote during my internship at INPP, at NCSR Demokritos with [Evangelia Drakopoulou](https://github.com/edrakopo) at the ANNIE experiment of Fermilab. 

To create a conda environment with the needed dependecies run:
```conda env create -f env.yml```
Then: 
```conda activate reco_env```
```pip install ibug```

To reproduce table 1 and train uncertainty models for tables 2 and 3 run ```bash reproduce_train.sh```. The results for the table 1 will be in the 'results' directory.

To reproduce the metrics of each uncertainty model run one-by-one:
```python ibug_catb_test.py```
```python ibug_xgb_test.py```
```python CBU_pred.py```
```python ibug_cbu.py```