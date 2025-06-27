Predictive Uncertainty in Gradient-Boosted Regression Trees : A Muon Energy Reconstruction Case Study
--

This repository hosts the code base for the report that I wrote during my internship at INPP, at NCSR Demokritos with [Evangelia Drakopoulou](https://github.com/edrakopo) and in the context of the **ANNIE Experiment** of Fermilab. 

## 📌 Overview

This project investigates the performance of various regression models (mainly Gradient Boosted Trees) in reconstructing muon energy from detector data. The focus is on evaluating and visualizing **predictive uncertainty**, using ensemble methods and statistical confidence intervals.

## 📄 Report

The full analysis, methodology and figures are detailed in the following report:

👉 [ReconstructionUncertainty_Report.pdf](./ReconstructionUncertainty_Report.pdf)


## 🛠️ Reproducability

To create a conda environment with the needed dependecies run:
```conda env create -f env.yml```
Then: 
```conda activate reco_env``` and
```pip install ibug```

To reproduce table 1 and train uncertainty models for tables 2 and 3 run ```bash reproduce_train.sh```. The results for the table 1 will be in the 'results' directory.

To reproduce the metrics of each uncertainty model run one-by-one:
```python ibug_catb_test.py```
```python ibug_xgb_test.py```
```python CBU_pred.py```
```python ibug_cbu.py```

## 📁 Structure

├── src/ # Scripts for preprocessing and modeling
├── figures/ # Plots and visualizations
├── data/ # Placeholder for input data (not included)
├── ReconstructionUncertainty_Report.pdf
└── README.md



