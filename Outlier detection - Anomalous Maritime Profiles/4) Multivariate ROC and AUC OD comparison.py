# Outliers detection - Project 2 
# https://towardsdatascience.com/top-3-python-packages-for-outlier-detection-2dc004be9014

import pandas as pd
import os, time
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from sklearn.metrics import roc_curve, auc
#_____________________________________________________________________________________________________________________


classiffiers = ['KNN_Clf','IForest_Clf','Ensemble_top_5_Clf','ROD_Clf','PCA_Clf','COF_Clf','ABOD_Clf','MCD_Clf',
                'LODA_Clf','LOF_Clf','CBLOF_Clf','ECOD_Clf','Ensemble_all_Clf','SOS_Clf',
                'DeepSVDD_Clf','ALAD_Clf','OCSVM_Clf'] 

# Build ROC and AUC___________________________________________________________________________________________

ROC_data = {}
ROC_selected_data = {}
ROC_best_data = {}

list_of_values = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]


# Plot ROC and calcultate AUC for every classifier
for clf in classiffiers:
    
    filepath = "Results/_Anomalies/ROC and AUC/{}.csv".format(clf)
    ROC_data[clf] = pd.read_csv(filepath)
    
    ROC_selected_data[clf] = ROC_data[clf][ROC_data[clf]['Contamination'].isin(list_of_values)]
    
    plt.plot(ROC_selected_data[clf].FPR, ROC_selected_data[clf].TPR, linestyle='-', label = clf)
    auc2 = auc(x=ROC_data[clf].FPR, y=ROC_data[clf].TPR)
    #print("{}\t {}".format(clf, auc2))
    
    
plt.xlim(0, 0.65)
plt.ylim(0, 0.65)
plt.plot([0,0.7],[0,0.7])
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True posivite rate (TPR)")
plt.legend()

# Extract best TPR/FRP score on TPR > 0.5______________________________________

best_contaminations = []

for clf in classiffiers:
    
    print(clf)
    
    ROC_best_data[clf] = ROC_data[clf][ROC_data[clf]['TPR']>=0.5]
    maxValues = ROC_best_data[clf]['TPR/FPR'].max()
    #print(maxValues)
    
    best_contamin = ROC_data[clf][ROC_data[clf]['TPR/FPR'] == maxValues]
    try:
        best_contaminations.append(best_contamin['Contamination'].iloc[0])
    except:
        time.sleep(1)
