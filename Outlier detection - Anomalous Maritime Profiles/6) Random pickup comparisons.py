# Outliers detection - Project 2 
# https://towardsdatascience.com/top-3-python-packages-for-outlier-detection-2dc004be9014

import pandas as pd
import os, time
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from sklearn.metrics import roc_curve, auc
#_____________________________________________________________________________________________________________________

'''
classiffiers = ['KNN_Clf','IForest_Clf','Ensemble_top_5_Clf','ROD_Clf','PCA_Clf','COF_Clf','ABOD_Clf','MCD_Clf',
                'LODA_Clf','LOF_Clf','CBLOF_Clf','ECOD_Clf','Ensemble_all_Clf','SOS_Clf',
                'DeepSVDD_Clf','ALAD_Clf','OCSVM_Clf'] 
'''
classiffiers = ['Ensemble_all_Clf', 'KNN_Clf',
                'Ensemble_top_5_Clf','ECOD_Clf','SOS_Clf']



# Build Improvment rates and TP rates___________________________________________________________________________________________

RANDOM_data = {}
RANDOM_selected_data = {}
RANDOM_best_data = {}

# Improvement rates

x_data = 'Random pickups'
y_data = 'Improvment'

plt.rcParams.update({'font.size': 11.5})

for clf in classiffiers:
    
    #print(clf)
    
    filepath = "Results/_Anomalies/Human analized anomalies/Results/OD/{}.csv".format(clf)
    RANDOM_data[clf] = pd.read_csv(filepath)
    
    RDM_data = pd.DataFrame()
    RDM_data = RANDOM_data[clf] = pd.read_csv(filepath)
    
    #ROC_selected_data[clf] = ROC_data[clf][ROC_data[clf]['Contamination'].isin(list_of_values)]
    
    plt.plot(RDM_data[x_data], RDM_data[y_data], linestyle='-', label = clf, linewidth=2.5)
    
    average_improvment = RDM_data[y_data].mean()
    
    auc2 = auc(x=RDM_data[x_data], y=RDM_data[y_data])
    #print("{}\t{}\t{}".format(clf, auc2,average_improvment))
    

plt.xlim(0, 200)
plt.ylim(0, 0.70)
plt.xlabel("No. of random pickups")
plt.ylabel("Improvement over random chance")
plt.legend()
plt.savefig('Results/_Anomalies/Human analized anomalies/Results/OD/OD ranking/Improvments rates.png', bbox_inches='tight')


x_data = 'Random pickups'
y_data = 'OD TP'

# TP rates

plt.clf()


for clf in classiffiers:
    
    #print(clf)
    
    filepath = "Results/_Anomalies/Human analized anomalies/Results/OD/{}.csv".format(clf)
    RANDOM_data[clf] = pd.read_csv(filepath)
    
    RDM_data = pd.DataFrame()
    RDM_data = RANDOM_data[clf] = pd.read_csv(filepath)
    
    plt.plot(RDM_data[x_data], RDM_data[y_data], linestyle='-', label = clf, linewidth=2.5)
    
    average_improvment = RDM_data[y_data].mean()
    
    #auc2 = auc(x=RDM_data[x_data], y=RDM_data[y_data])
    #print("{}\t{}\t{}".format(clf, auc2,average_improvment))
    
plt.plot(RDM_data[x_data], RDM_data['Human TP'], linestyle='-', label = 'Random chance', linewidth=2.5)

plt.xlim(0, 200)
plt.ylim(0, 65)
plt.xlabel("No. of random pickups")
plt.ylabel("No. of suspect vessels discoveries")
plt.legend()
plt.savefig('Results/_Anomalies/Human analized anomalies/Results/OD/OD ranking/TP rates.png', bbox_inches='tight')
