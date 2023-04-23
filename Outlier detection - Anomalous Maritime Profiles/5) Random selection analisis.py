#import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os, time

df = pd.read_csv ("Results/_Anomalies/Human analized anomalies/_Anomaly_scores_All vessels.csv") # read the csv file with all data


OD_clasifier = 'Ensemble_all_Clf'
OD_df = df[['Human expertise',OD_clasifier]].copy()
OD_df = OD_df.round(0)

# Calculate performance for the OD
tp = 0; fp = 0; tn = 0; fn = 0;

for index, instance in OD_df.iterrows():
    
    actual = instance['Human expertise']
    predicted  = instance[OD_clasifier]
    
    if actual == 0 and predicted == 0:
        tn = tn + 1
    elif actual == 0 and predicted == 1:
        fp = fp + 1
    elif actual == 1 and predicted == 0:
        fn = fn + 1
    elif actual == 1 and predicted == 1:
        tp = tp + 1

tpr = tp/(tp+fn)
fpr = fp/(tn+fp)

tpr_fpr_ratio = tpr/fpr

print('TP: {}'.format(tp))
print('FP: {}'.format(fp))
print('TN: {}'.format(tn))
print('FN: {}'.format(fn))

print('TPR: {}'.format(tpr))
print('FPR: {}'.format(fpr))
print('TPR/FPR: {}'.format(tpr_fpr_ratio))



classiffiers = ['KNN_Clf','IForest_Clf','ROD_Clf','PCA_Clf','COF_Clf','ABOD_Clf','MCD_Clf',
                'LODA_Clf','LOF_Clf','CBLOF_Clf','ECOD_Clf','SOS_Clf',
                'DeepSVDD_Clf','OCSVM_Clf','ALAD_Clf','Ensemble_all_Clf','Ensemble_top_5_Clf']


for clf in classiffiers:
    
    OD_clasifier = classiffiers
    OD_df = df[['Human expertise',OD_clasifier]].copy()
    
    OD_df = OD_df.round(0)
    
    OD_df = OD_df.loc[OD_df[OD_clasifier] == 1]
    human_df = df[['Human expertise']].copy()

    RESULTS = pd.DataFrame(columns=['Random pickups', 'OD TP', 'Human TP', 'Improvment'])

    for no_pickups in range (1,OD_df.shape[0]+1):
        
        no_of_trials = 1000
        
        average_OD_TP = 0
        average_human_TP = 0
        
        sum_OD_tp = 0
        sum_human_tp = 0
        
        for trial in range (0,no_of_trials+1):
            
            selected_OD = OD_df.sample(no_pickups)
            selected_human = human_df.sample(no_pickups)

            try:
                sum_OD_tp = sum_OD_tp + selected_OD['Human expertise'].value_counts()[1]
            except:
                print("OD recorded no TP")
            
            try:
                sum_human_tp = sum_human_tp + selected_human['Human expertise'].value_counts()[1]
            except:
                print("Human recorded no TP")
            
        average_OD_TP = sum_OD_tp/trial
        average_human_TP = sum_human_tp/trial
        
        improvment = (average_OD_TP-average_human_TP)/average_human_TP
        
        new_row = {'Random pickups':no_pickups, 'OD TP':average_OD_TP, 'Human TP':average_human_TP,'Improvment':improvment}
        RESULTS = RESULTS.append(new_row, ignore_index=True)

    RESULTS.to_csv('Results/_Anomalies/Human analized anomalies/Results/OD/{}.csv'.format(clf), index=False)



