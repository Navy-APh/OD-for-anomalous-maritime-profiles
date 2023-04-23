# Outliers detection - Project 2 
# https://towardsdatascience.com/top-3-python-packages-for-outlier-detection-2dc004be9014

import pandas as pd
import os, time
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from sklearn.metrics import roc_curve, auc


# Extract ROC fo all clasifiers___________________________________________________________________________________________


classiffiers = ['KNN_Clf','IForest_Clf','ROD_Clf','PCA_Clf','COF_Clf','ABOD_Clf','MCD_Clf',
                'LODA_Clf','LOF_Clf','CBLOF_Clf','ECOD_Clf','Ensemble_Clf','SOS_Clf',
                'DeepSVDD_Clf','OCSVM_Clf','ALAD_Clf'] 

ROC_data = {}

for clf in classiffiers:
    ROC_data[clf] = pd.DataFrame(data = {'Contamination': [], 'TPR': [],'FPR': [],'TPR/FPR': []})

df = pd.read_csv ("demo_data_no_tugs.csv") # read the csv file with all data

availableSubclassess = df['Subclass'].unique() # extract type of subclasses

df_subclasses = {}

for contaminationLevel in range(1,51,1):
    
    #set contamination level
    contamin = contaminationLevel/100
    
    #iterate every type of subclasses
    for i in range(len(availableSubclassess)):
        # select only subclass vessel type from df
        selectedSubclass = availableSubclassess[i]
        df_subclasses[selectedSubclass] = df[df["Subclass"] == selectedSubclass] 
        
        #select the activities that need to be included in the analize
        df_subclasses_analized_activities = df_subclasses[selectedSubclass][['Sailing', 'In Port','Waiting', 'Signal Lost','Dark Activity']]
        no_of_alogos = 0
        

        # Outliers detection - Model 1--------------------------------------------------------------------------
        from pyod.models.abod import ABOD
        abod_clf = ABOD(contamination=contamin)
        abod_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['ABOD_Clf'] = abod_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        
        # Outliers detection - Model 2--------------------------------------------------------------------------
        from pyod.models.cblof import CBLOF
        cblof_clf = CBLOF(contamination=contamin,check_estimator=False, random_state=2)
        cblof_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['CBLOF_Clf'] = cblof_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 3--------------------------------------------------------------------------
        from pyod.models.mcd import MCD
        mcd_clf = MCD(contamination=contamin, random_state=2)
        mcd_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['MCD_Clf'] = mcd_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 4--------------------------------------------------------------------------
        from pyod.models.loda import LODA
        loda_clf = LODA(contamination=contamin)
        loda_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['LODA_Clf'] = loda_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 5--------------------------------------------------------------------------
        from pyod.models.lof import LOF
        lof_clf = LOF(contamination=contamin)
        lof_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['LOF_Clf'] = lof_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 6--------------------------------------------------------------------------
        from pyod.models.iforest import IForest
        iforest_clf = IForest(contamination=contamin)
        iforest_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['IForest_Clf'] = iforest_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 7--------------------------------------------------------------------------
        from pyod.models.knn import KNN
        knn_clf = KNN(contamination=contamin)
        knn_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['KNN_Clf'] = knn_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 8--------------------------------------------------------------------------
        from pyod.models.ocsvm import OCSVM
        ocsvm_clf = OCSVM(contamination=contamin)
        ocsvm_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['OCSVM_Clf'] = ocsvm_clf.labels_
        no_of_alogos = no_of_alogos + 1

        # Outliers detection - Model 9--------------------------------------------------------------------------
        from pyod.models.pca import PCA
        pca_clf = PCA(contamination=contamin)
        pca_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['PCA_Clf'] = pca_clf.labels_
        no_of_alogos = no_of_alogos + 1

        # Outliers detection - Model 10--------------------------------------------------------------------------
        from pyod.models.cof import COF
        cof_clf = COF(contamination=contamin)
        cof_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['COF_Clf'] = cof_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 11--------------------------------------------------------------------------
        from pyod.models.alad  import ALAD 
        alad_clf = ALAD(contamination=contamin)
        alad_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['ALAD_Clf'] = alad_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 12--------------------------------------------------------------------------
        from pyod.models.deep_svdd import DeepSVDD
        deep_svdd_clf = DeepSVDD(contamination=contamin)
        deep_svdd_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['DeepSVDD_Clf'] = deep_svdd_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 13--------------------------------------------------------------------------
        from pyod.models.ecod import ECOD
        ecod_clf = ECOD(contamination=contamin)
        ecod_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['ECOD_Clf'] = ecod_clf.labels_
        no_of_alogos = no_of_alogos + 1
        
        # Outliers detection - Model 14--------------------------------------------------------------------------
        from pyod.models.sos import SOS
        sos_clf = SOS(contamination=contamin)
        sos_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['SOS_Clf'] = sos_clf.labels_
        no_of_alogos = no_of_alogos + 1

        # Outliers detection - Model 15--------------------------------------------------------------------------
        from pyod.models.rod import ROD
        rod_clf = ROD(contamination=contamin)
        rod_clf.fit(df_subclasses_analized_activities)
        df_subclasses[selectedSubclass]['ROD_Clf'] = rod_clf.labels_
        no_of_alogos = no_of_alogos + 1

        
        # Plot the enssemble classifier plots
        
        df_subclasses[selectedSubclass]['Ensemble_Clf'] = (  df_subclasses[selectedSubclass]['ABOD_Clf'] + df_subclasses[selectedSubclass]['CBLOF_Clf'] + df_subclasses[selectedSubclass]['MCD_Clf'] 
                                    + df_subclasses[selectedSubclass]['LODA_Clf'] + df_subclasses[selectedSubclass]['LOF_Clf']   + df_subclasses[selectedSubclass]['IForest_Clf']
                                    + df_subclasses[selectedSubclass]['KNN_Clf']  + df_subclasses[selectedSubclass]['OCSVM_Clf'] + df_subclasses[selectedSubclass]['PCA_Clf']
                                    + df_subclasses[selectedSubclass]['COF_Clf']  + df_subclasses[selectedSubclass]['ALAD_Clf'] + df_subclasses[selectedSubclass]['DeepSVDD_Clf']
                                    + df_subclasses[selectedSubclass]['ECOD_Clf'] + df_subclasses[selectedSubclass]['SOS_Clf']   + df_subclasses[selectedSubclass]['ROD_Clf'])/no_of_alogos
     
        df_subclasses[selectedSubclass]['Ensemble_Clf'] = df_subclasses[selectedSubclass]['Ensemble_Clf'].round(0)
        
        
    result_df = pd.DataFrame()
    for dframe in df_subclasses:
        result_df = pd.concat([result_df,df_subclasses[dframe]], axis=0)
            
    for clf in classiffiers:
        
        tp = 0; fp = 0; tn = 0; fn = 0;
        
        for index, instance in result_df.iterrows():
            
            actual = instance['Human expertise']
            predicted  = instance[clf]
            
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
        
        ROC_data[clf] = ROC_data[clf].append({'Contamination': contamin, 'TPR': tpr,'FPR': fpr, 'TPR/FPR':tpr_fpr_ratio}, ignore_index=True)
        
        
    #save csv
    for clf in classiffiers:
        filepath = "Results/_Anomalies/ROC and AUC/{}.csv".format(clf)
        ROC_data[clf].to_csv(filepath, index=False)
        
    time.sleep(10)
              
