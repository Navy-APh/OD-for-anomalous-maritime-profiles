# Outliers detection - Project 2 
# https://towardsdatascience.com/top-3-python-packages-for-outlier-detection-2dc004be9014

import pandas as pd
import matplotlib.pyplot as plt

#_____________________________________________________________________________________________________________________

# define 3D plotting Function
def plot_3D(df_subclass, selectedSubclass, annomalyAlgo = "NoAnomaly",pictureIndex=0):
    
    x_label = 'Waiting'
    y_label = 'Signal Lost'
    z_label = 'Sailing'
    
    threshhold = 0.5
    
    #Select the normal values
    xdataNormal = df_subclass.loc[df_subclass[annomalyAlgo] <= threshhold, x_label]
    ydataNormal = df_subclass.loc[df_subclass[annomalyAlgo] <= threshhold, y_label]
    zdataNormal = df_subclass.loc[df_subclass[annomalyAlgo] <= threshhold, z_label]
    #Select the annomalous values
    xdataAnomaly = df_subclass.loc[df_subclass[annomalyAlgo] > threshhold, x_label]
    ydataAnomaly = df_subclass.loc[df_subclass[annomalyAlgo] > threshhold, y_label]
    zdataAnomaly = df_subclass.loc[df_subclass[annomalyAlgo] > threshhold, z_label]
    #Create 3d Figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #Plot all data on axes
    ax.clear();
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_zlabel(z_label, fontsize=10)
    ax.set_title(selectedSubclass + ": " + annomalyAlgo)
    
    #Plot normal data
    ax.scatter3D(xdataNormal, ydataNormal, zdataNormal, c='blue', marker='o',s=15);
    #Plot annomalus data
    ax.scatter3D(xdataAnomaly, ydataAnomaly, zdataAnomaly, c='red', marker='o',s=25);
    #Save the figure
    filepath = "Results/Plots/{}/3D_{}_{}_{}.png".format(selectedSubclass,selectedSubclass,pictureIndex,annomalyAlgo)
    fig.savefig(filepath, dpi = 600)
#_____________________________________________________________________________________________________________________ 


df = pd.read_csv ("demo_data.csv") # read the csv file with all data

availableSubclassess = df['Subclass'].unique() # extract type of subclasses

#iterate every type of subclasses
for i in range(len(availableSubclassess)):
    # select only subclass vessel type from df
    selectedSubclass = availableSubclassess[i]
    df_subclass = df[df["Subclass"] == selectedSubclass] 
    
    #select the activities that need to be included in the analize
    df_subclass_analized_activities = df_subclass[['Sailing', 'In Port','Waiting', 'Signal Lost','Dark Activity']]
    
    #set contamination level
    contamin = 0.35
    #set picture index to 0 (for ascending order sorting of the PNGs)
    pictureIndex = 0
    
    #plot the 3D data with no anomalies
    plot_3D(df_subclass, selectedSubclass, "NoAnomaly",pictureIndex)
    pictureIndex = pictureIndex+1
    
    #plot the 3D data with Human analized vessels
    plot_3D(df_subclass, selectedSubclass, "Human expertise",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 1--------------------------------------------------------------------------
    from pyod.models.abod import ABOD
    abod_clf = ABOD(contamination=contamin)
    #abod_clf = ABOD()
    abod_clf.fit(df_subclass_analized_activities)
    df_subclass['ABOD_Clf'] = abod_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "ABOD_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 2--------------------------------------------------------------------------
    from pyod.models.cblof import CBLOF
    cblof_clf = CBLOF(contamination=contamin,check_estimator=False, random_state=2)
    #cblof_clf = CBLOF(check_estimator=False, random_state=2)
    cblof_clf.fit(df_subclass_analized_activities)
    df_subclass['CBLOF_Clf'] = cblof_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "CBLOF_Clf",pictureIndex)
    pictureIndex = pictureIndex+1 
    
    # Outliers detection - Model 3--------------------------------------------------------------------------
    from pyod.models.mcd import MCD
    mcd_clf = MCD(contamination=contamin, random_state=2)
    #mcd_clf = MCD(random_state=2)
    mcd_clf.fit(df_subclass_analized_activities)
    df_subclass['MCD_Clf'] = mcd_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "MCD_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 4--------------------------------------------------------------------------
    from pyod.models.loda import LODA
    loda_clf = LODA(contamination=contamin)
    #loda_clf = LODA()
    loda_clf.fit(df_subclass_analized_activities)
    df_subclass['LODA_Clf'] = loda_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "LODA_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 5--------------------------------------------------------------------------
    from pyod.models.lof import LOF
    lof_clf = LOF(contamination=contamin)
    #lof_clf = LOF()
    lof_clf.fit(df_subclass_analized_activities)
    df_subclass['LOF_Clf'] = lof_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "LOF_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 6--------------------------------------------------------------------------
    from pyod.models.iforest import IForest
    iforest_clf = IForest(contamination=contamin)
    #iforest_clf = IForest()
    iforest_clf.fit(df_subclass_analized_activities)
    df_subclass['IForest_Clf'] = iforest_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "IForest_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 7--------------------------------------------------------------------------
    from pyod.models.knn import KNN
    knn_clf = KNN(contamination=contamin)
    #knn_clf = KNN()
    knn_clf.fit(df_subclass_analized_activities)
    df_subclass['KNN_Clf'] = knn_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "KNN_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 8--------------------------------------------------------------------------
    from pyod.models.ocsvm import OCSVM
    ocsvm_clf = OCSVM(contamination=contamin)
    #ocsvm_clf = OCSVM()
    ocsvm_clf.fit(df_subclass_analized_activities)
    df_subclass['OCSVM_Clf'] = ocsvm_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "OCSVM_Clf",pictureIndex)
    pictureIndex = pictureIndex+1

    # Outliers detection - Model 9--------------------------------------------------------------------------
    from pyod.models.pca import PCA
    pca_clf = PCA(contamination=contamin)
    #pca_clf = PCA()
    pca_clf.fit(df_subclass_analized_activities)
    df_subclass['PCA_Clf'] = pca_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "PCA_Clf",pictureIndex)
    pictureIndex = pictureIndex+1


    # Outliers detection - Model 10--------------------------------------------------------------------------
    from pyod.models.cof import COF
    cof_clf = COF(contamination=contamin)
    #cof_clf = COF()
    cof_clf.fit(df_subclass_analized_activities)
    df_subclass['COF_Clf'] = cof_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "COF_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 11--------------------------------------------------------------------------
    from pyod.models.alad  import ALAD 
    alad_clf = ALAD(contamination=contamin)
    #alad_clf = ALAD()
    alad_clf.fit(df_subclass_analized_activities)
    df_subclass['ALAD_Clf'] = alad_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "ALAD_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 12--------------------------------------------------------------------------
    from pyod.models.deep_svdd import DeepSVDD
    deep_svdd_clf = DeepSVDD(contamination=contamin)
    #deep_svdd_clf = DeepSVDD()
    deep_svdd_clf.fit(df_subclass_analized_activities)
    df_subclass['DeepSVDD_Clf'] = deep_svdd_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "DeepSVDD_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 13--------------------------------------------------------------------------
    from pyod.models.ecod import ECOD
    ecod_clf = ECOD(contamination=contamin)
    #ecod_clf = ECOD()
    ecod_clf.fit(df_subclass_analized_activities)
    df_subclass['ECOD_Clf'] = ecod_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "ECOD_Clf",pictureIndex)
    pictureIndex = pictureIndex+1
    
    # Outliers detection - Model 14--------------------------------------------------------------------------
    from pyod.models.sos import SOS
    sos_clf = SOS(contamination=contamin)
    #sos_clf = SOS()
    sos_clf.fit(df_subclass_analized_activities)
    df_subclass['SOS_Clf'] = sos_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "SOS_Clf",pictureIndex)
    pictureIndex = pictureIndex+1

    # Outliers detection - Model 15--------------------------------------------------------------------------
    from pyod.models.rod import ROD
    rod_clf = ROD(contamination=contamin)
    #rod_clf = ROD()
    rod_clf.fit(df_subclass_analized_activities)
    df_subclass['ROD_Clf'] = rod_clf.labels_
    plot_3D(df_subclass, selectedSubclass, "ROD_Clf",pictureIndex)
    pictureIndex = pictureIndex+1

    
    # Plot the enssembles classifier plots
    df_subclass['Ensemble_all_Clf'] = round((  df_subclass['ABOD_Clf'] + df_subclass['CBLOF_Clf'] + df_subclass['MCD_Clf'] 
                                + df_subclass['LODA_Clf'] + df_subclass['LOF_Clf']   + df_subclass['IForest_Clf']
                                + df_subclass['KNN_Clf']  + df_subclass['OCSVM_Clf'] + df_subclass['PCA_Clf']
                                + df_subclass['COF_Clf']  + df_subclass['ALAD_Clf'] + df_subclass['DeepSVDD_Clf']
                                + df_subclass['ECOD_Clf'] + df_subclass['SOS_Clf']   + df_subclass['ROD_Clf'])/(pictureIndex-2))
    plot_3D(df_subclass, selectedSubclass, "Ensemble_all_Clf",pictureIndex)
    
    df_subclass['Ensemble_top_5_Clf'] = (  df_subclass['KNN_Clf'] + df_subclass['IForest_Clf'] + df_subclass['ROD_Clf'] 
                                + df_subclass['PCA_Clf'] + df_subclass['COF_Clf'])/(5)
    plot_3D(df_subclass, selectedSubclass, "Ensemble_top_5_Clf",pictureIndex)
   
    filepath = "Results/_Anomalies/Automated anomalies detection/Anomaly_scores_{}.csv".format(selectedSubclass)
    df_subclass.to_csv(filepath, index=False)

    