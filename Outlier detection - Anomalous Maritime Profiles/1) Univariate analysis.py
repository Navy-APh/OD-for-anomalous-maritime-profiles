# Mahalanobis demo- Project 2 
# https://towardsdatascience.com/top-3-python-packages-for-outlier-detection-2dc004be9014

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from fitter import Fitter, get_common_distributions, get_distributions

#_____________________________________________________________________________________________________________________

df = pd.read_csv ("demo_data.csv") # read the csv file with all data

#Covariance matrixec
'''
df_analized_activities = df[['Sailing', 'In Port','Waiting', 'Signal Lost','Dark Activity']]

# for alltypes of classes
availableSubclassess = df['Subclass'].unique() # extract type of subclasses
#iterate every type of subclasses
for i in range(len(availableSubclassess)):
    # select only subclass vessel type from df
    selectedSubclass = availableSubclassess[i]
    df_subclass = df[df["Subclass"] == selectedSubclass] 
    
    #select the activities that need to be included in the analize
    #df_subclass_analized_activities = df_subclass[['Sailing', 'In Port','Waiting', 'Signal Lost','Dark Activity']]
    df_subclass_analized_activities = df_subclass[['Sailing', 'In Port','Waiting', 'Signal Lost']]
    
    
    cov_matrix = np.cov(df_subclass_analized_activities.T, bias=True)
    ax = plt.axes()
    sn.heatmap(cov_matrix, annot=True, fmt='g')
    ax.set_title(availableSubclassess[i])
    plt.show()
    #print(cov_matrix)
'''    
    
    #----------------------------------------------------------------------------------
    # KDE plots
'''
activityType = ['Sailing', 'In Port','Waiting', 'Signal Lost']

df_for_seaborn = df[['Subclass', 'Sailing', 'In Port','Waiting', 'Signal Lost']]

vessels_list = ['General Cargo', 'Tug','Crude Oil Tanker', 'Oil or Chemicals Tanker']
df_for_seaborn = df_for_seaborn[df_for_seaborn["Subclass"].isin(vessels_list)]

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(55,30))
#fig.suptitle("KDE Representations", fontsize = 60)
sn.set(font_scale=4.55)

for j in range(4):
    xCor = int(j/2)
    yCor = int(j%2)
    axes[xCor,yCor].set_title(activityType[j], fontsize = 70)
    axes[xCor,yCor].set_xlabel("No. of days", fontsize = 70)
    axes[xCor,yCor].set_ylabel("Density", fontsize = 70)
    sn.kdeplot(ax=axes[xCor,yCor], data=df_for_seaborn, shade = True, color = 'blue', clip=[0,700], edgecolor="black", linewidth=7, hue = "Subclass", x = activityType[j])
    sn.kdeplot(ax=axes[xCor,yCor], data=df_for_seaborn, shade = False, clip=[0,700], linewidth=10, hue = "Subclass", x = activityType[j])
'''
        
# Distributions' fitting

df_for_fittings = df[['Subclass','Sailing', 'In Port','Waiting', 'Signal Lost']]
availableSubclassess = df_for_fittings['Subclass'].unique() # extract type of subclasses



for i in range(len(availableSubclassess)):
    
    file_object = open('results.txt', 'a')
    
    # select only subclass vessel type from df
    selectedSubclass = availableSubclassess[i]
    
    #selected subclass
    file_object.write(selectedSubclass)
    df_fitted_subclass = df_for_fittings[df_for_fittings["Subclass"] == selectedSubclass] 
    
    activityType = ['Sailing', 'In Port','Waiting', 'Signal Lost']
    
    for j in range(len(activityType)):
        
        df_subclass_fitted_activities = df_fitted_subclass[[activityType[j]]]
    
        #selected activity
        file_object.write(activityType[j])
        file_object.write(" ")
    
        # count median mean std
        count = df_subclass_fitted_activities.count()
        file_object.write("count: " + str(count))
        median = df_subclass_fitted_activities.median()
        file_object.write("median: " + str(median))
        mean = df_subclass_fitted_activities.mean()
        file_object.write("mean: " + str(mean))
        std = df_subclass_fitted_activities.std()
        file_object.write("std: " + str(std)) 
        #f = Fitter(df_subclass_fitted_activities, distributions=['gamma','lognorm', "beta", "burr","norm"], timeout=60)
        
        
        f = Fitter(df_subclass_fitted_activities, distributions=["alpha", "anglit", "arcsine", "argus", "beta", "betaprime", "bradford", "burr", "burr12", "cauchy", "chi", "chi2", "cosine", "crystalball", "dgamma", "dweibull", "erlang", "expon", "exponnorm", "exponpow", "exponweib", "f", "fatiguelife", "fisk", "foldcauchy", "foldnorm", "frechet_l", "frechet_r", "gamma", "gausshyper", "genexpon", "genextreme", "gengamma", "genhalflogistic", "geninvgauss", "genlogistic", "gennorm", "genpareto", "gilbrat", "gompertz", "gumbel_l", "gumbel_r", "halfcauchy", "halfgennorm", "halflogistic", "halfnorm",
        "hypsecant", "invgamma", "invgauss", "invweibull", "johnsonsb", "johnsonsu", "kappa3", "kappa4", "ksone", "kstwo", "kstwobign", "laplace", "levy", "levy_l", "levy_stable", "loggamma", "logistic", "loglaplace", "lognorm", "loguniform", "lomax", "maxwell", "mielke", "moyal", "nakagami", "ncf", "nct", "ncx2",
        "norm", "norminvgauss", "pareto", "pearson3", "powerlaw", "powerlognorm", "powernorm", "rayleigh", "rdist", "recipinvgauss", "reciprocal", "rice", "rv_continuous", "rv_histogram", "semicircular", "skewnorm", "t", "trapz", "triang", "truncexpon", "truncnorm", "tukeylambda", "uniform", "vonmises", "vonmises_line", "wald", "weibull_max", "weibull_min", "wrapcauchy"], timeout=60)
        

        f.fit()
        
        #best distrib, SSE, KS
        #best_distrib = f.get_best(method = 'sumsquare_error')
        #print(best_distrib)
        summary = f.summary()
        file_object.write(str(summary.head(1)))
        
    
    file_object.write(" ")
    file_object.write("_______________________________________________________________")
    file_object.close()

