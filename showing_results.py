import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pathlib
import plotly.express as px


working_dir = './results_log/rebuttal'

col_names = ["Model", "init", "optimizer", "ld", "lr","activation", "epochs", "loss", "ari" ]

collected_resutls_file= pathlib.Path(working_dir+"/results_avg_rebuttal.csv")

if collected_resutls_file.exists():
    result_avg = pd.read_csv(collected_resutls_file,delimiter=",",header=None)
    result_avg.columns= col_names
else:
    idx = 0
    for filename in os.listdir(working_dir):
        with open(os.path.join(working_dir, filename), 'r') as f:
            result_tmp = pd.read_csv(f,delimiter=",",header=None)
        tmp_col_names = col_names.copy()
        for i in  range(1,4):
            tmp_col_names[-i] = tmp_col_names[-i]+"_"+str(idx)
        result_tmp.columns = tmp_col_names
        if (not idx):
            results=result_tmp.copy()
        else:
            results = pd.merge(results, result_tmp, how='outer', on=["Model", "init", "optimizer", "ld", "lr", "activation"], sort=True)

        idx+=1
    result_avg = results.iloc[:,0:9]
    result_avg.columns=col_names
    result_avg.iloc[:,6] = results.iloc[:,6:results.shape[1]:3].mean(axis=1)
    result_avg.iloc[:,7] = results.iloc[:,7:results.shape[1]:3].mean(axis=1)
    result_avg.iloc[:,8] = results.iloc[:,8:results.shape[1]:3].mean(axis=1)

    np.savetxt(working_dir+"/results_avg_rebuttal.csv",result_avg,delimiter=",",fmt = "%s")

sns.scatterplot(data = result_avg,x='ari',y='validation_loss',hue='Model')
plt.cla()
plt.savefig("./Figures_folder/all_models.png")
sns.set(style="ticks")
sns.set_theme(style="whitegrid")
sns.boxplot(data=result_avg, y='Model', x='ari', orient='h')
plt.savefig("./Figures_folder/all_models_box.png")
plt.cla()



np.nanquantile(result_avg.iloc[:,-2],0.75)
plt.ylim(0,2)

model_list = {'VanillaVAE','IWAE','BetaVAE','BetaTCVAE','CategoricalVAE','DIPVAE'}

for i in model_list:
    tmp= result_avg.loc[result_avg.Model== i]
    print(i)
    try:
        print(tmp.iloc[:, -1].isna().sum())
        print(tmp.iloc[:,-1].max().round(3))
        print("==============================================")
    except:
        pass


    y_lim = math.ceil(np.nanquantile(tmp.iloc[:,-2],0.75))
    y_min = math.floor((tmp.iloc[:,-2].min)())
    sns.scatterplot(data=tmp, x='ari', y='loss', hue='init')
    plt.savefig("./figures/"+i+"_init.png")
    plt.cla()
    sns.boxplot(data= tmp , y = 'init', x = 'ari' , orient = 'h')
    plt.savefig("./figures/"+i+"_init_box.png")
    plt.cla()
    plt.close()

    # For scatter plot use point size = 50 (i.e. s=50)
    sns.scatterplot(data=tmp, x='ari', y='loss', hue='optimizer')
    plt.savefig("./figures/" + i + "_opt.png")
    plt.cla()
    sns.boxplot(data=tmp, y='optimizer', x='ari', orient='h')
    plt.savefig("./figures/" + i + "_opt_box.png")
    plt.cla()
    plt.close()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='ld')
    plt.savefig("./figures/" + i + "_ld.png")
    plt.cla()
    sns.boxplot(data=tmp, y='ld', x='ari', orient='h')
    plt.savefig("./figures/" + i + "_ld_box.png")
    plt.cla()
    sns.boxplot(data=tmp, x='ld', y='loss')
    plt.savefig("./figures/" + i + "_ld_loss_box.png")
    plt.cla()
    plt.close()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='lr',palette='colorblind')
    plt.savefig("./figures/" + i + "_lr.png")
    plt.cla()
    sns.boxplot(data=tmp, y='lr', x='ari', orient='h')
    plt.savefig("./figures/" + i + "_lr_box.png")
    plt.cla()
    sns.boxplot(data=tmp, x='lr', y='loss')
    plt.savefig("./figures/" + i + "_lr_loss_box.png")
    plt.cla()
    plt.close()

    sns.scatterplot(data=tmp, x='epochs', y='loss', hue='lr',palette='colorblind')
    plt.savefig("./figures/" + i + "_epochs.png")
    plt.cla()


# Visualizing Disentanglement Routine

from matplotlib.colors import BoundaryNorm, ListedColormap
vae_1 = pd.read_csv("./disentangle_logs/default/Adam/10/0.001/VanillaVAE/disentanglement_single_test.csv")
vae_2 = pd.read_csv("./disentangle_logs/default/Adam/10/0.001/VanillaVAE/disentanglement_signature_file.csv")
vae_total = pd.concat([vae_1,vae_2],axis=1)
vae_subset = vae_total[['Dim','age_R_train','immune_R_train','metastat_R','SBS1_r','SBS2_r','SBS5_r','SBS13_r','SBS15_r','SBS40_r']]
#vae_subset = vae_total[['Dim','age_R_train','age_R_test','immune_R_train','immune_R_test','metastat_R','SBS1_r','SBS2_r','SBS5_r','SBS13_r','SBS15_r','SBS40_r','SBS45_r']]
vae_subset = vae_subset.set_index('Dim')
vae_subset = abs(vae_subset)
#subset_colname = ['Age_train','Age_test','Immune_train','Immune_test','Metastat','SBS1','SBS2','SBS5','SBS13','SBS15','SBS40','SBS45']
subset_colname = ['Age','Immune score','Metastat','SBS1','SBS2','SBS5','SBS13','SBS15','SBS40']
vae_subset.columns = subset_colname
plt.clf()
#my_colors = ['lightblue', 'red','darkred']
my_colors = ['#caebe7','#4ea7c2','#227780']
my_bound = [0,0.3,0.7,1]
my_norm = BoundaryNorm(my_bound, ncolors=len(my_colors))
sns.heatmap(vae_subset.T, linewidths=.5 , cmap=my_colors, norm= my_norm, vmin=0, vmax=1)
plt.show()


#Comparing the significance of correlation to the p-value

model_list = ['VanillaVAE','BetaVAE','BetaTCVAE','IWAE','CategoricalVAE','DIPVAE']
for y in model_list :
    print(y)
    for i in [10,20,30,50,100,200]:
        vae_1 = pd.read_csv("./disentangle_logs/default/Adam/"+str(i)+"/0.001/"+y+"/disentanglement_single_test.csv")
        vae_2 = pd.read_csv("./disentangle_logs/default/Adam/"+str(i)+"/0.001/"+y+"/disentanglement_signature_file.csv")
        vae_total = pd.concat([vae_1,vae_2],axis=1)
        vae_subset = vae_total[['Dim','age_pVal_train','immune_pVal_train','Metastat_pVal','SBS1_Pval','SBS2_Pval','SBS5_Pval','SBS13_Pval','SBS15_Pval','SBS40_Pval']]
        vae_subset = vae_subset.set_index('Dim')
        subset_colname = ['Age','Immune score','Metastat','SBS1','SBS2','SBS5','SBS13','SBS15','SBS40']
        vae_subset.columns = subset_colname

        vae_subset_2 = vae_total[['Dim','age_R_train','immune_R_train','metastat_R','SBS1_r','SBS2_r','SBS5_r','SBS13_r','SBS15_r','SBS40_r']]
        vae_subset_2 = vae_subset_2.set_index('Dim')
        subset_colname = ['Age','Immune score','Metastat','SBS1','SBS2','SBS5','SBS13','SBS15','SBS40']
        vae_subset_2.columns = subset_colname
        vae_subset_2= abs(vae_subset_2)
        print(i)
        print(vae_subset_2[vae_subset<0.01].mean())
