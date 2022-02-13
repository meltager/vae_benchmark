import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pathlib


working_dir = './results_log'

col_names = ["Model", "init", "optimizer", "ld", "lr", "epochs", "loss", "ari" ]

collected_resutls_file= pathlib.Path(working_dir+"/../results_avg.csv")

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
            results = pd.merge(results, result_tmp, how='outer', on=["Model", "init", "optimizer", "ld", "lr"], sort=True)

        idx+=1
    result_avg = results.iloc[:,0:8]
    result_avg.columns=col_names
    result_avg.iloc[:,5] = results.iloc[:,5:results.shape[1]:3].mean(axis=1)
    result_avg.iloc[:,6] = results.iloc[:,6:results.shape[1]:3].mean(axis=1)
    result_avg.iloc[:,7] = results.iloc[:,7:results.shape[1]:3].mean(axis=1)

    np.savetxt(working_dir+"/../results_avg.csv",result_avg,delimiter=",",fmt = "%s")

sns.scatterplot(data = result_avg,x='ari',y='loss',hue='Model')


np.nanquantile(result_avg.iloc[:,-2],0.75)
plt.ylim(0,2)

model_list = {'VanillaVAE','IWAE','LogCoshVAE','BetaVAE','BetaTCVAE','CategoricalVAE','DIPVAE'}

for i in model_list:
    tmp= result_avg.loc[result_avg.Model== i]
    print(i)
    try:
        print(tmp.iloc[:, -1].isna().sum())
        print(tmp.iloc[:,-1].max().round(3))
        print("==============================================")
    except:
        pass

    y_lim = math.ceil(np.nanquantile(result_avg.iloc[:,-2],0.75))
    sns.scatterplot(data=tmp, x='ari', y='loss', hue='init')
    plt.savefig("./figures_2/"+i+"_init.png")
    plt.ylim(0,y_lim)
    plt.savefig("./figures_2/" + i + "_init_75.png")
    plt.cla()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='optimizer')
    plt.savefig("./figures_2/" + i + "_opt.png")
    plt.ylim(0, y_lim)
    plt.savefig("./figures_2/" + i + "_opt_75.png")
    plt.cla()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='ld')
    plt.savefig("./figures_2/" + i + "_ld.png")
    plt.ylim(0, y_lim)
    plt.savefig("./figures_2/" + i + "_ld_75.png")
    plt.cla()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='lr',palette='colorblind')
    plt.savefig("./figures_2/" + i + "_lr.png")
    plt.ylim(0, y_lim)
    plt.savefig("./figures_2/" + i + "_lr_75.png")
    plt.cla()

    sns.scatterplot(data=tmp, x='epochs', y='loss', hue='lr',palette='colorblind')
    plt.savefig("./figures=2/" + i + "_epochs.png")
    plt.ylim(0, y_lim)
    plt.savefig("./figures_2/" + i + "_epochs_75.png")
    plt.cla()




tmp = result_avg.loc[result_avg.Model=='VanillaVAE']



print("Hello")
#results = np.loadtxt(working_dir+"/cluster_score.csv",dtype= str,delimiter=":")
#multi_idx = pd.MultiIndex.from_frame(tmp.loc[:,tmp.columns[0:4]])
#tmp_2 = pd.merge(tmp,result_tmp,how='outer',on=["Model", "init", " optimizer", "ld", "lr "],sort=True)

#result_avg.loc[(result_avg.Model=='IWAE') & (result_avg.ari.isna()),:]