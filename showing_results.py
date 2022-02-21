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
plt.savefig("./figures/all_models.png")
y_lim = math.ceil(np.nanquantile(result_avg.iloc[:, -2], 0.75))
y_min = math.floor((result_avg.iloc[:, -2].min)())
plt.ylim(0,y_lim)
plt.savefig("./figures/all_models_75.png")
plt.cla()
sns.set(style="ticks")
sns.set_theme(style="whitegrid")
sns.boxplot(data=result_avg, y='Model', x='ari', orient='h')
plt.savefig("./figures/all_models_box.png")
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
    plt.ylim(y_min,y_lim)
    plt.savefig("./figures/" + i + "_init_75.png")
    plt.cla()
    sns.boxplot(data= tmp , y = 'init', x = 'ari' , orient = 'h')
    plt.savefig("./figures/"+i+"_init_box.png")
    plt.cla()
    sns.boxplot(data= tmp , x = 'init', y = 'loss')
    plt.savefig("./figures/"+i+"_init_loss_box.png")
    plt.cla()
    plt.close()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='optimizer')
    plt.savefig("./figures/" + i + "_opt.png")
    plt.ylim(y_min, y_lim)
    plt.savefig("./figures/" + i + "_opt_75.png")
    plt.cla()
    sns.boxplot(data=tmp, y='optimizer', x='ari', orient='h')
    plt.savefig("./figures/" + i + "_opt_box.png")
    plt.cla()
    sns.boxplot(data=tmp, x='optimizer', y='loss')
    plt.savefig("./figures/" + i + "_opt_loss_box.png")
    plt.cla()
    plt.close()

    sns.scatterplot(data=tmp, x='ari', y='loss', hue='ld')
    plt.savefig("./figures/" + i + "_ld.png")
    plt.ylim(y_min, y_lim)
    plt.savefig("./figures/" + i + "_ld_75.png")
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
    plt.ylim(y_min, y_lim)
    plt.savefig("./figures/" + i + "_lr_75.png")
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
    plt.ylim(y_min, y_lim)
    plt.savefig("./figures/" + i + "_epochs_75.png")
    plt.cla()
'''
    for z in range(1,5):
        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        X = pd.get_dummies(tmp.loc[tmp.ari.isna()].iloc[:,z]).sum()
        wedges, texts = ax.pie(X, wedgeprops=dict(width=0.5), startangle=-40)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        for tmp_idx, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(str(X.index[tmp_idx]) + " = " + str(X.iloc[tmp_idx]), xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment, **kw)

        plt.savefig("./figures/" + i +"_" + tmp.columns[z]  +"_nan_circle.png")
        plt.cla()

    plt.close()

'''
tmp = result_avg.loc[result_avg.Model=='VanillaVAE']



print("Hello")
#results = np.loadtxt(working_dir+"/cluster_score.csv",dtype= str,delimiter=":")
#multi_idx = pd.MultiIndex.from_frame(tmp.loc[:,tmp.columns[0:4]])
#tmp_2 = pd.merge(tmp,result_tmp,how='outer',on=["Model", "init", " optimizer", "ld", "lr "],sort=True)

#result_avg.loc[(result_avg.Model=='IWAE') & (result_avg.ari.isna()),:]

'''
# Selecting the best model for 
for i in model_list:
    tmp = result_avg.loc[result_avg.Model == i]
    tmp = tmp.sort_values(by='ari', ascending=False)
    tmp['ari_round'] = tmp.ari.round(2)
    tmp = tmp.reset_index()
    print(tmp.iloc[tmp.loc[tmp.ari_round == tmp.ari_round.max()].loss.idxmin()])
    
    # Another Target Function
for i in model_list: 
    tmp = result_avg.loc[result_avg.Model == i].copy()
    tmp=tmp.drop(tmp[tmp.optimizer=='SGD'].index)
    tmp=tmp.drop(tmp[tmp.lr==0.1].index)
    tmp=tmp.drop(tmp[tmp.lr==0.01].index)
    tmp['target'] = tmp.ari - tmp.loss
    tmp = tmp.reset_index() 
    print(tmp.iloc[tmp.target.idxmax()])
    print("===========================================")
'''