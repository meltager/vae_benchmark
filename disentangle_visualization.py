'''
Make sure that the files has the heading as the first line
aslo remove all the cearly brackets and also the commas
Make sure that the file is neet
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pathlib

working_dir = './disentangle'
full_result=pd.DataFrame()
for filename in os.listdir(working_dir):
    with open(os.path.join(working_dir,filename),'r') as f:
        result_tmp = pd.read_csv(f,delimiter=":",header=0)
        sns.boxplot(data=result_tmp)
        plt.savefig("./figures/"+filename.split(".")[0]+"_disentangle.png")
        plt.cla()
        result_tmp.insert(0,"Model",filename.split(".")[0])
        full_result = full_result.append(result_tmp,ignore_index=True)
    print("Hello")

for i in range(2,6):
    sns.boxplot(data=full_result,x='Model',y=full_result.columns[i])
    plt.savefig("./figures/" + full_result.columns[i] + "_disentangle.png")
    plt.cla()

