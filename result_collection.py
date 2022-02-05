import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "logs"

init_list = {'default','normal','uniform','xavier_normal','xavier_uniform'}
optimizer_list = {'Adam','SGD','RMSprop'}
lr_list = {'0.1','0.01','0.001','0.0001','1e-05','1e-06'}
latent_list = {'10','20','30','50','100','200'}
model_list = {'VanillaVAE','IWAE','LogCoshVAE','BetaVAE','BetaTCVAE','CategoricalVAE','DIPVAE'}

col_names = ["Model", "init", " optimizer", "ld", "lr ", "epochs", "loss", "ari" ]
grid_reads = pd.DataFrame(columns=col_names)

for init in init_list:
    for opt in optimizer_list:
        for ld in latent_list :
            for lr in lr_list:
                for model in model_list:
                    working_dir= base_dir+ "/" + init + "/" + opt + "/" + ld + "/" + lr + "/" + model
                    #Try to read the log files
                    try:
                        train_log = np.loadtxt(working_dir+"/Train_log.csv",dtype= str,delimiter=",")
                        cluster = np.loadtxt(working_dir+"/cluster_score.csv",dtype= str,delimiter=":")

                        number_of_epochs = int(train_log[-1, 0].split(":")[1])
                        tmp_str = train_log[-1,1].split(":")
                        val_loss = float(tmp_str[1].split("(")[1].split(")")[0])
                        ari =  float(cluster[0, 1].split("}")[0])

                        #Add tot he data frame
                        grid_reads.loc[grid_reads.shape[0]] = [model, init, opt, ld, lr, number_of_epochs, val_loss, ari]

                    except:
                        print("reading model Failed :" + working_dir)


np.savetxt(base_dir+"/results.csv",grid_reads,delimiter=",",fmt = "%s")
print("finished")

