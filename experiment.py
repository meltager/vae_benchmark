import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from TCGA import *
import umap
import umap.plot
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn import metrics
import igraph as ig
import leidenalg as la
from itertools import combinations
import statsmodels.api as sm

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.num_val_data = 0
        self.num_test_data = 0
        self.train_trace=[]
        self.save_hyperparameters()

        self.dataset = TCGA(train_size=self.params['train_size'],test_size=self.params['test_size'])
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_data = batch
        self.curr_device = real_data.device

        results = self.forward(real_data.float())
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_data,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # This is the old version when using test tube logger
        #self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        #This is the newer version using tensorboard logger
        for key,val in train_loss.items():
            #self.logger.experiment.add_scalar(key,val)

            self.log(key,val, on_step=False,on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_data= batch
        self.curr_device = real_data.device

        results = self.forward(real_data.float())
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_data,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss
        #return {'val_loss': val_loss, 'log': {'val_loss': val_loss}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.log('val_loss',avg_loss)
        self.train_trace.append({'epoch':self.current_epoch,'val_loss':avg_loss})
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = None

        # Check what optimizer needs to be added, or else the default optimizer is 'Adam'
        if self.params['optimizer'] == "Adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.params['LR'],
                                   weight_decay=self.params['weight_decay'])
        elif self.params['optimizer'] == "SGD":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.params['LR'],
                                  weight_decay=self.params['weight_decay'])
        elif self.params['optimizer'] == "RMSprop":
            optimizer = optim.RMSprop(self.model.parameters(),
                                      lr=self.params['LR'],
                                      weight_decay=self.params['weight_decay'])
        else:
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.params['LR'],
                                   weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        if self.params['dataset'] == 'TCGA':
            dataset = TCGA_train()
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_data = dataset.__len__()
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        if self.params['dataset'] == 'TCGA':
            dataset = TCGA_validate()
            self.sample_dataloader = DataLoader(dataset,
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = False,
                                                 drop_last=False)
            try:
                self.num_val_data = dataset.__len__()
            except:
                pass

        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    @data_loader
    def test_dataloader(self):
        if self.params['dataset']=='TCGA':
            dataset = TCGA_test()
        else:
            raise ValueError('Undefined dataset type')
        try:
            self.num_test_data = dataset.__len__()
        except:
            pass

        return DataLoader(dataset,batch_size=self.params['batch_size'], shuffle = False,
                                                 drop_last=True)


    def get_data_umap(self,save_dir):
        data_to_embed =torch.tensor(np.nan_to_num(self.dataset.rna_data_subset.to_numpy())).float()
        try:
            final_mu,_ = self.model.encode(data_to_embed)
        except:
            final_mu, = self.model.encode(data_to_embed)

        mapper = umap.UMAP().fit(np.nan_to_num(final_mu.detach().numpy()))
        umap.plot.points(mapper, labels=self.dataset.meta_data.iloc[:, 1], color_key_cmap='Paired',show_legend=False)
        # umap.plot.plt.show()
        plt.title(self.model._get_name()+ "\n"+
                  " Init:" + str(self.model.init_method)+
                  " Opt:"+ self.params['optimizer']+
                  " Latent: "+str(self.model.latent_dim)+
                  " LR:" + str(self.params['LR']) )
        plt.savefig(save_dir + "/umap.svg", bbox_inches='tight', dpi=300)
        return



    def get_data_cluster(self,save_dir,n_neighbours = 15,draw_umap =False):
        cluster_score = []
        #Get the data from the latent space
        data_to_embed = torch.tensor(np.nan_to_num(self.dataset.rna_data_subset.to_numpy())).float()
        try:
            final_mu, _ = self.model.encode(data_to_embed)
        except:
            final_mu, = self.model.encode(data_to_embed)

        try:
            #Create a neighbourhood graph
            neighbours = NearestNeighbors(n_neighbors = n_neighbours, metric='minkowski',p=2)
            neighbours.fit(final_mu.detach().numpy())
            neighbours_list = neighbours.kneighbors(final_mu.detach().numpy(),return_distance = False)

            #Create Adj Mtx.    ==> There must be a smarter way to do it
            adj_mtx = np.zeros((final_mu.shape[0],final_mu.shape[0]))
            tmp_idx = 0
            for i in neighbours_list:
                for j in i:
                    adj_mtx[tmp_idx,j] = 1
                tmp_idx+=1

            # Create an iGraph from the Adj Matrix
            g = ig.Graph.Adjacency(adj_mtx, mode='undirected')

            # Cluster using Lieden algorithm
            partition = la.find_partition(g, la.ModularityVertexPartition, n_iterations=-1, seed=42)

            #draw umap
            if draw_umap:
                mapper = umap.UMAP().fit(np.nan_to_num(final_mu.detach().numpy()))
                umap.plot.points(mapper, labels=np.array(partition._membership), color_key_cmap='Paired',
                                 show_legend=True)

                plt.title(self.model._get_name() + "\n" +
                          " Init:" + str(self.model.init_method) +
                          " Opt:" + self.params['optimizer'] +
                        " Latent: " + str(self.model.latent_dim) +
                        " LR:" + str(self.params['LR']) +
                        " # Clusters: " + str(partition.__len__()))

                plt.savefig(save_dir + "/umap_clustered.svg", bbox_inches='tight', dpi=300)

                # Calc. the Clustering scores
            ari = metrics.adjusted_rand_score(self.dataset.meta_data.iloc[:, 1], np.array(partition._membership))
            print("ARI = " + str(ari))

            s_score = metrics.silhouette_score(np.nan_to_num(final_mu.detach().numpy()),
                                                   np.array(partition._membership), metric='euclidean')
            print("Silhouette_score = " + str(s_score))

        except:
            ari = np.nan
            s_score = np.nan

        cluster_score.append({'ARI':ari})
        cluster_score.append({'Silhouette':s_score})

        #Save the scores
        np.savetxt(save_dir+"/cluster_score.csv",cluster_score,delimiter=",",fmt = "%s")

    def test_multidim_disentanglement(self, save_dir, max_num_dim=3):
        collected_score = []
        #encode the data
        data_to_embed = torch.tensor(np.nan_to_num(self.dataset.rna_data_subset.to_numpy())).float()
        try:
            final_mu, _ = self.model.encode(data_to_embed)
        except:
            final_mu, = self.model.encode(data_to_embed)

        dim_tuple = tuple(range(0, final_mu.shape[1]))
        gender_target = pd.get_dummies(self.dataset.meta_data.gender)
        race_target = pd.get_dummies(self.dataset.meta_data.race).iloc[:,:-2]        # Remove last 2 because they are the unkown and unspecified variables
        age_target = -(self.dataset.meta_data.birth_days_to).fillna(0)

        for i in range(1,max_num_dim+1):
            # Create all the possible combinations for the latent dimensions
            combinations_list = tuple(combinations(dim_tuple,i))
            for y in combinations_list:
                train = final_mu[:,y].detach().numpy().reshape(-1,i)
                lnr_reg = LinearRegression().fit(train,gender_target)
                gender_score = lnr_reg.score(train,gender_target)
                lnr_reg = LinearRegression().fit(train, race_target)
                race_score =lnr_reg.score(train,race_target)
                lnr_reg = LinearRegression().fit(train, age_target)
                age_score = lnr_reg.score(train,age_target)
                collected_score.append({'combination': y, 'gender_score': gender_score,'race_score':race_score , 'age_score': age_score})
            #final_mu[:,combinations_list[0]]

        np.savetxt(save_dir + "/disentanglement.csv", collected_score, delimiter=",", fmt="%s")


    def test_singledim_disentanglement(self,save_dir):
        collected_score=[]
        col_names = ['Dim','gender_score_train','gender_train_beta','gender_train_pval',
                                    'gender_score_test','gender_test_beta','gender_test_pval',
                                    'race_score_train','race_score_test','age_R_train',
                                    'age_pVal_train','age_R_test','age_pVal_test',
                                    'immune_R_train', 'immune_pVal_train', 'immune_R_test', 'immune_pVal_test',
                                    'metastat_R', 'Metastat_pVal']
        #collected_results= pd.DataFrame(columns=col_names)
        collected_results= pd.DataFrame()
        train_data = pd.concat([self.dataset.train_set,self.dataset.validation_set])
        train_data_to_embed = torch.tensor(np.nan_to_num(train_data.to_numpy())).float()
        test_data_to_embed =  torch.tensor(np.nan_to_num(self.dataset.test_set)).float()

        try:
            train_mu, _ = self.model.encode(train_data_to_embed)
            test_mu, _  = self.model.encode(test_data_to_embed)
            rna_total, _= self.model.encode(torch.tensor(np.nan_to_num(self.dataset.rna_data_subset)).float())
        except:
            train_mu,  = self.model.encode(train_data_to_embed)
            test_mu,   = self.model.encode(test_data_to_embed)
            rna_total, = self.model.encode(self.dataset.rna_data_subset)

        age_target_train = -(self.dataset.meta_data.birth_days_to.loc[train_data.index])
        age_target_test = -(self.dataset.meta_data.birth_days_to.loc[self.dataset.test_set.index])

        #Get the Raw Data to calc. mutation signature
        if not hasattr(self.dataset,'rna_data'):
            # Read the raw file data
            self.dataset.rna_data = pd.read_table('./Data/rna_data.xena', index_col=0)
            self.dataset.rna_data = self.dataset.rna_data.transpose()
            self.dataset.rna_data = self.dataset.rna_data.loc[self.dataset.meta_data.index]

        # 2. Read the gene names from file
        gene_list = pd.read_csv('./Data/prob_list.csv')
        mutation_signature = self.dataset.rna_data.loc[:,self.dataset.rna_data.columns.isin(gene_list['Gene Name'])].sum(axis=1)

        for i in range(train_mu.shape[1]):

            #Gender Testing
            train =train_mu[:, i].detach().numpy().reshape(-1, 1)
            target =self.dataset.meta_data.gender.loc[train_data.index]
            reg = LogisticRegression().fit(train, target)
            gender_score = reg.score(train,target)
            #Use Statmodels
            train = sm.add_constant(train)
            model = sm.Logit(pd.get_dummies(target).iloc[:,0],train,missing='drop')
            gender_train_result = model.fit_regularized()

            test =test_mu[:, i].detach().numpy().reshape(-1, 1)
            target = self.dataset.meta_data.gender.loc[self.dataset.test_set.index]
            reg = LogisticRegression().fit(test, target)
            gender_score_test = reg.score(test, target)
            # Use Statmodels
            test = sm.add_constant(test)
            model = sm.Logit(pd.get_dummies(target).iloc[:,0], test, missing='drop')
            gender_test_result = model.fit_regularized()

            # RACE testing
            train = train_mu[:, i].detach().numpy().reshape(-1, 1)
            target = self.dataset.meta_data.race.loc[train_data.index]
            target.loc[target == '[Unknown]'] = np.NaN
            target.loc[target == '[Not Evaluated]'] = np.NaN
            tmp_target= target.loc[target.notna()]
            train = train[target.index.get_indexer(tmp_target.index)]
            reg = LogisticRegression(class_weight='balanced').fit(train, tmp_target)
            race_score = reg.score(train, tmp_target)

            test = test_mu[:, i].detach().numpy().reshape(-1, 1)
            target = self.dataset.meta_data.race.loc[self.dataset.test_set.index]
            target.loc[target == '[Unknown]'] = np.NaN
            target.loc[target == '[Not Evaluated]'] = np.NaN
            tmp_target= target.loc[target.notna()]
            test = test[target.index.get_indexer(tmp_target.index)]
            reg = LogisticRegression(class_weight='balanced').fit(test, tmp_target)
            race_score_test = reg.score(test, tmp_target)

            #Age Testing
            r_train, p_train = stats.spearmanr(train_mu[:, i].detach().numpy(), age_target_train, nan_policy='omit')
            r_test, p_test = stats.spearmanr(test_mu[:, i].detach().numpy(), age_target_test, nan_policy='omit')

            # immune gene testing
            immune_r_train, immune_p_train = stats.spearmanr(train_mu[:,i].detach().numpy(),mutation_signature.loc[train_data.index],nan_policy = 'omit')
            immune_r_test, immune_p_test = stats.spearmanr(test_mu[:,i].detach().numpy(),mutation_signature.loc[self.dataset.test_set.index],nan_policy = 'omit')

            # Metastatis test
            metastat_r, metastat_p = stats.spearmanr(rna_total[:,i].detach().numpy(),self.dataset.meta_data.new_tumor_event_dx_days_to,nan_policy='omit')

            # Calculate Regression to top 10 Cancer types
            #1-Get the most frequent 10 cancer types
            cancer_freq = self.dataset.meta_data.cancer_type_abbreviation.value_counts()
            cancer = pd.get_dummies(self.dataset.meta_data.cancer_type_abbreviation)
            train =train_mu[:, i].detach().numpy().reshape(-1, 1)
            train_c = sm.add_constant(train)
            test = test_mu[:, i].detach().numpy().reshape(-1, 1)
            test_c = sm.add_constant(test)
            tmp_result = pd.DataFrame()
            for y in range(0,10):
                tmp_cancer_target = cancer.loc[train_data.index, cancer.columns == cancer_freq.index[y]]
                model = sm.Logit(tmp_cancer_target,train_c,missing='drop')
                result = model.fit_regularized()
                tmp_result.insert(tmp_result.shape[0],cancer_freq.index[y]+'_train_coff',[result.params[1]])
                tmp_result.insert(tmp_result.shape[0],cancer_freq.index[y]+'_train_pval',[result.pvalues[1]])

                #Use skitlearn as verification
                reg = LogisticRegression().fit(train, tmp_cancer_target)
                test_score = reg.score(train, tmp_cancer_target)
                tmp_result.insert(tmp_result.shape[0],cancer_freq.index[y]+'_train_score',[test_score])

                #log the data
                tmp_result
                tmp_cancer_target = cancer.loc[self.dataset.test_set.index, cancer.columns == cancer_freq.index[y]]
                model = sm.Logit(tmp_cancer_target, test_c, missing='drop')
                result = model.fit_regularized()
                tmp_result.insert(tmp_result.shape[0],cancer_freq.index[y]+'_test_coff',[result.params[1]])
                tmp_result.insert(tmp_result.shape[0],cancer_freq.index[y]+'_test_pval',[result.pvalues[1]])

                # Use skitlearn as verification
                reg = LogisticRegression().fit(test, tmp_cancer_target)
                test_score = reg.score(test, tmp_cancer_target)
                tmp_result.insert(tmp_result.shape[0],cancer_freq.index[y]+'_test_score',[test_score])


            tmp = pd.DataFrame(
                [[i, gender_score,gender_train_result.params[1],gender_train_result.pvalues[1],
                  gender_score_test,gender_test_result.params[1],gender_test_result.pvalues[1],
                  race_score, race_score_test, r_train, p_train, r_test, p_test,
                  immune_r_train, immune_p_train,immune_r_test, immune_p_test,
                  metastat_r,metastat_p]],
                columns=col_names)

            tmp = pd.concat([tmp, tmp_result], axis=1).reindex(tmp.index)

            collected_results = collected_results.append(tmp,ignore_index=True)

        #np.savetxt(save_dir + "/disentanglement_single_test.csv", collected_score, delimiter=",", fmt="%s")
        collected_results.to_csv(save_dir + "/disentanglement_single_test.csv")


#self.dataset.rna_data[self.dataset.rna_data.index.isin(gene_list['Gene Name'])]

#tmp_x.loc[tmp_x['Sample Names'].str.contains(self.dataset.rna_data_subset.index[0],case=False)]

    def test_mutation_sig_disentanglment (self,save_dir,file_dir = './Data/signatures_data.csv'):
        # 1. Read the Mutation signature file
        tmp_mutation_sig = pd.read_csv(file_dir)
        tmp_sig = pd.DataFrame()
        selected_sig = pd.DataFrame()
        for i in range(self.dataset.meta_data.shape[0]):
            if not tmp_mutation_sig.loc[tmp_mutation_sig['Sample Names'].str.contains(self.dataset.rna_data_subset.index[i], case=False)].shape[
                       0] == 0:
                tmp_sig = tmp_mutation_sig.loc[tmp_mutation_sig['Sample Names'].str.contains(self.dataset.rna_data_subset.index[i], case=False)].copy()
                tmp_sig['Sample Names'] = self.dataset.rna_data_subset.index[i]
                selected_sig = pd.concat([selected_sig, tmp_sig])

        selected_sig = selected_sig.set_index('Sample Names')
        tmp_sbs = selected_sig.iloc[:,:].astype(bool).sum()
        #tmp_sbs.sort_values(ascending=False).plot(kind='bar')
        tmp_patient = selected_sig.iloc[:,:].sum(axis=1)
        #Normalize patient per signature
        selected_sig['sum'] = selected_sig.iloc[:, 2:].sum(axis=1)
        normalized_sig = selected_sig.iloc[:, 2:-2].div(selected_sig["sum"], axis=0)
        selected_norm_sig = normalized_sig.loc[tmp_patient>300,tmp_sbs>500]

        # Encode the same patients data
        selected_patients = self.dataset.rna_data_subset.loc[selected_norm_sig.index]
        selected_patients = torch.tensor(np.nan_to_num(selected_patients)).float()
        try:
            patients_mu, _ = self.model.encode(selected_patients)
        except:
            patients_mu, = self.model.encode(selected_patients)

        results = pd.DataFrame(index=range(patients_mu.shape[1]), columns=range(selected_norm_sig.shape[1]))

        for i in range(patients_mu.shape[1]):
            for y in range(selected_norm_sig.shape[1]):
                print(i,y )
                r, p_val = stats.spearmanr(patients_mu[:, i].detach().numpy(),selected_norm_sig.iloc[:,y],nan_policy='omit')
                results.iloc[i,2*y] = r
                results.iloc[i,(2*y)+1]=p_val

        results.to_cs(save_dir + "/disentanglement_signature_file.csv")

'''
tmp_sing = pd.DataFrame()
selected_sing =  pd.DataFrame()
for i in range(self.dataset.meta_data.shape[0]):
    if not tmp_x.loc[tmp_x['Sample Names'].str.contains(self.dataset.rna_data_subset.index[i],case=False)].shape[0] ==0:
        tmp_sing = tmp_x.loc[tmp_x['Sample Names'].str.contains(self.dataset.rna_data_subset.index[i],case=False)].copy()
        tmp_sing['Sample Names'] = self.dataset.rna_data_subset.index[i]
        selected_sing = pd.concat([selected_sing,tmp_sing])
    
selected_sing = selected_sing.set_index('Sample Names')
    
#tmp_sbs = selected_sing.iloc[:,2:].astype(bool).sum()
tmp_sbs = selected_sing.iloc[:,:].astype(bool).sum()
tmp_sbs.sort_values(ascending = False).plot(kind ='bar')
tmp_patient = selected_sing.iloc[:,2:].sum(axis=1)

selected_sing.loc[tmp_patient>300, tmp_sbs>1000]
    
    
'''
