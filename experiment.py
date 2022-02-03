import math

import matplotlib.pyplot as plt
import numpy as np
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
from sklearn import metrics
import igraph as ig
import leidenalg as la


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
        ari = metrics.adjusted_rand_score(self.dataset.meta_data.iloc[:,1],np.array(partition._membership))
        print("ARI = "+ str(ari))
        cluster_score.append({'ARI':ari})

        s_score = metrics.silhouette_score(final_mu.detach().numpy(), np.array(partition._membership), metric='euclidean')
        print("Silhouette_score = "+str(s_score))
        cluster_score.append({'Silhouette':s_score})

        #Save the scores
        np.savetxt(save_dir+"/cluster_score.csv",cluster_score,delimiter=",",fmt = "%s")
