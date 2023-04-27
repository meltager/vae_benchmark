import pathlib
from urllib import request
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import pickle

from torch.utils.data import DataLoader, Dataset


class SingletonMeta(type):

    _instances = None

    def __call__(cls, *args, **kwargs):
        if not cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances = instance
        return cls._instances

class GTEx(metaclass=SingletonMeta):
    def __init__(self, data_file_path = "./Data/GTEx/",data_file_name = "gene_reads.tsv",
                 train_size = 0.7, test_size=0):
        #Setting the folder and data file
        data_dir = pathlib.Path(data_file_path)
        file_path = pathlib.Path(data_file_path+data_file_name)
        pickle_file = pathlib.Path(data_file_path+"processed_data.pickle")
        meta_pickle = pathlib.Path(data_file_path+"meta_data.pickle")

        NUM_VAR_GENES = 5000  # FIXME: Must be moved to the Config file
        self.train_size = train_size
        self.test_size = test_size

        if pickle_file.exists():
            print("Reading from Pickle file")
            self.rna_data_subset = pickle.load(open(pickle_file, 'rb'))
            self.meta_data = pickle.load(open(meta_pickle, 'rb'))

        else:
            self.rna_data = pd.read_table(data_file_path+data_file_name,skiprows=2, index_col=0,delimiter="\t")
            self.rna_data = self.rna_data.drop("Description", axis=1)       #Delete the extra col of gene names and keep the ENSB name
            self.rna_data = self.rna_data.transpose()
            self.meta_data = pd.read_table(data_file_path+"meta_data.txt",index_col=0)

            common_idx = list(set(self.rna_data.index)&set(self.meta_data.index))
            self.rna_data = self.rna_data.loc[common_idx]
            self.meta_data = self.meta_data.loc[common_idx]

            # Selected top Variable genes
            var_genes = self.rna_data.mad(axis=0).sort_values(ascending=False)
            top_var_genes = var_genes.iloc[0:NUM_VAR_GENES, ].index
            self.rna_data_subset = self.rna_data.loc[:, top_var_genes]

            self.rna_data_subset= self.rna_data_subset.loc[common_idx]
            print('Data: Top Var genes selected')

            # Normalize data
            self.rna_data_subset = self.rna_data_subset.apply(zscore, nan_policy='omit')
            print('Data Final size = ' + str(self.rna_data_subset.shape))

            # Save the data in pickle file to faster load later
            with open(data_file_path+'processed_data.pickle','wb') as f:
                pickle.dump(self.rna_data_subset,f)
            with open(data_file_path+'meta_data.pickle','wb') as f:
                pickle.dump(self.meta_data,f)

        #FIXME: This is a dirty work around for the features to be in the same location as TCGA: Must be fixed in the code
        self.meta_data.iloc[:, 1] = self.meta_data.iloc[:, 4]
        self.split_data()


    def split_data(self):
        #Check the values of the Train, validation and test data
        if not ((self.train_size+ self.test_size) <1 and self.train_size > 0 and self.test_size >= 0):
            print("Values of Train, validation and test data is incorrect, they must be between 0 and 1 ")
            print("Using Default values : Train = 0.7, validation =0.3 and Test = 0")
            self.train_size = 0.7
            self.test_size = 0

        self.train_set,self.validation_set,train_idx, validation_idx = \
            train_test_split(self.rna_data_subset,self.meta_data.iloc[:,1], train_size=self.train_size)
        if self.test_size is not 0:
            validation_size = 1-((self.test_size)/(1-self.train_size))
            self.validation_set,self.test_set,validation_idx, test_idx = \
                train_test_split(self.validation_set,validation_idx,train_size=validation_size)
        print("==================================================")
        print('Train Data size= ' + str(self.train_set.shape))
        print('Vald. Data size= ' + str(self.validation_set.shape))
        if self.test_size:
            print('Test  Data size= ' + str(self.test_set.shape))
        print("==================================================")
        return

    def get_data(self,split):
        if split is 'train':
            return self.train_set
        elif split is 'validation':
            return self.validation_set
        elif split is 'test' and self.test_size is not 0:
            return self.test_set
        else:
            print('GTEx Data : Failed to return requested data')
            return None

class GTEx_train(Dataset):
    def __init__(self):
        self.data = GTEx().get_data("train").to_numpy()
        self.data = np.nan_to_num(self.data)

    def __getitem__(self, item):
        return self.data[item,:]

    def __len__(self):
        return self.data.shape[0]


class GTEx_validate(Dataset):
    def __init__(self):
        self.data = GTEx().get_data("validation").to_numpy()
        self.data = np.nan_to_num(self.data)

    def __getitem__(self, item):
        return self.data[item,:]

    def __len__(self):
        return self.data.shape[0]


class GTEx_test(Dataset):
    def __init__(self):
        self.data =GTEx().get_data("test").to_numpy()
        self.data = np.nan_to_num(self.data)

    def __getitem__(self, item):
        return self.data[item,:]

    def __len__(self):
        return self.data.shape[0]