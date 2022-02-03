import pathlib
from urllib import request
import gzip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

cv = lambda x: np.nanstd(x) / np.nanmean(x)


'''
#Version 1
Make a singleton Class from the Dataset class to handle the data in the hole code without having different instances. 
This idea is borrowed from the database application connections that allows one instance of the connection to handle the
data server communication. I'm Not in favor of the fight of the singleton is anti-pattern design as it has a lot of 
useful Usage in different fields.

The main idea of using the singleton here is that the TCGA class is going to make handle the data as whole and make the 
stratified split , then 3 class will inherit the split. Each Cluster will implement the __data__ , __get_item__, __len__
'''

'''
#Version 2
Working with singleton is a bit tricky, inheriting a singleton class make it also a singleton, which is not always the 
case that is needed in the singleton implementation. This is a python issue in implementing the singleton. Need to figure
it out. Now the design is as follow : 
1-TCGA is a singleton class that handles 
    A- Download/load Data 
    B- Preprocess the data
    c- Do the stratified splitting (split_data method)
    d- Get the Split data (get_data) 
'''


class SingletonMeta(type):

    _instances = None

    def __call__(cls, *args, **kwargs):
        if not cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances = instance
        return cls._instances


# Making the Singleton using the metaclass function as this is the intendet use for the metaclass actually.
class TCGA (metaclass=SingletonMeta):
    def __init__(self, xena_file_path="./Data/", xena_file_name="rna_data.xena",
                 debug=False, train_size=0.7, test_size=0):
        # Check the existence of the folder and the file
        data_dir = pathlib.Path(xena_file_path)
        file_path = pathlib.Path(xena_file_path+xena_file_name)
        NUM_VAR_GENES = 5000                                                # FIXME: Must be moved to the Config file
        self.train_size = train_size
        self.test_size = test_size


        if not data_dir.exists():
            # make the dir
            print('Target Dir is not found , creating one')
            data_dir.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            # Download the file
            print('File is not found , Downloading it')
            if 0:       # Commented out on macOS due to error is SSL certifi. , need to be checked on Linux
                data_url = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
                request.urlretrieve(data_url,xena_file_path+xena_file_name)
                with gzip.open(xena_file_path+xena_file_name, 'rb') as f_in:
                    with open(xena_file_path+xena_file_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        # Need to get also the annotation file

        # Read Data file
        self.rna_data = pd.read_table(xena_file_path+xena_file_name, index_col=0)
        self.rna_data = self.rna_data.transpose()
        #FIXME: Change the name of the metadata file to be variable and be able to get from the internet
        self.meta_data = pd.read_table("./Data/Survival_SupplementalTable_S1_20171025_xena_sp",index_col=0)
        print('Data read = '+ str(self.rna_data.shape))

        common_idx = list(set(self.rna_data.index)& set(self.meta_data.index))
        self.rna_data = self.rna_data.loc[common_idx]
        self.meta_data = self.meta_data.loc[common_idx]

        if debug:
            var = np.apply_along_axis(cv, axis=-1, arr=self.rna_data)
            plt.hist(var, density=False, bins=100)

        # Selected top Variable genes
        var_genes = self.rna_data.mad(axis=0).sort_values(ascending=False)
        top_var_genes = var_genes.iloc[0:NUM_VAR_GENES, ].index
        self.rna_data_subset = self.rna_data.loc[:, top_var_genes]

        self.rna_data_subset= self.rna_data_subset.loc[common_idx]
        print('Data: Top Var genes selected')

        # Normalize data
        self.rna_data_subset = self.rna_data_subset.apply(zscore, nan_policy='omit')
        print('Data Final size = ' + str(self.rna_data_subset.shape))

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
            print('TCGA Data : Failed to return requested data')
            return None


class TCGA_train(Dataset):
    def __init__(self):
        self.data = TCGA().get_data("train").to_numpy()
        self.data = np.nan_to_num(self.data)

    def __getitem__(self, item):
        return self.data[item,:]

    def __len__(self):
        return self.data.shape[0]


class TCGA_validate(Dataset):
    def __init__(self):
        self.data = TCGA().get_data("validation").to_numpy()
        self.data = np.nan_to_num(self.data)

    def __getitem__(self, item):
        return self.data[item,:]

    def __len__(self):
        return self.data.shape[0]


class TCGA_test(Dataset):
    def __init__(self):
        self.data =TCGA().get_data("test").to_numpy()
        self.data = np.nan_to_num(self.data)

    def __getitem__(self, item):
        return self.data[item,:]


    def __len__(self):
        return self.data.shape[0]