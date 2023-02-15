# Benchmarking Variational AutoEncoders on Cancer transcriptomics data

## Introduction
This code was developed as part of studying and investigating the effect of different VAE models and different
hyperparameters on the downstream analysis, also if the learned latent space by the VAEs could reveal something biologically
meaningful.
The VAE models where tested on the TCGA dataset. 
The study can be found currently on the pre-print server bioRxiv and can be accessed here: 
https://www.biorxiv.org/content/10.1101/2023.02.09.527832v1
The code presented in this repository is intended to facilitate the replication of the study results and allow others to 
adapt the code for their own research purposes.

## Study main findings 
In our study we tested the performance of six different VAE models namely: 
* Vanilla VAE
* &beta-VAE
* &beta-TCVAE
* DIP-VAE
* IWAE
* Categorical VAE

And we tested these VAE models against four different hyperparameters values.
* Latent space dimensionality: [10, 20, 30, 50, 100 & 200]
* Learning rate: [ 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, and 1e-6]
* Wight initialization: [Kaming uniform, Normal , uniform, Xavier Normal and Xavier Unifrom]
* Optimizer: [Adam, RMSprop and SGD]

And from this study we found the following findings:
1. Validaiton loss does not refelect downstream perfromance
![Correlation between validaiton loss and ARI of clusteres as downstream performance ](/output_figures/Figure_2.png)
2. Beta-TCVAE and DIP-VAE are the best performing models
![Performance of VAE models in the Clustering task](/output_figures/Figure_3.png)
3. Effect of Hyperparamters on the performance 
   1. Mid-range values for latent space size performs well across all models.
   2. Mid-range values for learning rate perform better than smaller or larger rates. 
   3. Using Uniform distribution underperform all other weight initialization methods.
   4. Adam optimizer is on average slightly better than other optimizers. 
4. Latent space disentanglement is not trivial to achieve in unsupervised manner. 

## Code overview 
To use this code you need to install PyTorch Lighting.


The code found in this repository can be found in the following structure:
* run.py
  * This the main raper for the code, which you can call with line arguments that show the settings needed, the following 
   is the list of options could be passed to the script 
    * -C : configuration file, which should include "name" variable stating the model name needed to run and an output folder to write the logs etc. to it. A complete list of configuration files supported by this code could be found under "configs/" folder 
    * -L : Learning rate 
    * -D : latent dim
    * -O : Optimizer, and could be either (Adam, RMSprop or SGD) only. 
    * -I : wight intilization technique and could be either (default [which is Kaming uniform], Normal , uniform, Xavier Normal or Xavier Unifrom]).
* TCGA.py
  * This is module resposible for the dataset used and do the trainig/validation/test splitting.
  * TCGA is a singleton class by desgin, so that we have only 1 copy of the dataset during the run to save memory.
* experiment.py
  * Handles all the expermeints and measures needed to be done on the model either during or after the training.  
* result_collection.py
  * Is used to collect all the information logged during the training 
* showing_results.py:
  * The functions in this file could generate the figures that are found in the paper and other figures to test the performance of the models.


## Acknowledgment 
The code in this repository is dependant on the implementation of VAE models on https://github.com/AntixK/PyTorch-VAE
