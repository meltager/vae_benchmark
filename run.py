import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch import save

valid_optimizers = ['Adam','SGD','RMSprop']
valid_inits = ['uniform','normal','xavier_uniform','xavier_normal','ones','zeros']

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--learning_rate','-L',
                    dest="learningRate",
                    metavar='Number',
                    help = 'learning rate in decimal format eg: 0.1',
                    default='0'
                    )
parser.add_argument('--latent_dim','-D',
                    dest="latentDim",
                    metavar='Number',
                    help= 'number of latent dim. eg: 10',
                    default='0'
                    )
parser.add_argument('--optimizer','-O',
                    dest="optimizer",
                    metavar='String',
                    help = 'Optimizer used eg: Adam',
                    default=''
                    )
parser.add_argument('--initialization_weight','-I',
                    dest="w_init",
                    metavar='String',
                    help='the weight initialization used eg: uniform',
                    default=''
                    )
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

#HACK: Check the values of the params if default use that of the config file
#FIXME: exception handler need to be fitted according to the error

try:
    if float(args.learningRate):
        config['exp_params']['LR'] = float(args.learningRate)
        print("Using Learning Rate = "+str(config['exp_params']['LR']))
except:
    print("Failed to parse the Learning Rate from the line args\nUsing the Config file learing Rate = "
          +config['exp_params']['LR'])

try:
    if int(args.latentDim):
        config['model_params']['latent_dim'] = int(args.latentDim)
        print("Using Latent Dim = "+ str(config['model_params']['latent_dim']))
except:
    print("Failed to parse the latent dim. from the line args\nUsing the Config file latent dim ="
          +config['model_params']['latent_dim'])

try:
    if args.optimizer in valid_optimizers:
        config['exp_params']['optimizer'] = args.optimizer
        print("Using Optimizer ="+ config['exp_params']['optimizer'])
except:
    print("Failed to parse the optimizer from the line args\nUsing the Config file optimizer = "
          +config['exp_params']['optimizer'])

try:
    if args.w_init in valid_inits:
        config['model_params']['init'] = args.w_init
        print("Using weight initalizer =" + config['model_params']['init'])
except:
    print("Failed to parse the weight init. from the line args\nUsing the Config file weight init. = "
          +config['model_params']['init'])

#setting the logs dir to be logs/init/optimizer/latent_dim/LR/Model/Version
config['logging_params']['save_dir']+=config['model_params']['init']+"/"
config['logging_params']['save_dir']+=config['exp_params']['optimizer']+"/"
config['logging_params']['save_dir']+=str(config['model_params']['latent_dim'])+"/"
config['logging_params']['save_dir']+=str(config['exp_params']['LR'])+"/"


#TestTubeLooger : Log to local file system in TensorBoard format but using a nicer folder structure
#tt_logger = TestTubeLogger(
tt_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    #debug=False,
    #create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

# Early stopping
my_early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.1,
   patience=3,
   verbose=True,
   mode='min'
)
runner = Trainer(
                 min_epochs=5,
                 logger=tt_logger,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=3,
                 auto_select_gpus=True,
                 log_every_n_steps=1,
                 callbacks=[my_early_stop_callback],
                 #early_stop_callback = True,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
experiment.get_data_umap(config['logging_params']['save_dir']+config['logging_params']['name'])
#save the entier model
#torch.save(experiment,config['logging_params']['save_dir']+config['logging_params']['name']+"/saved_model.tr")

#Save the model paramters (Faster, and recommended)
torch.save(experiment.state_dict(),config['logging_params']['save_dir']+config['logging_params']['name']+"/saved_model_params.tr")
#To load the model use the following :
#experiment.load_state_dict(torch.load(config['logging_params']['save_dir']+config['logging_params']['name']+"/saved_model_params.tr"))

# Save the Training log
np.savetxt(config['logging_params']['save_dir']+config['logging_params']['name']+"/Train_log.csv",experiment.train_trace,delimiter=",",fmt = "%s")

#Calc. Clustering
experiment.get_data_cluster(config['logging_params']['save_dir']+config['logging_params']['name'],draw_umap=True)