import torch
import numpy as np
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class CategoricalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 categorical_dim: int = 33, # Num classes
                 hidden_dims: List = None,
                 temperature: float = 0.5,
                 anneal_rate: float = 3e-5,
                 anneal_interval: int = 100, # every 100 batches
                 alpha: float = 30.,
                 **kwargs) -> None:
        super(CategoricalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.input_size = 5000  # Fixme: This must not be a fixed value, and passed by someway from the config file
        self.init_method = kwargs['init']


        # Setting the activation function
        if kwargs['activation'] == 'relu':
            self.activation = nn.ReLU()
        elif kwargs['activation'] == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        print("Using activation function: "+ self.activation._get_name())

        def init_weights(m):
            if type(m) == nn.Linear:
                print("Using initialization method: {}".format(self.init_method))
                if self.init_method == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
                elif self.init_method == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
                elif self.init_method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                elif self.init_method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                elif self.init_method == 'ones':
                    torch.nn.init.ones_(m.weight)
                elif self.init_method == 'zeros':
                    torch.nn.init.zeros_(m.weight)
                elif self.init_method =='default':
                    torch.nn.init.kaiming_uniform_(m.weight)
                else:
                    print("Init method not supported, using default")

        modules = []
        if hidden_dims is None:
            hidden_dims = [512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(self.input_size, h_dim),
                    nn.BatchNorm1d(h_dim),
                    self.activation
                    #nn.ReLU()
                )
                )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder.apply(init_weights)
        #self.fc_z = nn.Linear(hidden_dims[-1],self.latent_dim * self.categorical_dim)
        self.fc_z = nn.Linear(hidden_dims[-1],self.latent_dim )

        # Build Decoder
        modules = []

        #self.decoder_input = nn.Linear(self.latent_dim * self.categorical_dim, hidden_dims[-1] )

        hidden_dims.reverse()

        decode_layer = nn.Sequential(
            nn.Linear(self.latent_dim,hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            self.activation
            #nn.ReLU()
        )

        final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.input_size))

        self.decoder = decode_layer
        self.decoder.apply(init_weights)
        self.final_layer = final_layer

        self.sampling_dist = torch.distributions.OneHotCategorical(1. / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Latent code [B x D x Q]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        #z = z.view(-1, self.latent_dim, self.categorical_dim)
        return [z]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        #result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, z: Tensor, eps:float = 1e-7) -> Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        #s = s.view(-1, self.latent_dim * self.categorical_dim)
        return s


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        q = self.encode(input)[0]
        z = self.reparameterize(q)
        return  [self.decode(z), input, q]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        q = args[2]

        q_p = F.softmax(q, dim=-1) # Convert the categorical codes into probabilities

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        batch_idx = kwargs['batch_idx']

        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)

        recons_loss =F.mse_loss(recons, input, reduction='mean')

        # KL divergence between gumbel-softmax distribution
        eps = 1e-7

        # Entropy of the logits
        h1 = q_p * torch.log(q_p + eps)

        # Cross entropy with the categorical distribution
        h2 = q_p * np.log(1. / self.categorical_dim + eps)
        #kld_loss = torch.mean(torch.sum(h1 - h2, dim =(1,2)), dim=0)
        kld_loss = torch.mean(torch.sum(h1 - h2, dim =(1)), dim=0)


        # kld_weight = 1.2
        loss = self.alpha * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # [S x D x Q]

        M = num_samples * self.latent_dim
        np_y = np.zeros((M, self.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // self.latent_dim, self.latent_dim, self.categorical_dim])
        z = torch.from_numpy(np_y)

        # z = self.sampling_dist.sample((num_samples * self.latent_dim, ))
        #z = z.view(num_samples, self.latent_dim * self.categorical_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]