import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class DIPVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 lambda_diag: float = 10.,
                 lambda_offdiag: float = 5.,
                 **kwargs) -> None:
        super(DIPVAE, self).__init__()

        self.latent_dim = latent_dim
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
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
                    nn.Linear(self.input_size,h_dim),
                    nn.BatchNorm1d(h_dim),
                    self.activation
                    #nn.ReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder.apply(init_weights)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []


        hidden_dims.reverse()
        decode_layer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            self.activation
            #nn.ReLU()
        )

        final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.input_size))

        self.decoder = decode_layer
        self.decoder.apply(init_weights)
        self.final_layer = final_layer

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

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
        mu = args[2]
        log_var = args[3]

        kld_weight = 1 #* kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input, reduction='sum')


        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # DIP Loss
        centered_mu = mu - mu.mean(dim=1, keepdim = True) # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze() # [D X D]

        # Add Variance for DIP Loss II
        cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1 = 0), dim = 0) # [D x D]
        # For DIp Loss I
        # cov_z = cov_mu

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
                   self.lambda_diag * torch.sum((cov_diag - 1) ** 2)

        loss = recons_loss + kld_weight * kld_loss + dip_loss
        return {'loss': loss,
                'Reconstruction_Loss':recons_loss,
                'KLD':-kld_loss,
                'DIP_Loss':dip_loss}

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
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]