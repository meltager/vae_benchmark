import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class IWAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 num_samples: int = 5,
                 **kwargs) -> None:
        super(IWAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = 5000
        self.num_samples = num_samples
        self.init_method = kwargs['init']

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
                elif self.init_method == 'default':
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
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder.apply(init_weights)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        hidden_dims.reverse()
        modules = []

        decode_layer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU()
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
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (Tensor) [B x S x D]
        :return: (Tensor) [B x S x C x H x W]
        """
        B, _, _ = z.size()
        z = z.reshape(-1, self.latent_dim) #[BS x D]
        #result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        result = self.final_layer(result) #[BS x C x H x W ]
        result = result.reshape([B, -1, result.size(1)]) #[B x S x C x H x W]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        z= self.reparameterize(mu, log_var) # [B x S x D]
        eps = (z - mu) / log_var # Prior samples
        return  [self.decode(z), input, mu, log_var, z, eps]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        eps = args[5]

        #input = input.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4) #[B x S x C x H x W]
        input = input.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x C x H x W]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        log_p_x_z = ((recons - input) ** 2).flatten(2).mean(-1) # Reconstruction Loss [B x S]
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2) ## [B x S]
        # Get importance weights
        log_weight = (log_p_x_z + kld_weight * kld_loss) #.detach().data

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim = -1)
        # kld_loss = torch.mean(kld_loss, dim = 0)

        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim = 0)

        return {'loss': loss, 'Reconstruction_Loss':log_p_x_z.mean(), 'KLD':-kld_loss.mean()}

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
        z = torch.randn(num_samples, 1,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z).squeeze()
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        Returns only the first reconstructed sample
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0][:, 0, :]
