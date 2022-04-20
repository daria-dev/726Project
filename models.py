from torch import nn
import torch

'''
    Discriminator
'''
class Discriminator(nn.Module):
  def __init__(self, latent_space_dim):
    super().__init__()
    self.latent_space_dim = latent_space_dim
    layers = [nn.Linear(4*4*latent_space_dim, latent_space_dim),
              nn.BatchNorm1d(latent_space_dim),
              nn.Dropout(0.3),
              nn.ReLU(),
              nn.Linear(latent_space_dim, 32),
              nn.Dropout(0.3),
              nn.ReLU(),
              nn.Linear(32,40),
              nn.Sigmoid()]
    self.model = nn.Sequential(*layers)
    
  def forward(self, z_s):
    batch_size = z_s.shape[0]
    z_s = z_s.view(batch_size,self.latent_space_dim*4*4)
    return self.model(z_s)

'''
    Fader Network
'''
class FaderNetwork(nn.Module):
    def __init__(self, latent_space_dim, in_channels, attribute_dim):
        super().__init__()
        enc_layers = [nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(256, latent_space_dim, kernel_size=3, stride=2, padding=1)]
                  
        self.dec_layers = nn.ModuleList([nn.ConvTranspose2d(latent_space_dim+attribute_dim, 256, 3, 2, 1, 1, bias=False),
                  nn.ReLU(),
                  nn.ConvTranspose2d(256+attribute_dim, 128, 3, 2, 1, 1, bias=False),
                  nn.ReLU(),
                  nn.ConvTranspose2d(128+attribute_dim, 64, 3, 2, 1, 1, bias=False),
                  nn.ReLU(),
                  nn.ConvTranspose2d(64+attribute_dim, 32, 3, 2, 1, 1, bias=False),
                  nn.ReLU(),
                  nn.ConvTranspose2d(32+attribute_dim, 16, 3, 2, 1, 1, bias=False),
                  nn.ReLU(),
                  nn.ConvTranspose2d(16+attribute_dim, 3, 3, 2, 1, 1, bias=False),
                  nn.ReLU(),
                  nn.Tanh()])

        # 3 -> encoder -> z
        # z,6 -> decoder --> 6
        # loss = l2()
                  
        self.encoder = nn.Sequential(*enc_layers)

    def forward(self, images, attr):
        enc = self.encode(images)
        return self.decode(enc, attr)

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z, attr):
        z_s = torch.cat([z, attr], dim=1)
        out = z_s
        
        for i,l in enumerate(self.dec_layers):
            if type(l) == nn.ConvTranspose2d and i > 0:
                # if i == 4:
                #   attr = torch.cat([attr, attr[:,:,:7,:]], dim=2)
                #   attr = torch.cat([attr, attr[:,:,:,:7]], dim=3)
                #   out = torch.cat([out, attr], dim=1)
                # elif i == 6:
                #     print(attr.shape)
                #     attr = torch.cat([attr, attr[:,:,:14,:]], dim=2)
                #     attr = torch.cat([attr, attr[:,:,:,:14]], dim=3)
                #     out = torch.cat([out, attr], dim=1)
                # else:
                attr = torch.cat([attr, attr], dim=2)
                attr = torch.cat([attr, attr], dim=3)
                out = torch.cat([out, attr], dim=1)
            
            out = l(out)

        return out