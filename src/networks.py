import torch
import torch.nn as nn
import torch.nn.functional as F
from utilFuncs import set_seed

#%%
class Encoder(nn.Module):
  def __init__(self, encoderSettings):
    super(Encoder, self).__init__()
    set_seed(1234)
    self.input_layer = nn.Linear(encoderSettings['inputDim'], encoderSettings['hiddenDim'][0])
    # self.linear1 = nn.Linear(encoderSettings['inputDim'], encoderSettings['hiddenDim'][0])

# Create hidden layers
    self.hidden_layers = nn.ModuleList()
    for i in range(1, len(encoderSettings['hiddenDim'])):
      self.hidden_layers.append(nn.Linear(encoderSettings['hiddenDim'][i - 1], encoderSettings['hiddenDim'][i]))
      print('here encoder')

    # Define the mean and log variance layers
    
    self.output_layer = nn.Linear(encoderSettings['hiddenDim'][-1], encoderSettings['latentDim'])
    self.output_layer2 = nn.Linear(encoderSettings['hiddenDim'][-1], encoderSettings['latentDim'])

    self.N = torch.distributions.Normal(0, 1)
    self.kl = 0
    self.isTraining = False
  def forward(self, x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"


    # Forward pass through the encoder layers
    x = torch.relu(self.input_layer(x))
    for layer in self.hidden_layers:
      x = torch.relu(layer(x))
    # x = F.relu(self.linear1(x))
    mu =  self.output_layer(x)
    sigma = torch.exp(self.output_layer2(x))
    if(self.isTraining):

      self.z = mu + sigma*self.N.sample(mu.shape).to(device)
    else:
      self.z = mu

    self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return self.z.to(device)
#--------------------------#
class Decoder(nn.Module):
  def __init__(self, decoderSettings):
    super(Decoder, self).__init__()
    self.input_layer = nn.Linear(decoderSettings['latentDim'],  decoderSettings['hiddenDim'][-1])
    self.output_layer = nn.Linear(decoderSettings['hiddenDim'][0],decoderSettings['outputDim'])
    # self.linear1 = nn.Linear(decoderSettings['latentDim'], decoderSettings['hiddenDim'][0])

# Create hidden layers
    self.hidden_layers = nn.ModuleList()
    for i in range(len(decoderSettings['hiddenDim']) - 1, 0, -1):
      self.hidden_layers.append(nn.Linear(decoderSettings['hiddenDim'][i], decoderSettings['hiddenDim'][i - 1]))
      print('here decoder')
    # Define the mean and log variance layers
  
    # self.linear1 = nn.Linear(decoderSettings['latentDim'], decoderSettings['hiddenDim'])
    # self.linear2 = nn.Linear(decoderSettings['hiddenDim'], decoderSettings['outputDim'])

  def forward(self, z):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    z = torch.relu(self.input_layer(z))
    for layer in self.hidden_layers:
      z = torch.relu(layer(z))
    
    # z = F.relu(self.linear1(z)) #
    z = torch.sigmoid(self.output_layer(z)) # decoder op in range [0,1]
    return z.to(device)
#--------------------------#
class VariationalAutoencoder(nn.Module):
  def __init__(self, vaeSettings):
    super(VariationalAutoencoder, self).__init__()

    self.encoder = Encoder(vaeSettings['encoder'])
    self.decoder = Decoder(vaeSettings['decoder'])

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)
#--------------------------#
#%%
class MaterialNetwork(nn.Module):
  def __init__(self, nnSettings):
    self.nnSettings = nnSettings
    super().__init__()
    self.layers = nn.ModuleList()
    set_seed(1234)
    current_dim = nnSettings['inputDim']
    for lyr in range(nnSettings['numLayers']): # define the layers
      l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = nnSettings['numNeuronsPerLyr']
    self.layers.append(nn.Linear(current_dim, nnSettings['outputDim']))
    self.bnLayer = nn.ModuleList()
    for lyr in range(nnSettings['numLayers']): # batch norm
      self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

  def forward(self, x):
    m = nn.LeakyReLU();
    ctr = 0;
    for layer in self.layers[:-1]: # forward prop
      x = m(layer(x))#m(self.bnLayer[ctr](layer(x)));
      ctr += 1;
    opLayer = self.layers[-1](x)
    nnOut = torch.sigmoid(opLayer)
    z = self.nnSettings['zMin'] + self.nnSettings['zRange']*nnOut
    return z

#--------------------------#
#%%
class TopologyNetwork(nn.Module):
  def __init__(self, nnSettings):
    self.inputDim = nnSettings['inputDim']# x and y coordn of the point
    self.outputDim = nnSettings['outputDim']
    super().__init__()
    self.layers = nn.ModuleList()
    set_seed(1234)
    current_dim = self.inputDim
    for lyr in range(nnSettings['numLayers']): # define the layers
      l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = nnSettings['numNeuronsPerLyr']
    self.layers.append(nn.Linear(current_dim, self.outputDim))
    self.bnLayer = nn.ModuleList()
    for lyr in range(nnSettings['numLayers']): # batch norm
      self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

  def forward(self, x):
    m = nn.LeakyReLU()
    ctr = 0
    for layer in self.layers[:-1]: # forward prop
      x = m(self.bnLayer[ctr](layer(x)))
      ctr += 1
    opLayer = self.layers[-1](x)
    rho = torch.sigmoid(opLayer).view(-1)
    return rho