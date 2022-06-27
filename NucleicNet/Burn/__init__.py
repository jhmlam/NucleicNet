import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader# random_split

import os

#from M import RESN
#from M import STRA



# =================================
# Useful Catalogue (Dict)
# =================================
Catalogue_Activation = {
    # Non-parametric
    # > +- bound type
    "tanh": nn.Tanh,
    "softsign": nn.Softsign,

    # > + bound type
    "sigmoid": nn.Sigmoid,

    # > left bound type
    "relu": nn.ReLU, # Hard
    "softplus": nn.Softplus,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "elu": nn.ELU,

    # > Right bound type
    "logsigmoid": nn.LogSigmoid,    

    # > V type
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "silu": nn.SiLU,    

    # > Shrink
    "tanhshrink": nn.Tanhshrink,

    # Parametric
    "prelu": nn.PReLU,         #Hard
    "leakyrelu": nn.LeakyReLU, #Hard
    "randleakyrelu": nn.RReLU, #Hard
}




# ================================
# Tools
# ======================================

model_dict = {}
def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'




# ==================================
# Templates
# ==================================
# NOTE Below are some templates that helps to build up models
#      I wrote two versions. 
#      (1) Version one. W/o annotation to facilitate copying
#      (2) Version two. W/ annotation to explain what is the purpose of lines.


# =====================
# 0a. Creating a block 
# =====================
class TEMPLATE_FullyConnected(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                                torch.nn.Linear(28 * 28, 128),
                                torch.nn.Linear(128, 256), 
                                torch.nn.Linear(256, 10)
                                )
    def forward(self,x):
      net = self.net(x)
      return net
"""
# Single Usage 
M1_FullyConnected = M1_FullyConnected()
M1_FullyConnected(X)
"""

class TEMPLATE_FullyConnected(torch.nn.Module):
    def __init__(self):

        # NOTE Call the constructor of the parent class to perform the necessary initialization for params.
        super().__init__()

        # NOTE Initialise the layers implicitly. 
        #      All layers with layers that will be used has to be written here. 
        #      You cannot defer the definition of these funcitonals in forward. 
        #      NOTE However, nn.Sequential can be deferred to forward See the quote block below
        self.net = nn.Sequential(
                                torch.nn.Linear(28 * 28, 128),
                                torch.nn.Linear(128, 256), 
                                torch.nn.Linear(256, 10)
                                )
        """
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        """
    def forward(self,x):
      """
      net = nn.Sequential(
            self.layer_1, 
            self.layer_2, 
            self.layer_3
            )
      """
      net = self.net(x)
      return net

"""
# Single Usage 
M1_FullyConnected = M1_FullyConnected()
M1_FullyConnected(X)
"""
# =======================================
# 0b. Lightning Nested Module
# =====================================
class TEMPLATE_PytorchLightning(pl.LightningModule):

  def __init__(self, ManualInitiation = False):
      super().__init__()
     
      self._create_network()

      if ManualInitiation:
         self._init_params()


  # ================================
  # User Defined Area
  # ================================
  def _create_network(self):
      self.M1_FullyConnected = TEMPLATE_FullyConnected()

  def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


  # ==============================
  # Standard Lightning Area
  # ==============================
  def forward(self, x):

      # NOTE I would suggest to call from model and only use pl as a shell
      batch_size, channels, width, height = x.size()
      # (b, 1, 28, 28) -> (b, 1*28*28)
      x = x.view(batch_size, -1)
      x = self.M1_FullyConnected(x)
      x = torch.log_softmax(x, dim=1)
      return x

  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = F.nll_loss(logits, y) #self.cross_entropy_loss(logits, y)
      self.log('train_loss', loss)
      return loss


  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = F.nll_loss(logits, y)
      self.log('val_loss', loss)

  def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
      return [optimizer], [scheduler]
    #return optimizer
    


# ===============================
# 0c. Lightning Dataload
# ===============================
# NOTE The outline below is pretty free-form. The header definitions are what we neeed
class MNISTDataModule(pl.LightningDataModule):
  
  def setup(self, stage):
    from torchvision.datasets import MNIST
    from torchvision import datasets, transforms
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
    self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=64, num_workers =0)

  def val_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=64, num_workers = 0)

# ========================
# 0d. Training
# =======================
"""
data_module = MNISTDataModule()
model = TEMPLATE_PytorchLightning()
trainer = pl.Trainer(max_epochs = 3, auto_select_gpus=True, gpus=[0], precision = 16) # max_epochs default at 1000
trainer.fit(model, data_module)
"""


# 1. Chaining up blocks in pytorch. From http://d2l.ai/chapter_deep-learning-computation/model-construction.html
#    NOTE This is equivalent to nn.Sequential()
class TEMPLATE_DaisyChain(nn.Module):
    def __init__(self, *args):
        super().__init__()

        """
        # NOTE if you want to use this in a block, put it within the __init__ for necessary automatic intiiation of params
        net = DaisyChain(
                 nn.Linear(20, 256), 
                 nn.ReLU(), 
                 nn.Linear(256, 20), 
                 MLP2())
        net(X)
        print(net)
        """

        for idx, module in enumerate(args):
            # NOTE `module` is an instance of a `Module` subclass. We save it in the member variable 
            #      `_modules` of the `Module` class as a OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order they were added
        for block in self._modules.values():
            X = block(X)
        return X