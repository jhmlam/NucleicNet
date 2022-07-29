import torch
import numpy
import scipy
import seaborn
import pandas
import plotly
import torchmetrics
import biopandas
import sklearn
import tqdm

torch.backends.cudnn.is_available()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# disable debugs NOTE use only after debugging
torch.autograd.set_detect_anomaly(False)

A = torch.randn((90,90), device = 0)
A_sym = (A.T + A )/2 + torch.eye(90, device = 0)

torch.linalg.eigvalsh(A_sym)
torch.linalg.eigvals(A_sym)
torch.linalg.qr(A_sym)
