
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import vstack as scipysparsevstack
import scipy



# ===============================
# List of Things to Tensor
# ===============================
def ScipyCsr_TorchSparseFloat(databatchCSR:scipy.sparse.csr_matrix):
    
    databatchCSR = scipysparsevstack(databatchCSR).tocoo()
    i = torch.LongTensor(scipysparsevstack((databatchCSR.row, databatchCSR.col)))
    v = torch.FloatTensor(databatchCSR.data)
    s = torch.Size(databatchCSR.shape)

    return torch.sparse.FloatTensor(i, v, s)

def ScipyCsr_TorchFloat(databatchCSR:scipy.sparse.csr_matrix):

    return torch.FloatTensor(scipysparsevstack(databatchCSR).todense())




# =====================
# Misc
# ======================
def Bhw_Bnhw(I, device = torch.device('cuda'), n_channel = 3):
    """This converts a (b,h,w) to (b,1,h,w) and then simply repeat the color channel"""
    return I.unsqueeze(1).repeat(1,n_channel,1,1)

def B1hw_Bnhw(I, device = torch.device('cuda'), n_channel = 3):
    """This converts a (b,1,h,w) to repeat the color channel"""
    return I.repeat(1,n_channel,1,1)