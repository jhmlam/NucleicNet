import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import vstack as scipysparsevstack
import scipy

from NucleicNet import Fuel

def ScipysparseClassidx_Batch(batch:list): 

    data_batch, targets_batch = zip(*batch)

    # if returnsparse:
    #    data_batch = BatchScipyCsr_TorchSparseFloatTensor(data_batch)
    # else:
    
    data_batch = Fuel.T1.ScipyCsr_TorchFloat(data_batch)
    targets_batch = torch.ByteTensor(targets_batch)
    
    return data_batch, targets_batch