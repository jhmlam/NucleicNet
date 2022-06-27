import torch
torch.multiprocessing.set_sharing_strategy('file_system')
class BasicMap(torch.utils.data.Dataset):

    def __init__(self, data, targets, smoothenedtargets, 
                 transform  = None):
        
        self.data = data
        self.targets = targets
        self.smoothenedtargets = smoothenedtargets
        self.transform = transform

    def __getitem__(self, index:int):
        return self.data[index], self.targets[index], self.smoothenedtargets[index]

    def __len__(self):
        return self.data.shape[0]


        