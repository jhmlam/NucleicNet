import torch
torch.multiprocessing.set_sharing_strategy('file_system')
class BasicMap(torch.utils.data.Dataset):

    def __init__(self, data, targets, 
                 transform  = None):
        
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index:int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


        