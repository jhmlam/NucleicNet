# =================================================================================
#    NucleicNet
#    Copyright (C) 2019-2022  Jordy Homing Lam, JHML. All rights reserved.
#    
#    Acknowledgement. 
#    JHML would like to thank Mingyi Xue and Joshua Chou for their patience and efforts 
#    in the debugging process and Prof. Xin Gao and Prof. Xuhui Huang for their 
#    continuous support.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    * Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
#    * Cite our work at Lam, J.H., Li, Y., Zhu, L. et al. A deep learning framework to predict binding preference of RNA constituents on protein surface. Nat Commun 10, 4941 (2019). https://doi.org/10.1038/s41467-019-12920-0
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================

import torch
import pytorch_lightning as pl
import math
import warnings
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import types



class MultistepCosineAnnealingLRS(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False, 
                    jumpeverystep = 50000, T_max = 2000, 
                    jumpfactor = 0.5, eta_min=1e-8, 
                    AssignSteps = None):
        
        """This will keep decreasing until 1e-8 is reached"""
        self.optimizer = optimizer
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups][0]
        self.eta_min = eta_min 
        self.T_max = T_max
        self.jumpeverystep = jumpeverystep
        self.jumpfactor = jumpfactor
        self.AssignSteps = AssignSteps
        lr_lambda = lambda epoch: self.a_complicated_formula(epoch)

        self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        self.historicratio = 1.0
        super(MultistepCosineAnnealingLRS, self).__init__(optimizer, last_epoch, verbose)

    def a_complicated_formula(self, epoch):
        if self.AssignSteps is None:
            if (epoch > 0) & (epoch % self.jumpeverystep == 0):
                print(epoch, self.jumpeverystep, self.base_lr  *self.historicratio  - self.eta_min)
                self.historicratio *= self.jumpfactor
        else:
            if (epoch > 0) & (epoch in self.AssignSteps):
                print(epoch, self.jumpeverystep, self.base_lr  *self.historicratio  - self.eta_min)
                self.historicratio *= self.jumpfactor                


        if (epoch / self.T_max < 2.0):
            lr_temp = self.eta_min + \
                            (self.base_lr  *self.historicratio  - self.eta_min) \
                                *(1 + math.sin((math.pi * epoch / self.T_max) - (math.pi /2) )) / 2
            lr_temp /= self.base_lr
            return lr_temp

        else:
            if (epoch % (2* self.T_max)) <= self.T_max   :

                lr_temp = self.eta_min + \
                                (self.base_lr  *self.historicratio  - self.eta_min) \
                                    *(1 + math.cos((math.pi * epoch / self.T_max)  )) / 2
                lr_temp /= self.base_lr
        
            else:
                #print('baba')
                lr_temp = self.eta_min + \
                                (self.base_lr  *self.historicratio  - self.eta_min) \
                                    *(1 - math.cos((math.pi * epoch / self.T_max) )) / 2
                lr_temp /= self.base_lr

        
        if lr_temp > self.eta_min - 1e-10:
            return lr_temp

        else:
            return self.eta_min 

    def state_dict(self):

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]



class SimpleMultistepCosineLRS(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False, 
                    jumpeverystep = 50000, T_max = 2000, 
                    jumpfactor = 0.5, eta_min=1e-8, 
                    AssignSteps = None):
        
        """This will keep decreasing until 1e-8 is reached"""
        self.optimizer = optimizer
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups][0]
        self.eta_min = eta_min 
        self.T_max = T_max
        self.jumpeverystep = jumpeverystep
        self.jumpfactor = jumpfactor
        self.AssignSteps = AssignSteps
        lr_lambda = lambda epoch: self.a_complicated_formula(epoch)

        self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        self.historicratio = 1.0
        super(SimpleMultistepCosineLRS, self).__init__(optimizer, last_epoch, verbose)

    def a_complicated_formula(self, epoch):
        if self.AssignSteps is None:
            if (epoch > 0) & (epoch % self.jumpeverystep == 0):
                print(epoch, self.jumpeverystep, self.base_lr  *self.historicratio  - self.eta_min)
                self.historicratio *= self.jumpfactor
        else:
            if (epoch > 0) & (epoch in self.AssignSteps):
                print(epoch, self.jumpeverystep, self.base_lr  *self.historicratio  - self.eta_min)
                self.historicratio *= self.jumpfactor                
        #lr_temp = self.eta_min + \
                  #      (self.base_lr  *self.historicratio  - self.eta_min) \
                  #          *(1 + math.cos(math.pi * epoch / self.T_max)) / 2
        lr_temp = self.eta_min + \
                        (self.base_lr  *self.historicratio  - self.eta_min) \
                            *(1 + math.sin((math.pi * epoch / self.T_max) - (math.pi /2) )) / 2
        lr_temp /= self.base_lr

        if lr_temp > self.eta_min - 1e-10:
            return lr_temp

        else:
            return self.eta_min 


    def state_dict(self):

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]




class DescendingCosineAnnealingLR_HalfEpoch(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False, 
                    n_step_per_epoch = 500000, T_max = 2000, 
                    User_ShiftLrRatio = 0.5,
                    eta_min=1e-6, 
                    final_eta_min = 1e-7):
        
        self.optimizer = optimizer
        self.User_ShiftLrRatio = User_ShiftLrRatio
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups][0]
        self.eta_min = eta_min 
        self.T_max = T_max
        self.final_eta_min = final_eta_min / self.base_lr

        self.half_epoch_end = int(n_step_per_epoch / 2)
        lr_lambda = lambda epoch: self.a_complicated_formula(epoch)

        self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)

        super(DescendingCosineAnnealingLR_HalfEpoch, self).__init__(optimizer, last_epoch, verbose)

    def a_complicated_formula(self, epoch):

        lr_temp = self.eta_min* ( math.pow(self.User_ShiftLrRatio ,((epoch / self.half_epoch_end)))) + \
                        (self.base_lr* ( math.pow(self.User_ShiftLrRatio ,((epoch / self.half_epoch_end)))) - self.eta_min) \
                            *(1 + math.cos(math.pi * epoch / self.T_max)) / 2
        lr_temp /= self.base_lr
        #lr_temp = math.abs(lr_temp)
        if lr_temp > self.final_eta_min:
            return lr_temp
        else:
            lr_temp2 = self.final_eta_min + \
                (self.base_lr*10 - self.final_eta_min) \
                    *(1 + math.cos(math.pi * epoch / self.T_max)) / 2
            return lr_temp2

    def state_dict(self):

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

# ==========================================
# Simple code base
# ==========================================
# NOTE https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
class EXAMPLE_LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(EXAMPLE_LambdaLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


# =================================================================================
#    NucleicNet
#    Copyright (C) 2019-2022  Jordy Homing Lam, JHML. All rights reserved.
#    
#    Acknowledgement. 
#    JHML would like to thank Mingyi Xue and Joshua Chou for their patience and efforts 
#    in the debugging process and Prof. Xin Gao and Prof. Xuhui Huang for their 
#    continuous support.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    * Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
#    * Cite our work at Lam, J.H., Li, Y., Zhu, L. et al. A deep learning framework to predict binding preference of RNA constituents on protein surface. Nat Commun 10, 4941 (2019). https://doi.org/10.1038/s41467-019-12920-0
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================