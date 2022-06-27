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
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torchvision as tv
from NucleicNet import Burn, Fuel
import torchmetrics
import torch.nn as nn
import math
from torch import jit


# ==============================
# Dimension Protocols
# ==============================
# TODO This can be applied to 1D tensor batch for a dataloader.
class BVec_B1hw(nn.Module):
    def __init__(self, 
                    n_FeatPerShell = 80, 
                    n_Shell = 6,
                    ):
        if help:
            """
            # NOTE  This funcitional actually performs the inverse of vectorization
            # e.g.  The input here is some 1-D tensor f = (b,1,h*w) vectorized from some F = (b,1,h,w)
            #       f = Vec(F)
            #       where those 1-D tensor are structured by the index h e.g. shell in our case.
            #       then the index w is the feature index within a shell.  
            # NOTE  Remember Vec^-1(f) = F 
            #                         = ((Vec(I_w)^T kron I_h ) (I_w kron f) 
            """


        super().__init__()
        self.n_FeatPerShell = n_FeatPerShell
        self.n_Shell = n_Shell

    def forward(self, x):
        n_vector = x.shape[0]
        z = x.resize_(n_vector,self.n_Shell,self.n_FeatPerShell).unsqueeze_(1)
        return z


class Bnhw_Permute(jit.ScriptModule):
    def __init__(self, 
                    dimorder = (0,3,2,1),
                    ):
        if help:
            """
            # NOTE  This functional is handy to apply before Batch normalization which assumes (b,c,h,w)
            #       and the dimension of color channel will be fixed at the second dimension.
            #       In our case we perform convolution along the shell index h and 
            #       reserved index w as the feature index within a shell
            #       This is why we should permute as (0,3,2,1) from (0,1,2,3)
            """
        super().__init__()
        self.dimorder = dimorder
    @jit.script_method
    def forward(self, x):
        z = x.permute(*self.dimorder)
        return z

# ==========================
# Convolution
# ==========================

#class B1hw_ShellConv(nn.Module):
class B1hw_ShellConv(jit.ScriptModule):
    def __init__(self, 
                    n_FeatPerShell = 80, 
                    n_Shell = 6,
                    n_ShellMix = 2,
                    device = torch.device('cuda')
                    ):

        if help:
            """
            # NOTE This funcitional assumes an inverse of vectorized F  
            # e.g. The input here is some 1-D tensor f = (b,1,h*w) vectorized from some F = (b,1,h,w)
            #      f = Vec(F)
            #      where those 1-D tensor are structured by the index h e.g. shell in our case.
            #      then the index w is the feature index within a shell.  
            #      We want to convolve along the shell, find a useful combination among features in adjacent shells
            #      The last-padded mixture is discarded.
            #      TODO Downsample maybe implemented by recursion and stride.
            """
        #print(n_FeatPerShell)

        super().__init__()
        self.n_Shell = n_Shell
        self.kernel_size = (n_ShellMix, n_FeatPerShell)
        self.net = torch.nn.Conv2d(in_channels = 1, out_channels = n_FeatPerShell, 
                        kernel_size = self.kernel_size, 
                        padding=(n_ShellMix - 1 ,0), 
                        stride=1, dilation=1, groups=1, 
                        bias=False, padding_mode='zeros',  
                        device=None, dtype=None)
        self.selectshell = torch.arange(self.n_Shell, device= device)

    @jit.script_method
    def forward(self, x):
        z = self.net(x)
        z = z.permute(0,3,2,1)
        #print(z.shape)
        # NOTE Remove the last padding shell. also Handle kernel_size (n_ShellMix > 2, n_FeatPerShell)
        #      However, tending to FC mixing is not recommended. 

        z = z.index_select(2, self.selectshell)
        return z



# ========================
# Other useful loss
# =========================
import torch
import torch.nn.functional as F



# https://github.com/locuslab/projected_sinkhorn/blob/master/projected_sinkhorn/lambertw.py
# https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss_adaptive_gamma.py

def evalpoly(coeff, degree, z): 
    powers = torch.arange(degree,-1,-1).float().to(z.device)
    return ((z.unsqueeze(-1)**powers)*coeff).sum(-1)

def lambertw(z0, tol=1e-5): 
    OMEGA = 0.56714329040978387299997  # W(1, 0)
    EXPN1 = 0.36787944117144232159553  # exp(-1)
    # this is a direct port of the scipy version for the 
    # k=0 branch for *positive* z0 (z0 >= 0)

    # skip handling of nans
    if torch.isnan(z0).any(): 
        raise NotImplementedError

    w0 = z0.new(*z0.size())
    # under the assumption that z0 >= 0, then I_branchpt 
    # is never used. 
    I_branchpt = torch.abs(z0 + EXPN1) < 0.3
    I_pade0 = (-1.0 < z0)*(z0 < 1.5)
    I_asy = ~(I_branchpt | I_pade0)
    if I_pade0.any(): 
        z = z0[I_pade0]
        num = torch.Tensor([
            12.85106382978723404255,
            12.34042553191489361902,
            1.0
        ]).to(z.device)
        denom = torch.Tensor([
            32.53191489361702127660,
            14.34042553191489361702,
            1.0
        ]).to(z.device)
        w0[I_pade0] = z*evalpoly(num,2,z)/evalpoly(denom,2,z)

    if I_asy.any(): 
        z = z0[I_asy]
        w = torch.log(z)
        w0[I_asy] = w - torch.log(w)

    # split on positive and negative, 
    # and ignore the divergent series case (z=1)
    w0[z0 == 1] = OMEGA
    I_pos = (w0 >= 0)*(z0 != 1)
    I_neg = (w0 < 0)*(z0 != 1)
    if I_pos.any(): 
        w = w0[I_pos]
        z = z0[I_pos]
        for i in range(100): 
            # positive case
            ew = torch.exp(-w)
            wewz = w - z*ew
            wn = w - wewz/(w + 1 - (w + 2)*wewz/(2*w + 2))

            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_pos] = w

    if I_neg.any(): 
        w = w0[I_neg]
        z = z0[I_neg]
        for i in range(100):
            ew = torch.exp(w)
            wew = w*ew
            wewz = wew - z
            wn = w - wewz/(wew + ew - (w + 2)*wewz/(2*w + 2))
            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_neg] = wn
    return w0


def get_gamma(p):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*torch.log(p)))/(p*torch.log(p)))*torch.log(1-p)
    gamma_complex = (1-p)/(p*torch.log(p)) + lambertw(-y + 1e-12, k=-1)/torch.log(1-p)
    gamma = torch.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma

class SigmoidalFocalLoss(nn.Module):
    def __init__(self,alpha: float = 0.25,gamma: float = 2.0,
    reduction: str = "mean",label_smoothing: float = 0.0,
    UseGammaDict = False, device_ = torch.device("cuda")
    ):
        super().__init__()
        """
        https://arxiv.org/pdf/2002.09437.pdf
        https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
        """

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.UseGammaDict = UseGammaDict
        self.device_ = device_

    def forward( self,   inputs: torch.Tensor, targets: torch.Tensor,):
        # NOTE assume input is logits
        p = torch.sigmoid(inputs)
        if self.label_smoothing > 0.0:

            loss_ = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            ce_loss = loss_(inputs, targets)
        else:
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        p_t = p * targets + (1 - p) * (1 - targets)
        if self.UseGammaDict:

            pt_lt02 = (p_t <= 0.2)
            pt_lt05 = (p_t <=0.5)

            gamma = torch.ones_like(p_t, device = self.device_) * self.gamma
            gamma[pt_lt05] = 3.0
            gamma[pt_lt02] = 5.0

            loss = ce_loss * ((1 - p_t) ** gamma)
        else:
            loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss







# ========================
# Noise Tools
# ========================

class NoisyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.0125, bias: bool = True):

        """
        Reference
        Pyotorch Bolt
        https://arxiv.org/pdf/1706.10295.pdf
        """

        super().__init__(in_features, out_features, bias=bias)
        # NOTE The mu and sigma will be updated by gradient (adapt!) 

        weights = torch.full((out_features, in_features), sigma_init) 
        self.sigma_weight = nn.Parameter(weights)
        epsilon_weight = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", epsilon_weight)

        if bias:
            bias = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(bias)
            epsilon_bias = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", epsilon_bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std) # NOTE should we have kaiming/xavier normal init? why uniform?
        self.bias.data.uniform_(-std, std)

    def forward(self, input_x):

        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        noisy_weights = self.sigma_weight * self.epsilon_weight.data + self.weight

        return F.linear(input_x, noisy_weights, bias)



#class GhostBatchNorm2D(torch.nn.Module):
class GhostBatchNorm2D(jit.ScriptModule):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """
    # NOTE This means the minibatch size assumed multiple of 64
    def __init__(self, num_features, virtual_batch_size=64, eps=1e-05, momentum=0.1, affine=True,):
        super(GhostBatchNorm2D, self).__init__()

        self.input_dim = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm2d(num_features =self.input_dim, momentum=momentum)
        
    @jit.script_method
    def forward(self, x):
        # NOTE This updates the bn the number of chunks time.
        chunks = x.chunk(int(math.ceil(x.shape[0] / self.virtual_batch_size)), 0) 
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)

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