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

import socket
from copy import deepcopy
from datetime import datetime
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference
# * https://pytorch-lightning-spells.readthedocs.io/
# * https://github.com/wangleiofficial/label-smoothing-pytorch
# * https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py





class LabelSmoothCrossEntropy(nn.Module):

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps


    def forward(self, preds, targets, weight=None):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        if weight is None:
            loss = -log_preds.sum(dim=-1).mean()
        else:
            loss = -(log_preds.sum(dim=-1) * weight.unsqueeze(0)).mean()
        nll = F.nll_loss(log_preds, targets, weight=weight)
        x = loss / n
        y = nll
        epsilon = self.eps
        return epsilon * x + (1 - epsilon) * y


class MixUpCallback(Callback):
    """
    Assumes the first dimension is batch.
    """

    def __init__(self, alpha: float = 0.4, softmax_target: bool = False):
        super().__init__()
        self.alpha = alpha
        self.softmax_target = softmax_target


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):

        old_batch = batch
        batch, targets = batch # NOTE asumed targets as class index
        targets = torch.nn.functional.one_hot(targets.type(torch.int64), num_classes=4)
        batch_flipped = batch.flip(0).clone()
        lambd = np.random.beta(self.alpha, self.alpha, batch.size(0))
        lambd = np.concatenate(
            [lambd[:, np.newaxis], 1-lambd[:, np.newaxis]], axis=1
        ).max(axis=1)
        # Create the tensor and expand (for batch inputs)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(batch.size())-1)]
        ).expand(-1, *batch.shape[1:])
        # Combine input batch
        new_batch = (batch * lambd_tensor +
                     batch_flipped * (1-lambd_tensor))
        # Create the tensor and expand (for target)
        lambd_tensor = batch.new(lambd).view(
            -1, *[1 for _ in range(len(targets.size())-1)]
        ).expand(-1, *targets.shape[1:])
        # Combine targets
        if self.softmax_target:
            new_targets = torch.stack([
                targets.float(), targets.flip(0).float(), lambd_tensor
            ], dim=1)
        else:
            new_targets = (
                targets * lambd_tensor +
                targets.flip(0) * (1-lambd_tensor)
            )
        old_batch[0] = new_batch
        old_batch[1] = new_targets



class MixupSoftmaxLoss(nn.Module):
    """A softmax loss that supports MixUp augmentation.
    It requires the input batch to be manipulated into certain format.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, reduction: str = 'mean', label_smooth_eps: float = 0):
        super().__init__()
        # setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction
        self.weight = class_weights
        # self.label_smooth_eps = label_smooth_eps
        if label_smooth_eps:
            self.loss_fn: Callable = LabelSmoothCrossEntropy(eps=label_smooth_eps)
        else:
            self.loss_fn = F.cross_entropy


    def forward(self, output: torch.Tensor, target):
        """The feed-forward.

        The target tensor should have three columns:

        1. the first class.
        2. the second class.
        3. the lambda value to mix the above two classes.

        Args:
            output (torch.Tensor): the model output.
            target (torch.Tensor): Shaped (batch_size, 3).

        Returns:
            torch.Tensor: the result loss
        """
        weight = self.weight
        if weight is not None:
            weight = self.weight.to(output.device)
        if len(target.size()) == 2:
            loss1 = self.loss_fn(output, target[:, 0].long(), weight=weight)
            loss2 = self.loss_fn(output, target[:, 1].long(), weight=weight)
            assert target.size(1) in (3, 4)
            if target.size(1) == 3:
                lambda_ = target[:, 2]
                d = (loss1 * lambda_ + loss2 * (1-lambda_)).mean()
            else:
                lamb_1, lamb_2 = target[:, 2], target[:, 3]
                d = (loss1 * lamb_1 + loss2 * lamb_2).mean()
        else:
            # This handles the cases without MixUp for backward compatibility
            d = self.loss_fn(output, target, weight=weight)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

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