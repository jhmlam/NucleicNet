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

import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback

from NucleicNet.Burn.DA import *

# ======================================
# Stock trainer
# ======================================
def DefaultTrainer00(DIR_TrainingRoot,DIR_TrainLog, 
                    User_ExperiementName = 'AUCG_B1hwResnet',
                    User_SizeMinibatch = 256,
                    User_ShiftLrRatio = 0.1,
                    User_GradientClippingValue = 10000.0,
                    User_Mixup = False):

    # NOTE this allow easier debugging; in NucleicNet AUCG it takes 2 hours for minibatch 128 running 5 epochs
    assert (User_SizeMinibatch % 16 == 0), "ABORTED. User_SizeMinibatch be multiple of 16"
    # NOTE AUCG relu can be trained for max_epochs = 30 batch 2048 each model takes 1/2 day.
    # NOTE AUCG gelu can be trainied for max_epochs = 20 batch 1024 each model takes 8 hours
    # NOTE SXPR takes much longer time to run through the whole dataset. Recommended at  max_epochs = 6; 3 already take 22 hours
    max_epochs = 20 # min(int(User_SizeMinibatch /(16 * 1)) + 1, 100)  # NOTE Take 44 hours for a 96 depth 40 width training in 20 epoch. 
    callbacklist = [pl.callbacks.ModelCheckpoint(save_weights_only=False, 
                                    mode="max", monitor="hp_metric", save_top_k=20,
                                    filename="{epoch}-{step}-{hp_metric}"),
                                    pl.callbacks.LearningRateMonitor('step'),
                                    pl.callbacks.LearningRateMonitor('epoch'),
                                    #pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, 
                                    #swa_lrs=1e-3 / 4, annealing_epochs=10, annealing_strategy='cos', 
                                    #avg_fn=None, device=torch.device('cuda')) # NOTE we can use swa to abruptly decrease LR at the swa_epoch_start
                                    ]
    if User_Mixup:
        callbacklist.append(MixUpCallback(alpha = 0.4, softmax_target = False))


    tb_logger = pl_loggers.TensorBoardLogger(save_dir = DIR_TrainLog, name = User_ExperiementName)
    csv_logger = pl_loggers.CSVLogger(save_dir = DIR_TrainLog, name = User_ExperiementName)
    trainer = pl.Trainer(
                        # NOTE Algorithm, Training Behaviour
                        #auto_lr_find=True,                                  # Learning Rate Optimisation
                        gradient_clip_algorithm='norm',                     # Gradient Clipping {"value" , "norm"}
                        gradient_clip_val=User_GradientClippingValue,                            # None witll disable GC # clip gradients' global norm to <= this number larger network may need larger clip? default 10000 TODO Test
                        #accumulate_grad_batches={0: 10, 4: 5, 8: 1} ,          # Accumulates grads every k batches or as set up in the dict.
                        amp_backend='native', amp_level=None,                 # Mixed precision (“native” or “apex”). The optimization level to use (O1, O2, etc…). By default it will be set to “O2” if amp_backend is set to “apex”.
                        precision=32,                                       # Native precisino to be overided by apex
                        #overfit_batches=0.0,                                # Overfit a fraction of training data (float) or a set number of batches (int).
                        sync_batchnorm=False,                               # Synchronize batch norm layers between process groups/whole world.
                        track_grad_norm=-1,                                # 1 no tracking. Otherwise tracks that p-norm. May be set to ‘inf’ infinity-norm. 
                                                                            # If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
                        deterministic=False,
                        replace_sampler_ddp=True,
                        auto_scale_batch_size=False,
                        multiple_trainloader_mode='max_size_cycle',
                        reload_dataloaders_every_n_epochs=0,


                        #auto_lr_find=True,
                        # NOTE Training Schedule
                        max_epochs=max_epochs,


                        # NOTE Hardware, Directories
                        default_root_dir=DIR_TrainingRoot,
                        resume_from_checkpoint= None,#"/home/homingla/Project-NucleicNet/Models/AUCG-B1hwLayerResnet5CV2018_AUCG-B1hwLayerResnet5CV2018/80_81/checkpoints/epoch=1-step=61025.ckpt", #,None, #DIR_Checkpoint,
                        num_nodes=1,                                        # Number of GPU nodes for distributed training.
                        num_processes=1,                                    # Number of processes for distributed training with accelerator="cpu".
                        gpus=1, auto_select_gpus=False,
                        benchmark=True,                                   # Cudnn benchmark True for faster perfromance. https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2

                        # NOTE Logging, Checkpoint
                        profiler="simple",
                        #logger=True,  
                        log_every_n_steps=50,  #enable_checkpointing=True, 
                        logger=[tb_logger,csv_logger],
                        check_val_every_n_epoch=1, val_check_interval=0.01,
                        # NOTE The hp_metric is the macro accuracy therefore mode should be max
                        callbacks=callbacklist,

                        # NOTE Tools
                        fast_dev_run=False,     # NOTE debugger of a network
                        #fast_dev_run=1000,
                        #detect_anomaly=False,   # NOTE DEtect nan etc Slows down
                        #num_sanity_val_steps=2,    
                    )
    return trainer


# ============================
# Debugging tools
# =============================
def DebugTrainer00(DIR_TrainingRoot,DIR_TrainLog, 
                    User_ExperiementName = 'AUCG_B1hwResnet',
                    User_SizeMinibatch = 256,
                    User_ShiftLrRatio = 0.1):

    # NOTE this allow easier debugging; in NucleicNet AUCG it takes 2 hours for minibatch 128 running 5 epochs
    assert (User_SizeMinibatch % 16 == 0), "ABORTED. User_SizeMinibatch be multiple of 16"

    max_epochs = max(int(User_SizeMinibatch /(16 * 1)) + 1, 100) 


    tb_logger = pl_loggers.TensorBoardLogger(save_dir = DIR_TrainLog, name = User_ExperiementName)
    csv_logger = pl_loggers.CSVLogger(save_dir = DIR_TrainLog, name = User_ExperiementName)
    trainer = pl.Trainer(
                        # NOTE Algorithm, Training Behaviour
                        #auto_lr_find=True,                                  # Learning Rate Optimisation
                        gradient_clip_algorithm='norm',                     # Gradient Clipping {"value" , "norm"}
                        gradient_clip_val=100.0,                            # None witll disable GC Advice to set to eps of precision
                        #accumulate_grad_batches={0: 10, 4: 5, 8: 1} ,          # Accumulates grads every k batches or as set up in the dict.
                        amp_backend='native', amp_level=None,                 # Mixed precision (“native” or “apex”). The optimization level to use (O1, O2, etc…). By default it will be set to “O2” if amp_backend is set to “apex”.
                        precision=32,                                       # Native precisino to be overided by apex
                        #overfit_batches=0.0,                                # Overfit a fraction of training data (float) or a set number of batches (int).
                        sync_batchnorm=False,                               # Synchronize batch norm layers between process groups/whole world.
                        track_grad_norm=2,                                # 1 no tracking. Otherwise tracks that p-norm. May be set to ‘inf’ infinity-norm. 
                                                                            # If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
                        deterministic=False,
                        replace_sampler_ddp=True,
                        auto_scale_batch_size=False,
                        multiple_trainloader_mode='max_size_cycle',
                        reload_dataloaders_every_n_epochs=0,


                        #auto_lr_find=True,
                        # NOTE Training Schedule
                        max_epochs=max_epochs,


                        # NOTE Hardware, Directories
                        default_root_dir=DIR_TrainingRoot,
                        resume_from_checkpoint= None,#"/home/homingla/Project-NucleicNet/Models/AUCG-B1hwLayerResnet5CV2018_AUCG-B1hwLayerResnet5CV2018/80_81/checkpoints/epoch=1-step=61025.ckpt", #,None, #DIR_Checkpoint,
                        num_nodes=1,                                        # Number of GPU nodes for distributed training.
                        num_processes=1,                                    # Number of processes for distributed training with accelerator="cpu".
                        gpus=1, auto_select_gpus=False,
                        benchmark=True,                                   # Cudnn benchmark True for faster perfromance. https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2

                        # NOTE Logging, Checkpoint
                        profiler="simple",
                        #logger=True,  
                        log_every_n_steps=50,  #enable_checkpointing=True, 
                        logger=[tb_logger,csv_logger],
                        check_val_every_n_epoch=1, val_check_interval=0.01,
                        # NOTE The hp_metric is the macro accuracy therefore mode should be max
                        callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=False, mode="max", monitor="hp_metric", save_top_k=20),
                                    pl.callbacks.LearningRateMonitor('step'),
                                    pl.callbacks.LearningRateMonitor('epoch'),
                                    #pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0, swa_lrs=None, annealing_epochs=10, annealing_strategy='cos', avg_fn=None, device=torch.device('cuda'))
                                    ],

                        # NOTE Tools
                        #fast_dev_run=False,     # NOTE debugger of a network
                        fast_dev_run=1000,
                        #detect_anomaly=True,   # NOTE DEtect nan etc Slows down
                        #num_sanity_val_steps=2,    
                    )
    return trainer









# ==================================
# OTher tools
# ===================================
def CheckpointTrainer00(DIR_TrainingRoot,DIR_TrainLog, DIR_Checkpoint,
                    User_ExperiementName = 'AUCG_B1hwResnet',
                    User_SizeMinibatch = 256,
                    User_ShiftLrRatio = 0.1):

    # NOTE this allow easier debugging; in NucleicNet AUCG it takes 2 hours for minibatch 128 running 5 epochs
    assert (User_SizeMinibatch % 16 == 0), "ABORTED. User_SizeMinibatch be multiple of 16"

    max_epochs = min(int(User_SizeMinibatch /(16 * 1)) + 1, 5) # NOTE at max 5 epochs for first stage trial, but still it can afford more epochs

    tb_logger = pl_loggers.TensorBoardLogger(save_dir = DIR_TrainLog, name = User_ExperiementName)
    csv_logger = pl_loggers.CSVLogger(save_dir = DIR_TrainLog, name = User_ExperiementName)
    trainer = pl.Trainer(
                        # NOTE Algorithm, Training Behaviour
                        #auto_lr_find=True,                                  # Learning Rate Optimisation
                        gradient_clip_algorithm='norm',                     # Gradient Clipping {"value" , "norm"}
                        gradient_clip_val=1e-3,                            # None witll disable GC Advice to set to eps of precision
                        #accumulate_grad_batches={0: 10, 4: 5, 8: 1} ,          # Accumulates grads every k batches or as set up in the dict.
                        amp_backend='native', amp_level=None,                 # Mixed precision (“native” or “apex”). The optimization level to use (O1, O2, etc…). By default it will be set to “O2” if amp_backend is set to “apex”.
                        precision=32,                                       # Native precisino to be overided by apex
                        #overfit_batches=0.0,                                # Overfit a fraction of training data (float) or a set number of batches (int).
                        sync_batchnorm=False,                               # Synchronize batch norm layers between process groups/whole world.
                        track_grad_norm= -1,                                # 1 no tracking. Otherwise tracks that p-norm. May be set to ‘inf’ infinity-norm. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
                        deterministic=False,
                        replace_sampler_ddp=True,
                        auto_scale_batch_size=False,
                        multiple_trainloader_mode='max_size_cycle',
                        reload_dataloaders_every_n_epochs=0,


                        #auto_lr_find=True,
                        # NOTE Training Schedule
                        max_epochs=max_epochs,


                        # NOTE Hardware, Directories
                        default_root_dir=DIR_TrainingRoot,
                        resume_from_checkpoint= DIR_Checkpoint,
                        num_nodes=1,                                        # Number of GPU nodes for distributed training.
                        num_processes=1,                                    # Number of processes for distributed training with accelerator="cpu".
                        gpus=1, auto_select_gpus=False,
                        benchmark=True,                                   # Cudnn benchmark True for faster perfromance. https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2

                        # NOTE Logging, Checkpoint
                        profiler="simple",
                        #logger=True,  
                        log_every_n_steps=50,  #enable_checkpointing=True, 
                        logger=[tb_logger,csv_logger],
                        check_val_every_n_epoch=1, val_check_interval=0.01,
                        # NOTE The hp_metric is the macro accuracy therefore mode should be max
                        callbacks=[pl.callbacks.ModelCheckpoint(save_weights_only=False, mode="max", monitor="hp_metric", save_top_k=20),
                                    pl.callbacks.LearningRateMonitor('step'),
                                    #pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0, swa_lrs=None, annealing_epochs=10, annealing_strategy='cos', avg_fn=None, device=torch.device('cuda'))
                                    ],

                        # NOTE Tools
                        fast_dev_run=False,     # NOTE debugger of a network
                        #detect_anomaly=False,   # NOTE DEtect nan etc Slows down
                        #num_sanity_val_steps=2,    
                    )
    return trainer



# ==========================
# Reset paramters
# ============================
def ResetAllParameters(model_):
    for layer in model_.children():
        if hasattr(layer, 'reset_parameters'):
            #print('resetd',layer)
            layer.reset_parameters()
        else:
            ResetAllParameters(layer)



def TorchEmptyCache():

    torch.cuda.empty_cache()    
    torch.cuda.memory_allocated(0)
    torch.cuda.max_memory_allocated(0)













# ===========================
# Bayesian? Misc
# ===========================
def Misc_prior(self, outputs, soft_targets, alpha = 0.1, beta = 0.1):

        p = torch.ones(self.n_class).cuda() / self.n_class

        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)

        L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
        L_p = -torch.sum(torch.log(avg_probs) * p)
        L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

        loss2 = L_c +  alpha * L_p + beta * L_e
        return loss2


# TODO This does not seem to work.
class OBSOLETE_Callback_ShiftLrBoundOnEpochEnd(Callback):
    def __init__(self, User_ShiftLrRatio = 0.8) -> None:
        self.User_ShiftLrRatio = User_ShiftLrRatio
        super().__init__()
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.User_min_lr = pl_module.User_min_lr * self.User_ShiftLrRatio
        pl_module.learning_rate = pl_module.learning_rate * self.User_ShiftLrRatio 
        pl_module.configure_optimizers()

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