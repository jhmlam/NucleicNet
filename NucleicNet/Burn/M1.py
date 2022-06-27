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
#import torchvision as tv
from NucleicNet import Burn, Fuel
import torchmetrics
from NucleicNet.Burn import Catalogue_Activation
#from torchmetrics.functional import accuracy
import NucleicNet.Burn.T1
import NucleicNet.Burn.LRS
import NucleicNet.Burn.DA
# ==================================
# Naive adaptor
# ==================================
class B1hw_NaiveRepeatB3hwFcLogits(pl.LightningModule):
    def __init__(self, 
                    model   = None, # tv.models.resnet18(nested_module = True),
                    loss    = nn.CrossEntropyLoss(),
                    n_channel = 3,
                    n_class = 10):
        super().__init__()
        if help:

            """
            # NOTE This is a naive 'working' handle to take 1-D tensor forged to (b,1,h,w)
            #      and further forged to (b,3,h,w) by simply repeating the vector
            #      The critique here is that it does not handle batch-normalisation correctly, 
            #      because the meaning of color channel mismatches. (it 'smudges' all the features)
            #      This maybe considered the toy model to be compared for debugging purpose.
            #      However, it also offers a template to construct and model with a FcLogit output
            # NOTE The "model" flag assumes the last layer being named 'fc' for fully connected
            """

        self.n_channel = n_channel
        self.n_class = n_class
        self.nested_module = model

        # ==========================================
        # Construct a fc "suffix" adapter
        # ==========================================

        for _, lastfc in self.nested_module.named_modules():
            pass
        if str(lastfc).startswith("Linear"):
            fcsize = int(str(lastfc).split(", out_features=")[-1].split(",")[0])
            fcsize_infeature = int(str(lastfc).split("in_features=")[-1].split(",")[0])
        else:
            raise KeyError("ABORTED. Provided nested_module model does not end with a fc layer.")

        #self.suffixadaptor = nn.Linear(fcsize, self.n_class, bias=True)
        self.nested_module.fc = nn.Linear(fcsize_infeature, self.n_class, bias=True)

        # NOTE The FC output imply logits. This should be a loss accepting logit
        self.loss = loss 
        self.automatic_optimization = True
    def forward(self, x):
        #print(x.shape)    
        x = Fuel.T1.B1hw_Bnhw(x, n_channel = self.n_channel) # NOTE prefixadaptor
        x = self.nested_module(x)
        #x = self.suffixadaptor(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        ACC = torchmetrics.functional.accuracy(logits, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        PR = torchmetrics.functional.precision_recall(logits, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        self.log('train_precision', PR[0])
        self.log('train_recall', PR[1])
        self.log('train_acc', ACC)        
        self.log('train_loss_s', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        CE = nn.CrossEntropyLoss()
        loss = CE(logits, y) #self.loss(logits, y) # NOTE Validate without the random
        self.log('val_loss_s', loss, prog_bar=True)
        self.log("val_loss", loss)
        ACC = torchmetrics.functional.accuracy(logits, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        PR = torchmetrics.functional.precision_recall(logits, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        self.log('val_acc', ACC)
        self.log('val_recall', PR[1])
        self.log('val_precision', PR[0])
        return loss


    def predict_step(self, batch, batch_idx): 
        x, y = batch
        logits = self.forward(x)
        p = torch.softmax(logits, dim=1)
        return p


    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,150,300,600,1200,], gamma=0.1)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.99, 
                            verbose=True, threshold = 1e-5, min_lr= 1e-6, cooldown = 1000)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    'reduce_on_plateau': True,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss_s",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

# =================================
# Logit Template
# =================================
class B1hw_FcLogits(pl.LightningModule):
    def __init__(self, #learning_rate,
                    model   = None, 
                    User_Loss    = "CrossEntropyLoss", #nn.CrossEntropyLoss(),
                    hw_product = 480,
                    adaptive2dpool = (3,40),
                    #User_Activation = 'gelu',

                    n_class = 10, 
                    BiasInSuffixFc = True,
                    AddMultiLabelSoftMarginLoss = False, # NOTE Worsen stuff?
                    User_lr = 1e-05,
                    User_AdamW_weight_decay = 1e-2,
                    User_min_lr = 1e-6,
                    User_CooldownInterval = 753,
                    User_LrScheduler = 'CosineAnnealingLR',
                    User_ShiftLrRatio = 0.5,
                    # NOTE basic overfit
                    User_Dropoutp = 0.5,
                    User_AddL1 = 0.9,
                    User_NoiseX = 0.1,
                    User_NoiseY = 0.1,
                    User_NoiseZ = 0.1,
                    device = torch.device('cuda'),
                    ManualInitiation = False,
                    User_Mixup = False,
                    User_LabelSmoothing = 0.0,
                    User_NeighborLabelSmoothAngstrom = 1.5,
                    User_InputDropoutp = 0.1,
                    User_FocalLossAlpha = 0.25,
                    User_FocalLossGamma = 2,
                    **kwargs,
        ):
        super().__init__()
        if help:

            """
            # NOTE This is a template to take (b,1,h,w)
            #      The user-defined model has to be a torch functional with batch normalisation defined correctly.
            #      the last layer will be flattened to be fed to a Fc for logit traiinng.
            """
        #self.learning_rate = learning_rate 
        if User_lr is not None:
            self.learning_rate = User_lr
        self.n_class = n_class
        self.nested_module = model
        self.hw_product  = hw_product
        try:
            self.User_Activation = model.User_Activation
        except:
            self.User_Activation = 'relu'

        self.User_CooldownInterval = User_CooldownInterval
        self.User_min_lr = User_min_lr
        self.User_AdamW_weight_decay = User_AdamW_weight_decay
        self.User_LrScheduler = User_LrScheduler
        self.User_Dropoutp = User_Dropoutp
        self.AddMultiLabelSoftMarginLoss = AddMultiLabelSoftMarginLoss
        self.User_AddL1 = User_AddL1
        self.User_ShiftLrRatio = User_ShiftLrRatio
        self.User_NoiseX = User_NoiseX
        self.User_NoiseY = User_NoiseY
        self.User_Mixup = User_Mixup
        self.User_LabelSmoothing = User_LabelSmoothing
        self.User_NoiseZ = User_NoiseZ
        self.User_NeighborLabelSmoothAngstrom = User_NeighborLabelSmoothAngstrom
        self.User_InputDropoutp = User_InputDropoutp
        self.User_FocalLossAlpha = User_FocalLossAlpha
        self.User_FocalLossGamma = User_FocalLossGamma


        #self.AddL2 = AddL2
        self.device_ = device

        # Saving all entered hparam above
        self.save_hyperparameters(ignore = ["model_params", 'device', 'model', 'loss'])
        # ==========================================
        # Construct a fc "suffix" adapter
        # ==========================================
        self.prefix_layerD = nn.Sequential(
            nn.Dropout(p=self.User_InputDropoutp,inplace=False)
        )
        # Mapping to classification output
        self.suffix_layerA = nn.Sequential(
            #nn.AdaptiveAvgPool2d((adaptive2dpool[0],adaptive2dpool[1])), # NOTE referencing torchvision's implemetation a pooling layer can be addded
            nn.Flatten(start_dim=1, end_dim=- 1),
        )

        self.suffix_layerD = nn.Sequential(
                #nn.Dropout(p=self.User_Dropoutp, inplace=False),
                #NucleicNet.Burn.T1.NoisyLinear(self.hw_product, self.hw_product, bias=BiasInSuffixFc, sigma_init= self.User_NoiseZ),
                nn.Dropout(p=self.User_Dropoutp, inplace=False),
                nn.Linear(self.hw_product, self.hw_product, bias=BiasInSuffixFc),
                nn.Dropout(p=self.User_Dropoutp, inplace=False),
        )

        self.suffix_layerZ = nn.Sequential(
            nn.Linear(self.hw_product, self.n_class, bias=False)


        )

        if ManualInitiation:
            self._init_params()



        self.User_Loss = User_Loss # This is a string
        self.CatalogueLoss = {
            "CrossEntropyLoss" : nn.CrossEntropyLoss(label_smoothing=self.User_LabelSmoothing),
            "MixupSoftmaxLoss": NucleicNet.Burn.DA.MixupSoftmaxLoss(class_weights = None, reduction = 'mean', label_smooth_eps = self.User_LabelSmoothing),
            "SigmoidalFocalLoss": NucleicNet.Burn.T1.SigmoidalFocalLoss(alpha = self.User_FocalLossAlpha, gamma = self.User_FocalLossGamma,label_smoothing = self.User_LabelSmoothing),
            "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
        }

        self.loss = self.CatalogueLoss[self.User_Loss]
        self.automatic_optimization = True

    def forward(self, x):
        x = self.prefix_layerD(x)
        x = self.nested_module(x)
        # NOTE We cannot standardize the last layer as it will severely hurt performance. Adding noise here can be dangerous
        #if (self.training) & (self.User_NoiseZ > 0.0):
        #    x += self.User_NoiseZ *  torch.randn(x.shape, device = self.device_)# NOTE Noise added in training

        x = self.suffix_layerA(x)
        if self.User_Dropoutp > 0.0:
            x = self.suffix_layerD(x)
        x = self.suffix_layerZ(x)

        return x

    def training_step(self, batch, batch_idx):

        x, y, yyyy= batch 

        # Noise        
        x += self.User_NoiseX *  torch.randn(x.shape, device = self.device_)# NOTE Noise added in training
        logits = self.forward(x)

        p = F.softmax(logits,-1)
                                                                                                                       
        #loss = self.loss(logits, yyyy)
        loss = self.loss(logits, yyyy)

        if self.User_AddL1 > 0.0: 
            l1_reg = torch.tensor(0.0, device = self.device_)
            for _ , t in enumerate(self.parameters()):
                    l1_reg += torch.mean(torch.flatten(t).abs())
            loss += self.User_AddL1 * l1_reg

        # Visualisation scores
        if self.User_Mixup:
            y = torch.argmax(y, dim=1)




        #y_hot = torch.nn.functional.one_hot(y.type(torch.int64), num_classes=self.n_class)

        ACC = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        PR = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        ACC2 = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'weighted', num_classes = self.n_class)
        PR2 = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'weighted', num_classes = self.n_class)

        ACC_mi = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'micro', num_classes = self.n_class)
        #PR_mi = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'micro', num_classes = self.n_class)
        ACC_ma = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'macro', num_classes = self.n_class)
        #PR_ma = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'macro', num_classes = self.n_class)

        ACC2_mi = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'micro', num_classes = self.n_class)
        #PR2_mi = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'micro', num_classes = self.n_class)
        ACC2_ma = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'macro', num_classes = self.n_class)
        #PR2_ma = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'macro', num_classes = self.n_class)


        SF = self.CatalogueLoss["SigmoidalFocalLoss"] 
        SM = self.CatalogueLoss["MultiLabelSoftMarginLoss"] 
        self.log('train_softmargin',SM(p,yyyy) )
        self.log('train_sigmoidalfocal',SF(p,yyyy) )
        self.log('train_precision', PR[0])
        self.log('train_recall', PR[1])
        self.log('train_acc', ACC)        
        self.log('train_precision2', PR2[0])
        self.log('train_recall2', PR2[1])
        self.log('train_acc2', ACC2)   


        self.log('train_acc_mi', ACC_mi)
        #self.log('train_recall_mi', PR_mi[1])
        #self.log('train_precision_mi', PR_mi[0])
        self.log('train_acc2_mi', ACC2_mi)
        #self.log('train_recall2_mi', PR2_mi[1])
        #self.log('train_precision2_mi', PR2_mi[0])

        self.log('train_acc_ma', ACC_ma)
        #self.log('train_recall_ma', PR_ma[1])
        #self.log('train_precision_ma', PR_ma[0])
        self.log('train_acc2_ma', ACC2_ma)
        #self.log('train_recall2_ma', PR2_ma[1])
        #self.log('train_precision2_ma', PR2_ma[0])



        self.log('train_loss_s', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, yyyy = batch
        logits = self.forward(x)
        p = F.softmax(logits,-1)
        #CE = nn.CrossEntropyLoss()
        #loss = CE(logits, y) #self.loss(logits, y) # NOTE Validate without the random
        #loss_functional = nn.CrossEntropyLoss(label_smoothing=0.0)#self.User_LabelSmoothing)
        #loss = loss_functional(logits, y)
        loss = self.loss(logits, yyyy)

        #y_hot = torch.nn.functional.one_hot(y.type(torch.int64), num_classes=self.n_class)

        if self.User_Mixup:
            pass # NOTE Because mixup is not called in validation_start hook
        SF = self.CatalogueLoss["SigmoidalFocalLoss"] 
        SM = self.CatalogueLoss["MultiLabelSoftMarginLoss"] 
        self.log('val_softmargin',SM(p,yyyy) )
        self.log('val_sigmoidalfocal',SF(p,yyyy) )


        # NOTE I will just monitor L1
        l1_reg = torch.tensor(0.0, device = self.device_)
        for _ , t in enumerate(self.parameters()):
                l1_reg += torch.mean(torch.flatten(t).abs())

        self.log("L1", l1_reg)
        self.log('val_loss_s', loss, prog_bar=True)
        self.log("val_loss", loss)
        #print(p,y)
        ACC = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        PR = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'weighted', num_classes = self.n_class)
        ACC2 = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'weighted', num_classes = self.n_class)
        PR2 = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'weighted', num_classes = self.n_class)  

        ACC_mi = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'micro', num_classes = self.n_class)
        #PR_mi = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'micro', num_classes = self.n_class)
        ACC_ma = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'macro', num_classes = self.n_class)
        #PR_ma = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'macro', num_classes = self.n_class)

        ACC2_mi = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'micro', num_classes = self.n_class)
        #PR2_mi = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'micro', num_classes = self.n_class)
        ACC2_ma = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'macro', num_classes = self.n_class)
        #PR2_ma = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'macro', num_classes = self.n_class)


        self.log('val_acc', ACC)
        self.log('val_recall', PR[1])
        self.log('val_precision', PR[0])
        self.log('val_acc2', ACC2)
        self.log('val_recall2', PR2[1])
        self.log('val_precision2', PR2[0])

        self.log('val_acc_mi', ACC_mi)
        #self.log('val_recall_mi', PR_mi[1])
        #self.log('val_precision_mi', PR_mi[0])
        self.log('val_acc2_mi', ACC2_mi)
        #self.log('val_recall2_mi', PR2_mi[1])
        #self.log('val_precision2_mi', PR2_mi[0])

        self.log('val_acc_ma', ACC_ma)
        #self.log('val_recall_ma', PR_ma[1])
        #self.log('val_precision_ma', PR_ma[0])
        self.log('val_acc2_ma', ACC2_ma)
        #self.log('val_recall2_ma', PR2_ma[1])
        #self.log('val_precision2_ma', PR2_ma[0])

        self.log("hp_metric", ACC)


        return loss

    def predict_step(self, batch, batch_idx): 
        x, y = batch
        logits = self.forward(x)
        p = torch.softmax(logits, dim=1)
        return p

    def predict_p(self, batch): 
        x = batch
        logits = self.forward(x)
        p = torch.softmax(logits, dim=1)
        return p


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = self.User_AdamW_weight_decay)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
        #                            weight_decay=0.01, nesterov=True)
        #optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate, rho=0.9, eps=1e-06, weight_decay=0.001)


        lr_scheduler_ReduceLrEveryEpoch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                patience=1, factor=0.1, mode='min',
                                verbose=True,
                                threshold_mode='rel',
                                threshold = 0.01, # NOTE In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. 
                                min_lr= self.User_min_lr, cooldown = self.User_CooldownInterval)
        
        if self.User_LrScheduler == 'CosineAnnealingLR':
            lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.User_CooldownInterval, 
                                                                        eta_min=self.User_min_lr, 
                                                                        last_epoch=-1, verbose=False)
        if self.User_LrScheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.User_CooldownInterval, T_mult=1, 
                                                                        eta_min=self.User_min_lr, 
                                                                        last_epoch=-1, verbose=False)
        if self.User_LrScheduler == 'DescendingCosineAnnealingLR_HalfEpoch':
            lr_scheduler = NucleicNet.Burn.LRS.DescendingCosineAnnealingLR_HalfEpoch(optimizer, T_max = self.User_CooldownInterval, 
                                                                        eta_min=self.User_min_lr, 
                                                                        last_epoch=-1, verbose=False,
                                                                        n_step_per_epoch = 500000, 
                                                                        User_ShiftLrRatio = self.User_ShiftLrRatio,
                                                                        final_eta_min = 1e-10)
        if self.User_LrScheduler == 'SimpleMultistepCosineLRS':
            lr_scheduler = NucleicNet.Burn.LRS.SimpleMultistepCosineLRS(optimizer, last_epoch=-1, verbose=False, 
                                                                        jumpeverystep = 20000, 
                                                                        T_max = self.User_CooldownInterval, 
                                                                        jumpfactor = 0.5, eta_min=self.User_min_lr,
                                                                        AssignSteps = [10000, 160000]
                                                                        )
        if self.User_LrScheduler == 'SimpleMultistepCosineLRS_SXPR':
            lr_scheduler = NucleicNet.Burn.LRS.SimpleMultistepCosineLRS(optimizer, last_epoch=-1, verbose=False, 
                                                                        jumpeverystep = 20000, 
                                                                        T_max = self.User_CooldownInterval, 
                                                                        jumpfactor = 0.5, eta_min=self.User_min_lr,
                                                                        AssignSteps = [10000, 160000]
                                                                        )
        if self.User_LrScheduler == 'MultistepCosineAnnealingLRS':
            lr_scheduler = NucleicNet.Burn.LRS.MultistepCosineAnnealingLRS(optimizer, last_epoch=-1, verbose=False, 
                                                                        jumpeverystep = 20000, 
                                                                        T_max = self.User_CooldownInterval, 
                                                                        jumpfactor = 0.5, eta_min=self.User_min_lr,
                                                                        AssignSteps = [10000] #[40000, 80000, 160000]
                                                                        )



        return (
                    {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": lr_scheduler,
                            'reduce_on_plateau': True,
                            "interval": "step",
                            "frequency": 1,
                            "monitor": "train_loss_s",
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                        },
                    },
                    
        )

    def _init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=Catalogue_Activation[self.User_Activation])
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)





# We should try to move the inverse-vec here. possibly also some eigvec here after B1hw we can use einsum
# TODO TODO BVec_FcLogits(pl.LightningModule)


# =================================
# Resnet 
# =================================
from torch import jit
# NOTE These are customised to take radial shell inputs. with B1hw (h is the shell, w is the feature)
class B1hw_BlockPreActResnet(jit.ScriptModule):
    def __init__(self, 
                    n_FeatPerShell = 80, 
                    n_Shell = 6,
                    n_ShellMix = 2,
                    act_fn = Catalogue_Activation['gelu'],
                    blockposition = 0,
                    User_NoiseZ = None,
                    device = torch.device('cuda'),
                    ):
        super().__init__()
        self.User_NoiseZ = User_NoiseZ
        self.blockposition = blockposition
        self.device_ = device
        if help:
            """
            # NOTE For some 1-D tensor f = (b,1,h*w) vectorized from some F = (b,1,h,w)
            #      f = Vec(F) NOTE This is done in T1.BVec_B1hw
            #      The input here is some F = (b,1,h,w)
            #      where f was structured by shell index h; index w is the feature index within a shell.  
            #      The output is a Resnet block with color channel batch normalised correctly
            #      https://arxiv.org/abs/1603.05027
            #      TODO Downsample maybe implemented by recursion and stride.
            """
        #print('bpreact, n_feat',n_FeatPerShell)
        self.net = nn.Sequential(
            # NOTE Batch normalisation done right woth Preactivation
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
        )
        self.net2a = nn.Sequential(
            #nn.BatchNorm2d(num_features = n_FeatPerShell),
            NucleicNet.Burn.T1.GhostBatchNorm2D(num_features= n_FeatPerShell, virtual_batch_size=128, eps=1e-05, momentum=0.1, affine=True),
        )
        self.net2b= nn.Sequential(
            act_fn(),
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
        )
        self.net3 = nn.Sequential(
            # NOTE Convolve along shell extracting useful combination of features in adjacent shells
            NucleicNet.Burn.T1.B1hw_ShellConv(n_FeatPerShell = n_FeatPerShell, n_Shell = n_Shell, n_ShellMix = n_ShellMix), 
 
            # NOTE  Batch normalisation done right woth Preactivation
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
            #nn.BatchNorm2d(num_features = n_FeatPerShell),
            NucleicNet.Burn.T1.GhostBatchNorm2D(num_features= n_FeatPerShell, virtual_batch_size=128, eps=1e-05, momentum=0.1, affine=True),
            act_fn(),
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),

            #nn.AdaptiveMaxPool2d((int(n_Shell/2), int(n_FeatPerShell/2))),
            #torch.nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, 
            #                        output_padding=0, groups=1, bias=False, dilation=1, 
            #                        padding_mode='zeros', device=None, dtype=None),

            # NOTE Convolve along shell extracting useful combination of features in adjacent shells
            NucleicNet.Burn.T1.B1hw_ShellConv(n_FeatPerShell = n_FeatPerShell,n_Shell = n_Shell, n_ShellMix = n_ShellMix),
        )

    @jit.script_method
    def forward(self, x):

        z = self.net(x)
        z = self.net2a(z)
        if (self.blockposition <= 2) & (self.training) & (self.User_NoiseZ > 0.0):
            z += self.User_NoiseZ *  torch.randn(z.shape, device = self.device_)
        z = self.net2b(z)
        z = self.net3(z)
        out = z + x
        return out


#class B1hw_BlockBottleneckResnet(nn.Module):
class B1hw_BlockBottleneckResnet(jit.ScriptModule):
    def __init__(self, 
                    n_FeatPerShell = 80, 
                    n_channelbottleneck = 40,
                    n_Shell = 6,
                    n_ShellMix = 2,
                    act_fn = Catalogue_Activation['gelu'],
                    device = torch.device('cuda'),
                    blockposition = 0,
                    User_NoiseZ = None,
                    ):
        super().__init__()

        if help:
            """
            # NOTE For some 1-D tensor f = (b,1,h*w) vectorized from some F = (b,1,h,w)
            #      f = Vec(F) NOTE This is done in T1.BVec_B1hw
            #      The input here is some F = (b,1,h,w)
            #      where f was structured by shell index h; index w is the feature index within a shell.  
            #      The output is a Resnet block with color channel batch normalised correctly
            #      https://arxiv.org/abs/1603.05027
            #      TODO Downsample maybe implemented by recursion and stride.
            # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
            # while original implementation places the stride at the first 1x1 convolution(self.conv1)
            # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
            # This variant is also known as ResNet V1.5 and improves accuracy according to
            # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
            """
        self.n_Shell = n_Shell
        self.selectshell = torch.arange(self.n_Shell, device= device)
        self.act_fn = act_fn()
        self.net1 = nn.Sequential(
            # NOTE Batch normalisation done right woth Preactivation
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
            nn.Conv2d(n_FeatPerShell, n_channelbottleneck, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm2d(num_features = n_channelbottleneck),
            NucleicNet.Burn.T1.GhostBatchNorm2D(num_features= n_channelbottleneck, virtual_batch_size=128, eps=1e-05, momentum=0.1, affine=True),
        )
        self.net2a = nn.Sequential(
            act_fn(),
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
 
            #nn.AdaptiveMaxPool2d((int(n_Shell/2), int(n_channelbottleneck/2))),
            #torch.nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, 
            #                        output_padding=0, groups=1, bias=False, dilation=1, 
            #                        padding_mode='zeros', device=None, dtype=None),

        )
        self.widedrop = nn.Dropout(p=0.1, inplace=False)
        self.net2b = nn.Sequential(
            
            # NOTE Convolve along shell extracting useful combination of features in adjacent shells
            NucleicNet.Burn.T1.B1hw_ShellConv(n_FeatPerShell = n_channelbottleneck, n_Shell = n_Shell, n_ShellMix = n_ShellMix), 
 
            # NOTE  Batch normalisation done right woth Preactivation
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
            #nn.BatchNorm2d(num_features = n_channelbottleneck),
            NucleicNet.Burn.T1.GhostBatchNorm2D(num_features= n_channelbottleneck, virtual_batch_size=128, eps=1e-05, momentum=0.1, affine=True),
            act_fn(),
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),

            # NOTE Convolve along shell extracting useful combination of features in adjacent shells
            #NucleicNet.Burn.T1.B1hw_ShellConv(n_FeatPerShell = n_FeatPerShell,n_Shell = n_Shell, n_ShellMix = n_ShellMix),
            nn.Conv2d(in_channels = 1, out_channels = n_FeatPerShell, 
                        kernel_size = (n_ShellMix,n_channelbottleneck), 
                        padding=(n_ShellMix - 1 ,0), 
                        stride=1, dilation=1, groups=1, 
                        bias=False, padding_mode='zeros'),  
                        #device=None, dtype=None),
            NucleicNet.Burn.T1.Bnhw_Permute(dimorder=(0,3,2,1)),
            
        )
        self.blockposition = blockposition
        self.device_ = device
        assert (User_NoiseZ is not None), 'ABORTED. B1hw_BlockBottleneckResnet User_NoiseZ is None'
        
        self.User_NoiseZ = User_NoiseZ

    @jit.script_method
    def forward(self, x):

        z = self.net1(x)

        if (self.blockposition <= 2) & (self.training) & (self.User_NoiseZ > 0.0):
            z += self.User_NoiseZ *  torch.randn(z.shape, device = self.device_)
        z = self.net2a(z)
        #if (self.blockposition <= 2): 
        #    z=self.widedrop(z)
        # NOTE Worsen if drop here.
        z = self.net2b(z)
        #print(z.shape)
        z = z.index_select(2, self.selectshell)
        out = z + x
        out = self.act_fn(out)
        return out



Catalogue_BlockResnet = {"B1hw_BlockPreActResnet" : B1hw_BlockPreActResnet,
                        "B1hw_BlockBottleneckResnet" : B1hw_BlockBottleneckResnet}


# NOTE Resnet 
#class B1hw_LayerResnet(nn.Module):
class B1hw_LayerResnet(jit.ScriptModule):
    def __init__(self, 
                    n_FeatPerShell = 80, 
                    n_Shell = 6,
                    n_ShellMix = 2,
                    User_Activation = 'gelu',
                    User_Block = "B1hw_BlockPreActResnet",
                    n_Blocks = 16,
                    #ShellIPCA = None,
                    ManualInitiation = False,
                    device = torch.device('cuda')):
        super(B1hw_LayerResnet, self).__init__()

        if help:
            """
            # NOTE For some 1-D tensor f = (b,1,h*w) vectorized from some F = (b,1,h,w)
            #      f = Vec(F) NOTE This is done in T1.BVec_B1hw
            #      The input here is some F = (b,1,h,w)
            #      where f was structured by shell index h; index w is the feature index within a shell.  
            #      The output is a series of Resnet block
            #      TODO Downsample maybe implemented by recursion and stride.
            """

        self.n_Blocks = n_Blocks
        self.User_Activation = User_Activation
        self.User_Block = User_Block

        self.n_FeatPerShell = n_FeatPerShell
        self.n_Shell = n_Shell
        self.n_ShellMix = n_ShellMix


        self._create_network()
        if ManualInitiation:
            self._init_params()
        #self.save_hyperparameters()

    def _create_network(self):
      blocks = []
      for _ in range(self.n_Blocks):
            current_Block_Functional = Catalogue_BlockResnet[self.User_Block]
            current_Activation_Functional = Catalogue_Activation[self.User_Activation]
            blocks.append(
                current_Block_Functional(
                    n_FeatPerShell = self.n_FeatPerShell, 
                    n_Shell = self.n_Shell,
                    n_ShellMix = self.n_ShellMix,
                    act_fn = current_Activation_Functional)
                # TODO With bottleneck layer
                    )
            
      self.blocks = nn.Sequential(*blocks)

    def _init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=Catalogue_Activation[self.HparamDict["Activation"]])
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    @jit.script_method
    def forward(self, x):

        x = self.blocks(x)
        return x

#class B1hw_LayerResnetBottleneck(nn.Module):
class B1hw_LayerResnetBottleneck(jit.ScriptModule):
    def __init__(self, 
                    n_FeatPerShell = 80, 
                    n_Shell = 6,
                    n_ShellMix = 2,
                    User_Activation = 'gelu',
                    User_n_channelbottleneck = 40,
                    User_Block = "B1hw_BlockPreActResnet",
                    n_Blocks = 16,
                    User_NoiseZ = 0.01,
                    #ShellIPCA = None,
                    ManualInitiation = False,
                    device = torch.device('cuda')):
        super(B1hw_LayerResnetBottleneck, self).__init__()

        if help:
            """
            # NOTE For some 1-D tensor f = (b,1,h*w) vectorized from some F = (b,1,h,w)
            #      f = Vec(F) NOTE This is done in T1.BVec_B1hw
            #      The input here is some F = (b,1,h,w)
            #      where f was structured by shell index h; index w is the feature index within a shell.  
            #      The output is a series of Resnet block
            #      TODO Downsample maybe implemented by recursion and stride.
            """

        self.n_Blocks = n_Blocks
        self.User_Activation = User_Activation
        self.User_Block = User_Block
        self.User_NoiseZ = User_NoiseZ
        self.n_FeatPerShell = n_FeatPerShell
        self.n_Shell = n_Shell
        self.n_ShellMix = n_ShellMix
        self.n_channelbottleneck = User_n_channelbottleneck

        self._create_network()
        if ManualInitiation:
            self._init_params()
        #self.save_hyperparameters()

    def _create_network(self):
      blocks = []
      for _ in range(int(self.n_Blocks/2)):
            current_Block_Functional = Catalogue_BlockResnet[self.User_Block]
            current_Block_Bottleneck = Catalogue_BlockResnet["B1hw_BlockBottleneckResnet"]
            current_Activation_Functional = Catalogue_Activation[self.User_Activation]
            blocks.append(
                current_Block_Functional(
                    n_FeatPerShell = self.n_FeatPerShell, 
                    n_Shell = self.n_Shell,
                    n_ShellMix = self.n_ShellMix,
                    act_fn = current_Activation_Functional,
                    blockposition = _ ,
                    User_NoiseZ = self.User_NoiseZ
                    )
                    )
            blocks.append(
                current_Block_Bottleneck(
                    n_FeatPerShell = self.n_FeatPerShell, 
                    n_channelbottleneck = self.n_channelbottleneck,
                    n_Shell = self.n_Shell,
                    n_ShellMix = self.n_ShellMix,
                    act_fn = current_Activation_Functional,
                    blockposition = _ ,
                    User_NoiseZ = self.User_NoiseZ)

                    )
            
      self.blocks = nn.Sequential(*blocks)
    



    def _init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=Catalogue_Activation[self.HparamDict["Activation"]])
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    @jit.script_method
    def forward(self, x):

        x = self.blocks(x)
        #x = self.normlayer(x) # NOTE This is NOT a good idea. batch normalising the last layer will hurt performance
        return x
# =========================
# Legacy B1hw_FcLogits Config for optimizer
# =========================




def LEGACY_configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = self.User_AdamW_weight_decay)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate, rho=0.9, eps=1e-06, weight_decay=0.001)
        if self.User_LrScheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.951, 
                                verbose=False,
                                # NOTE This threshold is the "within epsilon" for the loss to be counted as unchanged
                                # should be around 1e-4 to 5 
                                # it alsoimplies the lowest minima acheivable,
                                threshold = 5e-5, 
                                min_lr= self.User_min_lr, cooldown = self.User_CooldownInterval)
        if self.User_LrScheduler == 'CosineAnnealingLR':
            lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.User_CooldownInterval, 
                                                                        eta_min=self.User_min_lr, 
                                                                        last_epoch=-1, verbose=False)
        if self.User_LrScheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.User_CooldownInterval, T_mult=1, 
                                                                        eta_min=self.User_min_lr, 
                                                                        last_epoch=-1, verbose=False)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    'reduce_on_plateau': True,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "train_loss_s",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

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