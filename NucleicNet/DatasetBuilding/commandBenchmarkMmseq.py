# ============== Click Please.Imports
import sys
import glob
import gc


import random
random.seed(42)
import pandas as pd

import torch
import seaborn as sns
import time
import matplotlib.pyplot as plt



import tqdm



# ================
# Torch related
# ==============
import torch 
from torch import nn



import pytorch_lightning as pl
import torchmetrics

# Turn on cuda optimizer
#print(torch.backends.cudnn.is_available())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# disable debugs NOTE use only after debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
# Disable gradient tracking
torch.no_grad()


# =============
# NN
# =================
sys.path.append('../')
from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandReadPdbFtp import ReadMmseqExternalSymmetry, MakeMmseqClanGraph
from NucleicNet.DatasetBuilding.commandDataFetcherMmseq import FetchIndex, FetchTask, FetchDataset
from NucleicNet import Burn, Fuel
import NucleicNet.Burn.M1
import NucleicNet.Burn.util


# =========================
# Benchmark Organiser
# =========================
class BenchmarkHelper:
    def __init__(self,
        model = None,
        DIR_DerivedData = "../Database-PDB/DerivedData/",
        DIR_Typi = "../Database-PDB/typi/",
        DIR_FuelInput = "../Database-PDB/feature/",
        DIR_Benchmark = '../Benchmark/',
        n_CrossFold = 9,
        ClanGraphBcPercent = 90,
        User_featuretype = 'altman',
        User_DesiredBatchDatasize    = 7000000,
        User_Task = "AUCG",
        n_row_maxhold = 10000,
        #Df_grand = None,
        n_datasetworker = 16,
        device_ = torch.device('cuda'),
        User_SxprDatastride = 10,

    ):
        MkdirList([DIR_Benchmark])

        assert model is not None, "ABORTED. Supply a model"
        self.model = model
        self.model.to(device_)
        self.model.eval()
        
        self.DIR_DerivedData = DIR_DerivedData 
        self.DIR_Typi = DIR_Typi 
        self.DIR_FuelInput = DIR_FuelInput 
        self.n_CrossFold = n_CrossFold 
        self.ClanGraphBcPercent = ClanGraphBcPercent 
        self.User_featuretype = User_featuretype 
        self.User_DesiredBatchDatasize    = User_DesiredBatchDatasize  
        self.User_Task = User_Task 
        self.n_row_maxhold = n_row_maxhold        

        self.User_TrainValSplit = 0.5
        self.n_datasetworker = n_datasetworker
        self.User_SxprDatastride = User_SxprDatastride
        # ==================
        # Initialise
        # ==================
        self._Initialise_CrossFoldScope()





    # ================================
    # Execution Bodies
    # ===============================
    def Fetch_FoldPdbidBatches(self,ckp_hyperparam):
        # ============================
        # Cross-Folds Pdbids
        # ============================
        print("Getting TrainValTest batches")

        User_PdbidTraining = ckp_hyperparam["User_PdbidTraining"]
        User_PdbidValidation = ckp_hyperparam["User_PdbidValidation"]
        User_PdbidTesting = ckp_hyperparam["User_PdbidTesting"]

        return User_PdbidTraining, User_PdbidValidation, User_PdbidTesting



    # NOTE This is a subroutine...
    def Diagnose(self, Testing_PdbidBatches, ckp_hyperparam):
        """
        This snippet will run a given model defined at initiation on a set of Testing_PdbidBatches 
        """
        benchmark_df = []
        with torch.no_grad():
            Testing_PdbidBatches_ = Testing_PdbidBatches#[:3]


            pdbid_done = []
            for trial in range(100):
                for pdbid in tqdm.tqdm(Testing_PdbidBatches_):
                    #print(self.ClassName_ClassIndex_Dict, pdbid, self.User_Task)
                    if pdbid in pdbid_done:
                        continue
                    if trial > 0:
                        print('Trying %s for %s-th time' %(pdbid, trial+1))
                    try:
                        PART0_LoadDataset = True
                        if PART0_LoadDataset:
                            if self.User_Task == 'AUCG':
                                ds_ = self.FetchDatasetC.GetDataset(
                                                    Assigned_PdbidBatch = [pdbid],
                                                    ClassName_ClassIndex_Dict = self.ClassName_ClassIndex_Dict,
                                                    Assigned_PdbidWeight = None,
                                                    User_Task = self.User_Task,
                                                    PerformZscoring = True, PerformReduction = ckp_hyperparam['User_PerformReduction'],
                                                    TestingPhase = True
                                                    )
                            else:
                                ds_ = self.FetchDatasetC.GetDataset(
                                                    Assigned_PdbidBatch = [pdbid],
                                                    ClassName_ClassIndex_Dict = self.ClassName_ClassIndex_Dict,
                                                    Assigned_PdbidWeight = None,
                                                    User_Task = self.User_Task,
                                                    PerformZscoring = True, PerformReduction = ckp_hyperparam['User_PerformReduction'],
                                                    TestingPhase = True, 
                                                    User_datastride = self.User_SxprDatastride  # NOTE 10-30 depend on how much mmeory we have; 5 starts to complain about memroy. It only applies to Nonsite
                                                                                                # NOTE That I have also tested that it will raise/decrease accuracy when less/more datastride is used
                                                                        #      Apparently this behavior is due to a more/less compelete retrieval of samples from the inner volume shell 
                                                                        #      Say, drawing samples uniformly without replacement from a sphere containing a lattice we should expect it's more probable to sample points from the 
                                                                        #      outer shell volume of the sphere to be drawn. In our case, these are points containing less texture in the Altman feature.
                                                                        #      But as drawing more exampolkes without replacement, we start getting more points from the inner shell, approximating the accuracy closer to full data i.e. User_datastride == 1.
                                                    ) 
                            test_loader_t          = torch.utils.data.DataLoader(ds_, 
                                                                                batch_size=1024, 
                                                                                drop_last=False, num_workers=4, 
                                                                                pin_memory=True,worker_init_fn=None, 
                                                                                prefetch_factor=3, persistent_workers=False,
                                                                                shuffle=False)

                        PART1_EnumerateInBatch = True
                        if PART1_EnumerateInBatch:
                            y_list = []
                            p_list = []                
                            for _, xy in enumerate(test_loader_t):

                                x = xy[0] #torch.rand((1024,1,6,80), device = device)
                                y = xy[1]
                                x = x.to('cuda')
                                y = y.to('cuda')
                                p = self.model.predict_step((x,y),0)
                                y_list.append(y)
                                p_list.append(p)

                                del x,y
                                gc.collect()

                            y = torch.cat(y_list)
                            p = torch.cat(p_list)
                            del test_loader_t, ds_
                            NucleicNet.Burn.util.TorchEmptyCache()
                            gc.collect()               

                        PART2_GatherPerformance = True
                        if PART2_GatherPerformance:
                            ACC = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'weighted', num_classes = self.n_class)
                            PR = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'weighted', num_classes = self.n_class)
                            ACC_mi = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'micro', num_classes = self.n_class)
                            PR_mi = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'micro', num_classes = self.n_class)
                            ACC_ma = torchmetrics.functional.accuracy(p, y, top_k =1, average = 'macro', num_classes = self.n_class)
                            PR_ma = torchmetrics.functional.precision_recall(p, y, top_k =1, average = 'macro', num_classes = self.n_class)
                            y_hot = torchmetrics.utilities.data.to_onehot(y, num_classes=self.n_class)
                            p_hot = p.argmax(1)
                            CF = torchmetrics.functional.confusion_matrix(p_hot, y, num_classes = self.n_class, 
                            normalize='true', threshold=0.5, multilabel=False)
                            ACC2 = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'weighted', num_classes = self.n_class)
                            PR2 = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'weighted', num_classes = self.n_class)
                            ACC2_mi = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'micro', num_classes = self.n_class)
                            PR2_mi = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'micro', num_classes = self.n_class)
                            ACC2_ma = torchmetrics.functional.accuracy(p, y, top_k =2, average = 'macro', num_classes = self.n_class)
                            PR2_ma = torchmetrics.functional.precision_recall(p, y, top_k =2, average = 'macro', num_classes = self.n_class)
                            benchmark_df.append([pdbid, 
                                                float(ACC.cpu().numpy()), 
                                                float(PR[0].cpu().numpy()), 
                                                float(PR[1].cpu().numpy()), 
                                                CF.cpu().numpy(),
                                                float(ACC2.cpu().numpy()), 
                                                float(PR2[0].cpu().numpy()), 
                                                float(PR2[1].cpu().numpy()), 

                                                float(ACC_mi.cpu().numpy()), 
                                                float(PR_mi[0].cpu().numpy()), 
                                                float(PR_mi[1].cpu().numpy()), 
                                                float(ACC_ma.cpu().numpy()), 
                                                float(PR_ma[0].cpu().numpy()), 
                                                float(PR_ma[1].cpu().numpy()), 

                                                float(ACC2_mi.cpu().numpy()), 
                                                float(PR2_mi[0].cpu().numpy()), 
                                                float(PR2_mi[1].cpu().numpy()), 
                                                float(ACC2_ma.cpu().numpy()), 
                                                float(PR2_ma[0].cpu().numpy()), 
                                                float(PR2_ma[1].cpu().numpy()), 
                                                ]
                                            )

                            #print(pdbid,float(ACC.cpu().numpy()), 
                            #            float(PR[0].cpu().numpy()), 
                            #            float(PR[1].cpu().numpy()))
                            
                            del ACC, PR, y_hot, p_hot,CF, ACC2,PR2, ACC2_mi,PR2_mi, ACC2_ma,PR2_ma, ACC_mi,PR_mi, ACC_ma,PR_ma, 
                            NucleicNet.Burn.util.TorchEmptyCache()
                            gc.collect()
                        time.sleep(1)
                        pdbid_done.append(pdbid)

                    except RuntimeError:
                        NucleicNet.Burn.util.TorchEmptyCache()
                        NucleicNet.Burn.util.TorchEmptyCache()
                        NucleicNet.Burn.util.TorchEmptyCache()
                        print("Cuda run out of memory at %s. Will be re-tried next round" %pdbid)
                        time.sleep(5)


        benchmark_df = pd.DataFrame(benchmark_df, 
                        columns=['Pdbid', 'Accuracy-Top1', 'Precision-Top1','Recall-Top1', 'Confusion',
                                            'Accuracy-Top2', 'Precision-Top2','Recall-Top2',
                                            'Accuracy-Top1(micro)', 'Precision-Top1(micro)','Recall-Top1(micro)',
                                            'Accuracy-Top1(macro)', 'Precision-Top1(macro)','Recall-Top1(macro)',
                                            'Accuracy-Top2(micro)', 'Precision-Top2(micro)','Recall-Top2(micro)',
                                            'Accuracy-Top2(macro)', 'Precision-Top2(macro)','Recall-Top2(macro)',
                                            ]
                        )

        return benchmark_df

    # NOTE This is a wrapper for diagnose
    def Testing(self, ckp_hyperparam, 
                ):
        #self.ckp_hyperparam = ckp_hyperparam
        FoldIndex = ckp_hyperparam['User_SelectedCrossFoldIndex']
        Train_PdbidBatches, Val_PdbidBatches, Testing_PdbidBatches = self.Fetch_FoldPdbidBatches( ckp_hyperparam = ckp_hyperparam)
        test_benchmark = self.Diagnose(Testing_PdbidBatches, ckp_hyperparam)
        test_benchmark.loc[:,"Status"] = "Testing"
        NucleicNet.Burn.util.TorchEmptyCache()
        val_benchmark = self.Diagnose(Val_PdbidBatches, ckp_hyperparam)
        val_benchmark.loc[:,"Status"] = "Validation"
        NucleicNet.Burn.util.TorchEmptyCache()
        benchmark_df = pd.concat([test_benchmark,val_benchmark], ignore_index=True)

        return benchmark_df








    # =============================
    # Initialisations
    # =============================
    def _Initialise_CrossFoldScope(self):

        FetchTaskC = FetchTask(DIR_DerivedData = self.DIR_DerivedData,
                                    DIR_Typi = self.DIR_Typi,
                                    DIR_FuelInput = self.DIR_FuelInput,

                                    Df_grand = None, #self.Df_grand,
                                    TaskNameLabelLogicDict = None,
                                    n_row_maxhold = self.n_row_maxhold)
        self.FetchTaskC = FetchTaskC


        FetchDatasetC = FetchDataset(
                            DIR_DerivedData = self.DIR_DerivedData,
                            DIR_Typi = self.DIR_Typi,
                            DIR_FuelInput = self.DIR_FuelInput,
                            User_DesiredDatasize    = self.User_DesiredBatchDatasize, # NOTE This controls the number of new batch of dataset-dataloader being reloaded into memory
                            User_featuretype = self.User_featuretype,
                            n_datasetworker = self.n_datasetworker,
                            ClanGraphBcPercent = self.ClanGraphBcPercent)
        self.FetchDatasetC = FetchDatasetC
        # =========================
        # Get Definition of Tasks
        # =========================
        # NOTE This collects task name and how to get corresponding data in typi 
        TaskNameLabelLogicDict = FetchTaskC.Return_TaskNameLabelLogicDict()

        # =======================
        # Task Clan Fold Dataframe
        # =======================
        # NOTE This collects the assignment of pdbid to task, clan and fold. It skips when file already there.
        for bc in [90]:
            FetchTaskC.Make_TaskClanFoldDf(n_CrossFold = self.n_CrossFold, 
                                            ClanGraphBcPercent = bc, User_Task = self.User_Task)


        CrossFoldDfList = FetchTaskC.Return_CrossFoldDfList(n_CrossFold = self.n_CrossFold, 
                                                            ClanGraphBcPercent = self.ClanGraphBcPercent, 
                                                            User_Task = self.User_Task)

       
        # ==========================
        # Name of class to index in logit
        # ===============================

        classindex_str = sorted(TaskNameLabelLogicDict[self.User_Task].keys()) 
        ClassName_ClassIndex_Dict = dict(zip(classindex_str, range(len(classindex_str))))

        


        # ====================
        # Collect it into self!
        # =====================

        self.CrossFoldDfList = CrossFoldDfList
        
        self.TaskNameLabelLogicDict = TaskNameLabelLogicDict
        self.ClassName_ClassIndex_Dict = ClassName_ClassIndex_Dict
        self.n_class = len(ClassName_ClassIndex_Dict.keys())
        return 



# ==========================
# Loading models
# ==========================
class LoadCkpStateIntoModel:
    def __init__(self):
        pass
    def B1hw_LayerResnetBottleneck(self,
                        CKPT_PATH = "/home/homingla/Project-NucleicNet/Models/AUCG-B1hwLayerResnet5CV2018_AUCG-B1hwLayerResnet5CV2018/80_81/checkpoints/epoch=1-step=61025.ckpt",
                        device = torch.device('cuda'), 
                        ):
        checkpoint = torch.load(CKPT_PATH)#, map_location=device)
        model = NucleicNet.Burn.M1.B1hw_FcLogits(
                            model   = NucleicNet.Burn.M1.B1hw_LayerResnetBottleneck(n_FeatPerShell = 80, 
                                                        n_Shell = 6,
                                                        n_ShellMix = 2,
                                                        User_Activation = checkpoint['hyper_parameters']["User_Activation"],
                                                        User_Block = "B1hw_BlockPreActResnet",
                                                        n_Blocks = checkpoint['hyper_parameters']["User_n_ResnetBlock"],
                                                        ManualInitiation = False,
                                                        User_n_channelbottleneck = checkpoint['hyper_parameters']["User_n_channelbottleneck"],
                                                        User_NoiseZ = checkpoint['hyper_parameters']["User_NoiseZ"],
                                                        ),

                            loss    = nn.CrossEntropyLoss(label_smoothing=checkpoint['hyper_parameters']["User_LabelSmoothing"]),
                            n_class = 4,
                            hw_product = checkpoint['hyper_parameters']["hw_product"],
                            AddMultiLabelSoftMarginLoss = False, # TODO Worsen stuff? One-vs-all likely of no use.
                            User_lr = checkpoint['hyper_parameters']["User_lr"], #User_lr,
                            User_min_lr = checkpoint['hyper_parameters']["User_min_lr"], #User_min_lr,
                            User_LrScheduler = checkpoint['hyper_parameters']["User_LrScheduler"], #User_LrScheduler,
                            User_CooldownInterval = checkpoint['hyper_parameters']["User_CooldownInterval"], #User_CooldownInterval, 
                            BiasInSuffixFc = checkpoint['hyper_parameters']["BiasInSuffixFc"], 
                            # NOTE some kwargs for hparam record
                            User_SizeMinibatch = checkpoint['hyper_parameters']["User_SizeMinibatch"], #User_SizeMinibatch,
                            User_LabelSmoothing = checkpoint['hyper_parameters']["User_LabelSmoothing"], #User_LabelSmoothing,
                            User_PerformReduction = checkpoint['hyper_parameters']["User_PerformReduction"], #User_PerformReduction,
                            User_n_ResnetBlock = checkpoint['hyper_parameters']["User_n_ResnetBlock"], #User_n_ResnetBlock,
                            User_AdamW_weight_decay = checkpoint['hyper_parameters']["User_AdamW_weight_decay"], #User_AdamW_weight_decay,
                            User_Activation = checkpoint['hyper_parameters']["User_Activation"], #User_Activation,
                            User_SelectedCrossFoldIndex = checkpoint['hyper_parameters']["User_SelectedCrossFoldIndex"], #User_SelectedCrossFoldIndex,
                            User_Dropoutp = checkpoint['hyper_parameters']["User_Dropoutp"], #User_Dropoutp,
                            User_AddL1 = checkpoint['hyper_parameters']["User_AddL1"], #User_AddL1,
                            User_n_channelbottleneck = checkpoint['hyper_parameters']["User_n_channelbottleneck"], #User_n_channelbottleneck,
                            User_ShiftLrRatio = checkpoint['hyper_parameters']["User_ShiftLrRatio"], #User_ShiftLrRatio,
                            User_NoiseX = checkpoint['hyper_parameters']["User_NoiseX"], #User_NoiseX,
                            User_NoiseY = checkpoint['hyper_parameters']["User_NoiseY"],
                            User_Mixup = checkpoint['hyper_parameters']["User_Mixup"], #User_Mixup,
                            User_NumReductionComponent = checkpoint['hyper_parameters']["User_NumReductionComponent"],
                            User_NoiseZ = checkpoint['hyper_parameters']["User_NoiseZ"],
                            #User_PdbidIncluded = checkpoint['hyper_parameters']["User_PdbidExcluded"],
                            User_PdbidTraining = checkpoint['hyper_parameters']["User_PdbidTraining"],
                            User_PdbidValidation = checkpoint['hyper_parameters']["User_PdbidValidation"],
                            User_PdbidTesting = checkpoint['hyper_parameters']["User_PdbidTesting"],
                            User_InputDropoutp = checkpoint['hyper_parameters']["User_InputDropoutp"],
                            User_FocalLossAlpha = checkpoint['hyper_parameters']["User_FocalLossAlpha"],
                            User_FocalLossGamma = checkpoint['hyper_parameters']["User_FocalLossGamma"],
                            User_n_CrossFold = checkpoint['hyper_parameters']["User_n_CrossFold"],
                            User_ClanGraphBcPercent = checkpoint['hyper_parameters']["User_ClanGraphBcPercent"],
                            User_Task = checkpoint['hyper_parameters']["User_Task"],
                            User_NeighborLabelSmoothAngstrom = checkpoint['hyper_parameters']["User_NeighborLabelSmoothAngstrom"],
                        )


        model.load_state_dict(checkpoint['state_dict'])


        return model , checkpoint['hyper_parameters'], checkpoint


class LoadCkpStateIntoModelSXPR:
    def __init__(self):
        pass
    def B1hw_LayerResnetBottleneck(self,
                        CKPT_PATH = "/home/homingla/Project-NucleicNet/Models/AUCG-B1hwLayerResnet5CV2018_AUCG-B1hwLayerResnet5CV2018/80_81/checkpoints/epoch=1-step=61025.ckpt",
                        device = torch.device('cuda'), 
                        ):
        checkpoint = torch.load(CKPT_PATH)#, map_location=device)
        model = NucleicNet.Burn.M1.B1hw_FcLogits(
                            model   = NucleicNet.Burn.M1.B1hw_LayerResnetBottleneck(n_FeatPerShell = 80, 
                                                        n_Shell = 6,
                                                        n_ShellMix = 2,
                                                        User_Activation = checkpoint['hyper_parameters']["User_Activation"],
                                                        User_Block = "B1hw_BlockPreActResnet",
                                                        n_Blocks = checkpoint['hyper_parameters']["User_n_ResnetBlock"],
                                                        ManualInitiation = False,
                                                        User_n_channelbottleneck = checkpoint['hyper_parameters']["User_n_channelbottleneck"],
                                                        User_NoiseZ = checkpoint['hyper_parameters']["User_NoiseZ"],
                                                        ),

                            loss    = nn.CrossEntropyLoss(label_smoothing=checkpoint['hyper_parameters']["User_LabelSmoothing"]),
                            n_class = 4,
                            hw_product = checkpoint['hyper_parameters']["hw_product"],
                            AddMultiLabelSoftMarginLoss = False, # TODO Worsen stuff? One-vs-all likely of no use.
                            User_lr = checkpoint['hyper_parameters']["User_lr"], #User_lr,
                            User_min_lr = checkpoint['hyper_parameters']["User_min_lr"], #User_min_lr,
                            User_LrScheduler = checkpoint['hyper_parameters']["User_LrScheduler"], #User_LrScheduler,
                            User_CooldownInterval = checkpoint['hyper_parameters']["User_CooldownInterval"], #User_CooldownInterval, 
                            BiasInSuffixFc = checkpoint['hyper_parameters']["BiasInSuffixFc"], 
                            # NOTE some kwargs for hparam record
                            User_SizeMinibatch = checkpoint['hyper_parameters']["User_SizeMinibatch"], #User_SizeMinibatch,
                            User_LabelSmoothing = checkpoint['hyper_parameters']["User_LabelSmoothing"], #User_LabelSmoothing,
                            User_PerformReduction = checkpoint['hyper_parameters']["User_PerformReduction"], #User_PerformReduction,
                            User_n_ResnetBlock = checkpoint['hyper_parameters']["User_n_ResnetBlock"], #User_n_ResnetBlock,
                            User_AdamW_weight_decay = checkpoint['hyper_parameters']["User_AdamW_weight_decay"], #User_AdamW_weight_decay,
                            User_Activation = checkpoint['hyper_parameters']["User_Activation"], #User_Activation,
                            User_SelectedCrossFoldIndex = checkpoint['hyper_parameters']["User_SelectedCrossFoldIndex"], #User_SelectedCrossFoldIndex,
                            User_Dropoutp = checkpoint['hyper_parameters']["User_Dropoutp"], #User_Dropoutp,
                            User_AddL1 = checkpoint['hyper_parameters']["User_AddL1"], #User_AddL1,
                            User_n_channelbottleneck = checkpoint['hyper_parameters']["User_n_channelbottleneck"], #User_n_channelbottleneck,
                            User_ShiftLrRatio = checkpoint['hyper_parameters']["User_ShiftLrRatio"], #User_ShiftLrRatio,
                            User_NoiseX = checkpoint['hyper_parameters']["User_NoiseX"], #User_NoiseX,
                            User_NoiseY = checkpoint['hyper_parameters']["User_NoiseY"],
                            User_Mixup = checkpoint['hyper_parameters']["User_Mixup"], #User_Mixup,
                            User_NumReductionComponent = checkpoint['hyper_parameters']["User_NumReductionComponent"],
                            User_NoiseZ = checkpoint['hyper_parameters']["User_NoiseZ"],
                            #User_PdbidIncluded = checkpoint['hyper_parameters']["User_PdbidExcluded"],
                            User_PdbidTraining = checkpoint['hyper_parameters']["User_PdbidTraining"],
                            User_PdbidValidation = checkpoint['hyper_parameters']["User_PdbidValidation"],
                            User_PdbidTesting = checkpoint['hyper_parameters']["User_PdbidTesting"],
                            User_InputDropoutp = checkpoint['hyper_parameters']["User_InputDropoutp"],
                            User_FocalLossAlpha = checkpoint['hyper_parameters']["User_FocalLossAlpha"],
                            User_FocalLossGamma = checkpoint['hyper_parameters']["User_FocalLossGamma"],
                            User_n_CrossFold = checkpoint['hyper_parameters']["User_n_CrossFold"],
                            User_ClanGraphBcPercent = checkpoint['hyper_parameters']["User_ClanGraphBcPercent"],
                            User_Task = checkpoint['hyper_parameters']["User_Task"],
                            User_NeighborLabelSmoothAngstrom = checkpoint['hyper_parameters']["User_NeighborLabelSmoothAngstrom"],
                            #User_GradientClippingValue = checkpoint['hyper_parameters']["User_GradientClippingValue"],
                            User_datastride = checkpoint['hyper_parameters']["User_datastride"],
                        )


        model.load_state_dict(checkpoint['state_dict'])


        return model , checkpoint['hyper_parameters'], checkpoint

# ==============================
# Wrapper!
# ==============================
class BenchmarkWrapper:
    def __init__(self,
        DIR_Benchmark = '../Benchmark/',
        DIR_Models = "/home/homingla/Project-NucleicNet/Models/",
        OutputKeyword = "AUCG-B1hwLayerResnet9CV",
        DIR_DerivedData = '../Database-PDB/DerivedData/',
        DIR_Typi = '../Database-PDB/typi/',
        DIR_Feature = '../Database-PDB/feature/',
        n_CrossFold = 9, ClanGraphBcPercent = 90, # TODO Put into haram?
        User_Task = "AUCG", 
        Checkpointlist = [],
        Mmseq = False, # TODO ,
        User_SxprDatastride = 10, # NOTE Memory demanding when < 10
        ):


        self.DIR_Benchmark = DIR_Benchmark
        self.OutputKeyword = OutputKeyword
        self.DIR_Feature = DIR_Feature

        if type(Checkpointlist) is str:

            self.Checkpointlist = sorted(glob.glob(Checkpointlist))
        else:
            assert type(Checkpointlist) is list, "ABORTED. Provide a list of valid directory or use a glob string"
            self.Checkpointlist = sorted(Checkpointlist)

        self.DIR_Models = os.path.abspath(DIR_Models)



        # NOTE I use the OutputKeyword to organise the benchmark for the sake of readability.
        self.OutputLabel = []
        for ckppath in self.Checkpointlist:
            ckpinfo = ckppath.split(self.DIR_Models)[-1].split("/")
            self.OutputLabel.append(ckpinfo[2])
        self.OutputLabel  = "--".join(self.OutputLabel)

        self.n_CrossFold = n_CrossFold
        self.ClanGraphBcPercent = ClanGraphBcPercent
        self.User_Task = User_Task
        self.DIR_DerivedData = DIR_DerivedData
        self.DIR_Typi = DIR_Typi
        self.DIR_SaveBenchmark = DIR_Benchmark + OutputKeyword + "/" + self.OutputLabel
        self.DIR_Benchmark = DIR_Benchmark
        MkdirList([DIR_Benchmark, DIR_Benchmark + OutputKeyword, self.DIR_SaveBenchmark])

        self.User_SxprDatastride = User_SxprDatastride

    def Run(self):

        # ====================
        # Assign attribute
        # ======================
        



        Checkpointlist = self.Checkpointlist

        grandbenchmark_df = []
        for i in range(len(Checkpointlist)):

            if self.User_Task == "AUCG": # NOTE we have two extra flag in the hyperparameter file of sxpr... TODO Either move these away or make the AUCG also that those flags
                LoadCkpStateIntoModelC = LoadCkpStateIntoModel()
                model, ckp_hyperparam, checkpoint = LoadCkpStateIntoModelC.B1hw_LayerResnetBottleneck(CKPT_PATH =Checkpointlist[i])
            else:
                LoadCkpStateIntoModelC = LoadCkpStateIntoModelSXPR()
                model, ckp_hyperparam, checkpoint = LoadCkpStateIntoModelC.B1hw_LayerResnetBottleneck(CKPT_PATH =Checkpointlist[i])
            #print("=========== Training =================")
            #print(ckp_hyperparam["User_PdbidTraining"])
            #print("=========== Validation ===============")
            #print(ckp_hyperparam["User_PdbidValidation"])
            #print("=========== Testing ==================")
            #print(ckp_hyperparam["User_PdbidTesting"])
            #print("======================================")
            # NOTE We will take the last ckpt for this
            self.ClanGraphBcPercent  = ckp_hyperparam["User_ClanGraphBcPercent"]
            self.n_CrossFold = ckp_hyperparam["User_n_CrossFold"]
            self.User_Task = ckp_hyperparam["User_Task"]


            n_CrossFold = self.n_CrossFold 
            ClanGraphBcPercent = self.ClanGraphBcPercent 
            with open(self.DIR_DerivedData + "/TaskClanFoldDf_Task%s_Mmseq%s_Fold%s.pkl" %(self.User_Task, ClanGraphBcPercent,n_CrossFold ), "rb") as fn:
                TaskClanFoldDf = pickle.load(fn)


            BenchmarkHelperC = BenchmarkHelper( model = model,
                                                DIR_DerivedData = self.DIR_DerivedData,
                                                DIR_Typi = self.DIR_Typi,
                                                DIR_FuelInput = self.DIR_Feature,
                                                DIR_Benchmark = self.DIR_Benchmark,
                                                n_CrossFold = self.n_CrossFold,
                                                ClanGraphBcPercent = self.ClanGraphBcPercent,
                                                User_featuretype =  'altman', #ckp_hyperparam["User_featuretype"],
                                                User_DesiredBatchDatasize    = 7000000,
                                                User_Task = self.User_Task,
                                                n_row_maxhold = 10000,
                                                #Df_grand = None,
                                                n_datasetworker = 16,
                                                device_ = torch.device('cuda'),
                                                User_SxprDatastride = self.User_SxprDatastride)

            benchmark_df = BenchmarkHelperC.Testing(ckp_hyperparam)
            benchmark_df.loc[:,'Checkpoint'] = Checkpointlist[i]

            #User_Task = ckp_hyperparam['User_Task']

            TaskClanFoldDf_ = TaskClanFoldDf.loc[TaskClanFoldDf["Task"] == self.User_Task]
            PdbidToClan = {}
            PdbidToFold = {}
            for row_i, row in tqdm.tqdm(TaskClanFoldDf_.iterrows()):
                for i in row['PdbidList']:
                    PdbidToFold[i] = row['CrossFold']
                    PdbidToClan[i] = row['Clan']

            benchmark_df.loc[:,"Clan"] = benchmark_df['Pdbid'].map(PdbidToClan)
            benchmark_df.loc[:,"CrossFold"] = benchmark_df['Pdbid'].map(PdbidToFold)
            benchmark_df.loc[:,"Pdbid "] = benchmark_df['Pdbid'].str[:4]
            benchmark_df.loc[:,"Multistate"] = benchmark_df['Pdbid'].str[4:]


            # Hyperparam
            for k in ckp_hyperparam.keys():
                if k in ["User_PdbidTraining", "User_PdbidValidation", "User_PdbidTesting"]:
                    continue
                try:
                    benchmark_df.loc[:, "%s" %(k)] = float(ckp_hyperparam[k])
                except:
                    benchmark_df.loc[:, "%s" %(k)] = str(ckp_hyperparam[k])

            grandbenchmark_df.append(benchmark_df)

            
            del model, ckp_hyperparam
            gc.collect()

        grandbenchmark_df = pd.concat(grandbenchmark_df, ignore_index= True)







        # TODO Select Best of Multistate
        grandbenchmark_df.to_pickle("%s/DfBenchmark__%s__%s.pkl" %(self.DIR_SaveBenchmark, self.OutputKeyword, self.OutputLabel))

        self.grandbenchmark_df = grandbenchmark_df


        return self.grandbenchmark_df

    def Plot(self):

        # TODO Organise the clans produce an average with a larger dot
        # NOTE THese are highlights printout for weakest 
        #Weak_Training = self.grandbenchmark_df.loc[(self.grandbenchmark_df['Status'] == 'TrainingValidation') &(self.grandbenchmark_df['Accuracy-Top1'] < 0.5)][['Pdbid','Accuracy-Top1' ]]
        #Weak_Testing = grandbenchmark_df.loc[(grandbenchmark_df['Status'] == 'Testing') &(grandbenchmark_df['Accuracy-Top1'] < 0.5)]#[['Pdbid','Accuracy-Top1' ]]
        #grandbenchmark_df


        # ===============================
        # Plot all
        # ===============================
        plt.figure(figsize = (15,12))
        g = sns.histplot(data=self.grandbenchmark_df, x="Precision-Top1", y="Recall-Top1", kde=False, bins = 30, palette='YlGnBu', thresh = None, cbar = True)
        sns.scatterplot(data=self.grandbenchmark_df, x="Precision-Top1", y="Recall-Top1", 
                        marker = 'o', s = 8.0, hue = 'Status', palette='husl', alpha = 0.8)

        plt.title("Recall/Precision Distribution")
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.show()
        #print(Weak_Testing)
