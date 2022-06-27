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

from scipy import spatial, sparse
from biopandas.pdb import PandasPdb
from biopandas.pdb import engines as PandasPdbEngines
from sklearn.cluster import DBSCAN
import pandas as pd
import itertools
import os
from collections import Counter
from warnings import warn
import shutil
import tqdm
import gc
import sys
import time

# ================
# Torch related
# ==============
import torch 

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






sys.path.append('../')
from NucleicNet.DatasetBuilding.commandHalo import Halo
from NucleicNet.DatasetBuilding.commandTypify import OOC_RetrieveHaloCoords
from NucleicNet.DatasetBuilding.commandFeaturisation import Featurisation
from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding import commandBenchmark
from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandReadPdbFtp import ReadBCExternalSymmetry, MakeBcClanGraph
from NucleicNet.DatasetBuilding.commandDataFetcher import FetchIndex, FetchTask, FetchDataset
from NucleicNet.DatasetBuilding.commandBenchmark import BenchmarkWrapper
from NucleicNet import Burn, Fuel
import NucleicNet.Burn.M1
import NucleicNet.Burn.util



# ==========================
# Model loading
# ==========================

class PredictionHelper:
    def __init__(self,
        model = None,
        DIR_DerivedData = "../Database-PDB/DerivedData/",
        DIR_Typi = "../Database-PDB/typi/", # NOTE Inherit from wrapper
        DIR_FuelInput = "../Database-PDB/feature/", # NOTE Inherit from wrapper
        DIR_HaloFolder = "../Database-PDB/halo/", # NOTE Inherit from wrapper
        DIR_Benchmark = '../Benchmark/',
        n_CrossFold = 9,
        ClanGraphBcPercent = 90,
        User_featuretype = 'altman',
        User_DesiredBatchDatasize    = 7000000,
        User_Task = "AUCG",
        n_row_maxhold = 10000,
        #Df_grand = None,
        n_datasetworker = 16,
        assignedpdbid = '4tst', 
        aux_id = 0,
        device_ = torch.device('cuda'),
        User_NeighborLabelSmoothAngstrom = 1.5
        
    ):
        #MkdirList([DIR_Benchmark])
        print("helper receive", DIR_Typi)
        # NOTE Get Model
        assert model is not None, "ABORTED. Supply a model"
        self.model = model
        self.model.to(device_)
        self.model.eval()
        
        self.DIR_DerivedData = DIR_DerivedData 
        self.DIR_Typi = DIR_Typi 
        self.DIR_FuelInput = DIR_FuelInput 
        self.DIR_HaloFolder = DIR_HaloFolder

        #self.n_CrossFold = n_CrossFold 
        self.ClanGraphBcPercent = ClanGraphBcPercent 
        self.User_featuretype = User_featuretype 
        self.User_DesiredBatchDatasize    = User_DesiredBatchDatasize  
        self.User_Task = User_Task 
        self.n_row_maxhold = n_row_maxhold        
        self.User_NeighborLabelSmoothAngstrom = User_NeighborLabelSmoothAngstrom
        #self.User_TrainValSplit = 0.5
        self.n_datasetworker = n_datasetworker

        self.assignedpdbid = assignedpdbid 
        self.aux_id = aux_id

        # ==================
        # Initialise
        # ==================
        self._Initialise_CrossFoldScope()







    # NOTE This is a subroutine...
    def Diagnose(self, Testing_PdbidBatches, ckp_hyperparam):
        """
        This snippet will run a given model defined at initiation on a set of Testing_PdbidBatches 
        """
        benchmark_df = []
        with torch.no_grad():
            Testing_PdbidBatches_ = Testing_PdbidBatches


            pdbid_done = []
            for retrial_i in range(10):
                for pdbid in tqdm.tqdm(Testing_PdbidBatches_):
                    print(self.ClassName_ClassIndex_Dict, pdbid, self.User_Task)
                    if pdbid in pdbid_done:
                        continue
                    try:
                        PART0_LoadDataset = True
                        if PART0_LoadDataset:

                            ds_ = self.FetchDatasetC.GetNoOpSingleDataset(
                                                    Assigned_PdbidBatch = [pdbid],
                                                    ClassName_ClassIndex_Dict = self.ClassName_ClassIndex_Dict,
                                                    Assigned_PdbidWeight = None,
                                                    User_Task = self.User_Task,
                                                    PerformZscoring = True, PerformReduction = ckp_hyperparam['User_PerformReduction'],
                                                    TestingPhase = True,
                                                    User_datastride = 1
                                                    )

                            test_loader_t          = torch.utils.data.DataLoader(ds_, 
                                                                                batch_size=1024, 
                                                                                drop_last=False, num_workers=4, 
                                                                                pin_memory=True,worker_init_fn=None, 
                                                                                prefetch_factor=3, persistent_workers=False,
                                                                                shuffle=False)

                        
                        PART1_EnumerateInBatch = True
                        if PART1_EnumerateInBatch:

                            p_list = []                
                            for _, xy in enumerate(test_loader_t):
                                # TODO Get the index in halo and the xyz coordinate of halo here.

                                x = xy[0] 
                                y = xy[1]
                                x = x.to('cuda')
                                y = y.to('cuda')
                                p = self.model.predict_step((x,y),0)

                                p_list.append(p)

                                del x,y
                                gc.collect()


                            p = torch.cat(p_list)
                            p = p.detach().cpu().numpy()

                            del test_loader_t, ds_
                            NucleicNet.Burn.util.TorchEmptyCache()
                            gc.collect()               
                        

                        PART2_GetHaloXyz = True
                        if PART2_GetHaloXyz:
                            with open(self.DIR_HaloFolder+"/%s.halotup"%(pdbid),'rb') as fn:
                                halo_tuple = pickle.load(fn)
                            halo_coord = OOC_RetrieveHaloCoords(halo_tuple)
                            halo_tree = spatial.cKDTree(halo_coord)

                            
                            # NOTE P from Top1 prediction
                            dominantlab = np.argmax(p, axis=1)
                            dominantp = np.zeros_like(p) + 0.001
                            dominantp[np.arange(dominantlab.size), dominantlab] = 1
                            dominantp /= np.sum(dominantp,axis =1)[...,np.newaxis]



                            # NOTE P from Top2 prediction. 
                            #      Ideally, a polynomial regression can be done when sequencing data is available e.g. Renormalise(a* p^2 + b* p^1 + c)
                            #      But currently we dont have it juxtaposed with the structural data (yet).
                            #      So, we simply do a quadratic scoring here renormalise(p^2)

                            dominantlab2 = np.argpartition(p, 2, axis =1)[:2]
                            dominantp2 = p 
                            dominantp2[np.arange(dominantlab2.shape[0]), dominantlab2[:,0].T] = 0.0
                            dominantp2[np.arange(dominantlab2.shape[0]), dominantlab2[:,1].T] = 0.0
                            dominantp2 += 0.001
                            dominantp2 = np.power(dominantp2,2)
                            dominantp2 /= np.sum(dominantp2,axis =1)[...,np.newaxis]

                            # NOTE Neighborhood smoothen. This is recommended.
                            if self.User_NeighborLabelSmoothAngstrom > 0.0:

                                neighbor_selectedindex_lol = halo_tree.query_ball_point(halo_coord, self.User_NeighborLabelSmoothAngstrom)
                                mean_p = np.zeros_like(p)

                                mean_dominantpTop1 = np.zeros_like(dominantp)
                                mean_dominantpTop2 = np.zeros_like(dominantp)
                                for i_row in range(halo_coord.shape[0]):
                                    #n_neighbour = len(neighbor_selectedindex_lol[i_row])
                                    mean_p[i_row] += np.mean(p[neighbor_selectedindex_lol[i_row], :], axis=0)
                                    mean_dominantpTop1[i_row] += np.mean(dominantp[neighbor_selectedindex_lol[i_row], :], axis=0)
                                    mean_dominantpTop2[i_row] += np.mean(dominantp2[neighbor_selectedindex_lol[i_row], :], axis=0)

                            else:
                                mean_p = p
                                mean_dominantpTop1 = dominantp
                                mean_dominantpTop2 = dominantp2
                            



                            # NOTE Calculate sum PlogP
                            n_class = p.shape[1]
                            info_content = np.log2(n_class) #- H - smallsamplecorrection
                            smallsamplecorrection = 1 / np.log2(2) * (n_class -1 ) / (2 * p.shape[0]) # NOTE very small in general
                            # NOTE negative numbers
                            plogp_raw = (np.sum(np.multiply(p,np.log2(p)), axis =1)[...,np.newaxis])
                            plogp_smoothened = (np.sum(np.multiply(mean_p,np.log2(mean_p)), axis = 1)[...,np.newaxis])
                            plogp_dominantsmoothenedTop1 = (np.sum(np.multiply(mean_dominantpTop1,np.log2(mean_dominantpTop1)), axis = 1)[...,np.newaxis])
                            plogp_dominantsmoothenedTop2 = (np.sum(np.multiply(mean_dominantpTop2,np.log2(mean_dominantpTop2)), axis = 1)[...,np.newaxis])

                            # NOTE for simplicity in presentation, the max height is set at info_content - smallsamplecorrection, so that it is always betwoeen 1 and 0
                            InfoHeight_raw = (info_content - smallsamplecorrection + plogp_raw) / (info_content - smallsamplecorrection)
                            InfoHeight_smoothened = (info_content - smallsamplecorrection + plogp_smoothened)  / (info_content - smallsamplecorrection)
                            InfoHeight_dominantsmoothenedTop1 = (info_content - smallsamplecorrection + plogp_dominantsmoothenedTop1)  / (info_content - smallsamplecorrection)
                            InfoHeight_dominantsmoothenedTop2 = (info_content - smallsamplecorrection + plogp_dominantsmoothenedTop2)  / (info_content - smallsamplecorrection)
                            haloindex = np.array(list(range(len(halo_coord)))).astype(int)[...,np.newaxis]

                            columnnames = ['HaloIdx', 'x', 'y', 'z']
                            columnnames.extend(['Raw_%s'%(iiii) for iiii in list(range(p.shape[1]))])
                            columnnames.extend(['Smoothened_%s'%(iiii) for iiii in list(range(p.shape[1]))])
                            columnnames.extend(['SmoothenedDominantTop1_%s'%(iiii) for iiii in list(range(p.shape[1]))])
                            columnnames.extend(['SmoothenedDominantTop2_%s'%(iiii) for iiii in list(range(p.shape[1]))])
                            #columnnames.extend(['Entropy_RawP', "Entropy_SmoothenedP", "Entropy_SmoothenedDominantP"])
                            columnnames.extend(['Bit Fraction (Raw)', 
                                                "Bit Fraction (Smoothened)", 
                                                "Bit Fraction (Dominant Top1, Smoothened)", 
                                                "Bit Fraction (Dominant Top2, Smoothened)"])





                            tempdf = pd.DataFrame(np.concatenate([haloindex,halo_coord, 
                                                                p, mean_p, mean_dominantpTop1,mean_dominantpTop2,
                                                                #plogp_raw,plogp_smoothened,plogp_dominantsmoothened 
                                                                InfoHeight_raw, InfoHeight_smoothened, InfoHeight_dominantsmoothenedTop1, InfoHeight_dominantsmoothenedTop2,
                                                                ], axis= 1), columns=columnnames)
                            tempdf.loc[:,"Pdbid"] = pdbid
                            benchmark_df.append(tempdf)
                            pdbid_done.append(pdbid)

                    except RuntimeError:
                        NucleicNet.Burn.util.TorchEmptyCache()
                        time.sleep(2)
                        print("Cuda run out of memory at %s retrial %s or run it later!" %pdbid, retrial_i)
                        NucleicNet.Burn.util.TorchEmptyCache()
                        time.sleep(2)
                        #NucleicNet.Burn.util.TorchEmptyCache()



        if len(benchmark_df) == 1:
            benchmark_df = benchmark_df[0]
        else:
            benchmark_df = pd.concat(benchmark_df)
        #print(benchmark_df)
        return benchmark_df

    # NOTE This is a wrapper for diagnose
    def Testing(self, ckp_hyperparam, 
                ):

        Testing_PdbidBatches = ['%s%08d'%(self.assignedpdbid,self.aux_id )]

        test_benchmark = self.Diagnose(Testing_PdbidBatches, ckp_hyperparam)
        test_benchmark.loc[:,"Status"] = "Prediction"
        NucleicNet.Burn.util.TorchEmptyCache()



        return test_benchmark








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
        # =========================
        # Get Definition of Tasks
        # =========================
        # NOTE This collects task name and how to get corresponding data in typi 
        TaskNameLabelLogicDict = FetchTaskC.Return_TaskNameLabelLogicDict()
        # ==========================
        # Name of class to index in logit
        # ===============================

        classindex_str = sorted(TaskNameLabelLogicDict[self.User_Task].keys()) 
        ClassName_ClassIndex_Dict = dict(zip(classindex_str, range(len(classindex_str))))

        self.TaskNameLabelLogicDict = TaskNameLabelLogicDict
        self.ClassName_ClassIndex_Dict = ClassName_ClassIndex_Dict
        self.n_class = len(ClassName_ClassIndex_Dict.keys())
        #print(ClassName_ClassIndex_Dict, TaskNameLabelLogicDict)


        # ========================
        # Dataset fetcher
        # ========================
        ##print(self.DIR_Typi, 'fetcher receive')
        FetchDatasetC = FetchDataset(
                            DIR_DerivedData = self.DIR_DerivedData,
                            DIR_Typi = self.DIR_Typi,
                            DIR_FuelInput = self.DIR_FuelInput,
                            User_DesiredDatasize    = self.User_DesiredBatchDatasize, # NOTE This controls the number of new batch of dataset-dataloader being reloaded into memory
                            User_featuretype = self.User_featuretype,
                            n_datasetworker = self.n_datasetworker,
                            ClanGraphBcPercent = self.ClanGraphBcPercent)
        self.FetchDatasetC = FetchDatasetC


        # =======================
        # Task Clan Fold Dataframe
        # =======================
        


        


        # ====================
        # Collect it into self!
        # =====================

        #self.CrossFoldDfList = CrossFoldDfList
        

        return 









class PredictionWrapper:
    def __init__(self,
        DIR_Benchmark = '/home/homingla/Downloads/NucleicNetServer/',
        DIR_Models = "/home/homingla/Project-NucleicNet/Models/",
        DIR_DerivedData = '../Database-PDB/DerivedData/',
        User_Task = "AUCG", 
        Checkpointlist = [],
        User_NeighborLabelSmoothAngstrom = 1.5,
        User_ModelAverage = True):

        n_CrossFold = 9
        ClanGraphBcPercent = 90
        self.DIR_Benchmark = DIR_Benchmark
        DIR_Typi = DIR_Benchmark + '/typi/'
        DIR_Feature = DIR_Benchmark + '/feature/'
        DIR_HaloFolder = DIR_Benchmark + '/halo/'

        self.DIR_Benchmark = DIR_Benchmark
        #self.OutputKeyword = User_Task
        self.DIR_Feature = DIR_Feature
        self.DIR_HaloFolder = DIR_HaloFolder
        self.User_Task = User_Task
        self.DIR_DerivedData = DIR_DerivedData
        self.DIR_Typi = DIR_Typi
        self.User_NeighborLabelSmoothAngstrom = User_NeighborLabelSmoothAngstrom
        self.User_ModelAverage = User_ModelAverage

        if type(Checkpointlist) is str:
            self.Checkpointlist = sorted(glob.glob(Checkpointlist))
        else:
            assert type(Checkpointlist) is list, "ABORTED. Provide a list of valid directory or use a glob string"
            self.Checkpointlist = sorted(Checkpointlist)

        self.DIR_Models = os.path.abspath(DIR_Models)


        self.assignedpdbid = "4tst"

        # NOTE I use the OutputKeyword to organise the benchmark for the sake of readability.
        self.OutputLabel = []
        for ckppath in self.Checkpointlist:
            ckpinfo = ckppath.split(self.DIR_Models)[-1].split("/")
            self.OutputLabel.append(ckpinfo[2])
        self.OutputLabel  = "--".join(self.OutputLabel)

        self.n_CrossFold = n_CrossFold
        self.ClanGraphBcPercent = ClanGraphBcPercent


        

        MkdirList([DIR_Benchmark, DIR_Benchmark + User_Task]) # NOTE we will make a folder for each task


    def Run(self, aux_id = 0):

        # ====================
        # Assign attribute
        # ======================
        
        # TODO Get testing
        #DIR_Feature

        Checkpointlist = self.Checkpointlist

        result_df = []
        for i in range(len(Checkpointlist)):
            print("Checkpoint %s" %(i))
            LoadCkpStateIntoModelC = commandBenchmark.LoadCkpStateIntoModel()
            model, ckp_hyperparam, checkpoint = LoadCkpStateIntoModelC.B1hw_LayerResnetBottleneck(CKPT_PATH =Checkpointlist[i])



            PredictionHelperC = PredictionHelper( model = model,
                                                DIR_DerivedData = self.DIR_DerivedData,
                                                DIR_Typi = self.DIR_Typi,
                                                DIR_FuelInput = self.DIR_Feature,
                                                DIR_Benchmark = self.DIR_Benchmark,
                                                DIR_HaloFolder = self.DIR_HaloFolder,
                                                n_CrossFold = self.n_CrossFold,
                                                ClanGraphBcPercent = self.ClanGraphBcPercent,
                                                User_featuretype =  'altman', #ckp_hyperparam["User_featuretype"],
                                                User_DesiredBatchDatasize    = 7000000,
                                                User_Task = self.User_Task,
                                                n_row_maxhold = 10000,
                                                #Df_grand = None,
                                                n_datasetworker = 16,
                                                device_ = torch.device('cuda'),
                                                assignedpdbid = self.assignedpdbid, 
                                                User_NeighborLabelSmoothAngstrom = self.User_NeighborLabelSmoothAngstrom,
                                                aux_id = aux_id)

            benchmark_df = PredictionHelperC.Testing(ckp_hyperparam)
            benchmark_df.loc[:,'Checkpoint'] = Checkpointlist[i]
            result_df.append(benchmark_df)

            self.ClassName_ClassIndex_Dict = PredictionHelperC.ClassName_ClassIndex_Dict
            del PredictionHelperC ,LoadCkpStateIntoModelC, model, ckp_hyperparam, checkpoint
            gc.collect()

        # NOTE This is before model averaging
        result_df = pd.concat(result_df, ignore_index= True)

        # NOTE Model averaged
        result_df_modelaveraged = result_df.groupby(['Pdbid', 'HaloIdx'], as_index=False).mean().reset_index()

        return result_df, result_df_modelaveraged, self.ClassName_ClassIndex_Dict




# =======================
# Server Functions
# =======================


class Server:
    def __init__(self, SaveCleansed = True, SaveDf = True,
                Select_HeavyAtoms = True,
                Select_Resname = [
                    "ALA","CYS","ASP","GLU","PHE","GLY", 
                    "HIS","ILE","LYS","LEU","MET","ASN", 
                    "PRO","GLN","ARG","SER","THR","VAL", 
                    "TRP","TYR"
                    ],
                DIR_ServerFolder = "",
                DsspExecutable = "../NucleicNet/util/dssp",
                DIR_DerivedDataFolder = "../Database-PDB/DerivedData",
                DIR_ClassIndexCopy = "../Database-PDB/DerivedData/ClassIndex.pkl"
                ):


        self.assignedpdbid = "4tst" # NOTE Feature require name of file to begin with a numeric. 4tst is a pdbid that will never get assigned https://proteopedia.org/wiki/index.php/4tst
        self.SaveCleansed = SaveCleansed
        self.SaveDf = SaveDf
        self.Select_HeavyAtoms = Select_HeavyAtoms
        self.Select_Resname = Select_Resname
        self.DsspExecutable = DsspExecutable
        self.DIR_DerivedDataFolder = DIR_DerivedDataFolder

        # NOTE Folders
        self.DIR_ServerFolder = DIR_ServerFolder
        self.DIR_SaveCleansedFolder = DIR_ServerFolder + '/apo/'
        self.DIR_HaloFolder = DIR_ServerFolder + '/halo/'
        self.DIR_TypiFolder = DIR_ServerFolder + '/typi/'
        self.DIR_FeatureFolder = DIR_ServerFolder + '/feature/'
        self.DIR_DsspFolder = DIR_ServerFolder + '/dssp/'
        self.DIR_SxprFolder = DIR_ServerFolder + '/sxpr/'
        self.DIR_AucgFolder = DIR_ServerFolder + '/aucg/'
        MkdirList([ self.DIR_ServerFolder, 
                    self.DIR_SaveCleansedFolder , 
                    self.DIR_HaloFolder, self.DIR_TypiFolder, self.DIR_FeatureFolder , self.DIR_DsspFolder, 
                    self.DIR_AucgFolder, self.DIR_SxprFolder])
        shutil.copyfile(DIR_ClassIndexCopy, self.DIR_TypiFolder + "ClassIndex.pkl")
        
        with open(self.DIR_DerivedDataFolder + '/ClassIndex.pkl', 'rb') as fn:
            self.tc_mapping = pickle.load(fn)
        print( self.tc_mapping)




    # ===============================
    # User file processing
    # ================================
    def SimpleSanitise( self,
                aux_id = 0,
                DIR_InputPdbFile = "uploaded/uploaded.pdb",
                ):

        pdbid = self.assignedpdbid



        ppdb = PandasPdb()
        ppdb.read_pdb(DIR_InputPdbFile)
        
        assert ppdb.df['ATOM'].shape[0] != 0, "ABORTED. User submitted a empty pdb file"


        PART1_Filtering = True
        if PART1_Filtering:

            # NOTE Remove Hydrogens
            if self.Select_HeavyAtoms:
                tempdf = ppdb.df['ATOM']
                ppdb.df['ATOM'] = tempdf.loc[tempdf['element_symbol'] != 'H']

            # NOTE Select only certain residues
            if len(self.Select_Resname) > 0 :
                tempdf = ppdb.df['ATOM']
                ppdb.df['ATOM'] = tempdf.loc[tempdf["residue_name"].isin(self.Select_Resname)]


            # NOTE For simplicity we only allow alternative location A or the default. Otherwise we will need to produce multiple files complicating everything like the training stage
            tempdf = ppdb.df['ATOM']
            ppdb.df['ATOM'] = tempdf.loc[tempdf["alt_loc"].isin(['','A'])]

        PART2_SaveSeparate = True
        if not  ppdb.df['ATOM'].empty:
            # NOTE we will only save the atom record for simplicity.

            if self.SaveDf:
                ppdb.df['ATOM'].to_pickle('%s/%s.pkl'%(self.DIR_SaveCleansedFolder, pdbid), 
                                                        compression='infer', protocol=4, storage_options=None)
            if self.SaveCleansed:
                #ppdb.to_pdb("%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id), 
                #                                records=['ATOM'], gz=False, append_newline=True)
                self.UTIL_PdbDfDictToStr(ppdb.df, RemoveLinesStartingWith = [], 
                                        DIR_Save = "%s/%s%08d.pdb"% (self.DIR_SaveCleansedFolder,pdbid,aux_id),
                                        records = ["ATOM"])


            pass

        del ppdb
        gc.collect()
        return

    def MakeHalo(self, aux_id = 0,
        ):
        pdbid = self.assignedpdbid

        HaloC = Halo(HaloLowerBound = 2.5, HaloUpperBound = 5.0, LatticeSpacing = 1.0, 
             DIR_OutputHaloTupleFolder = self.DIR_HaloFolder,
             DIR_InputPdbFolder = self.DIR_SaveCleansedFolder, 
             DIR_OutputTypifyFolder = self.DIR_TypiFolder, # NOTE Just for completeness
             n_MultiprocessingWorkers = 10,
             CallMkdirList = False,
             ) 
        HaloC.Voxelise(PlottingHalo = False, WritingXyz = True,
                        DIR_JustOnePdb = "%s/%s%08d.pdb"% (self.DIR_SaveCleansedFolder,pdbid, aux_id), SavingHalo = True, UpdateExist = False)
        #
        return


    def MakeDssp(self, aux_id = 0):
        
        f = self.DIR_SaveCleansedFolder + "%s%08d.pdb" %(self.assignedpdbid, aux_id)
        subprocess.call("%s -i %s -o %s/%s%08d.dssp" %(
        self.DsspExecutable,
        f,
        self.DIR_SaveCleansedFolder,
        self.assignedpdbid,
        aux_id
        #f.split("/")[-1].split(".")[0]
        ), shell = True)
  
        return
    def MakeFeature(self):

        FeaturisationC = Featurisation(        
                DIR_OutputFeatureVector = self.DIR_FeatureFolder,
                DIR_InputPdbFolder = self.DIR_SaveCleansedFolder, 
                DIR_InputHaloFolder = self.DIR_HaloFolder,
                DIR_DsspExecutable = "../NucleicNet/util/dssp",
                DIR_AltmanFolder = "../NucleicNet/util/feature-3.1.0/",
                n_multiprocessingworkers= 5,
                UpdateExist = True,SavingFeature = True)
        FeaturisationC.Altman()

        return
        

    def MakeDummyTypi(self, aux_id = 0):


        pdbid = "%s%08d"% (self.assignedpdbid, aux_id)


        with open(self.DIR_HaloFolder+"/%s.halotup"%(pdbid),'rb') as fn:
            halo_tuple = pickle.load(fn)
        halo_coord = OOC_RetrieveHaloCoords(halo_tuple)
        halo_tree = spatial.cKDTree(halo_coord)

        # NOTE This has shape (n_halo, n_possible_class)
        SparseClassMatrix = sparse.csr_matrix((halo_coord.shape[0], 
                                    len(self.tc_mapping.keys()))).tolil()

        # Assign random classes as a dummy to build dataset
        for i in range(halo_coord.shape[0]):
            randkey = np.abs(np.random.randn(len(self.tc_mapping.keys())))
            j = 0
            for k in self.tc_mapping.keys():
                if randkey[j] < 0.5: # NOTE abs above, so very likely
                    SparseClassMatrix[i,self.tc_mapping[k]] += 1.0
                j+=1


        SparseClassMatrix = SparseClassMatrix.tocsr(copy=False)

        sparse.save_npz(self.DIR_TypiFolder + '/%s.typi.npz' %(pdbid),
                                                SparseClassMatrix,compressed = True)




    def MakeSXPR(self,  Checkpointlist = ["../Models/SXPR-9CV_SXPR-9CV/28_29/checkpoints/epoch=7-step=27114-hp_metric=0.5113999843597412.ckpt",
                                          "../Models/SXPR-9CV_SXPR-9CV/30_31/checkpoints/epoch=7-step=25856-hp_metric=0.5144857168197632.ckpt",
                                          "../Models/SXPR-9CV_SXPR-9CV/32_33/checkpoints/epoch=7-step=23986-hp_metric=0.5110856890678406.ckpt",],
                        aux_id = 0,
                        ):



        PredictionWrapperC = PredictionWrapper(
                DIR_Benchmark = self.DIR_ServerFolder + '/',
                DIR_Models = "../Models/",
                DIR_DerivedData = "../Database-PDB/DerivedData/",
                User_Task = "SXPR", 
                Checkpointlist = Checkpointlist)

        result_df, result_df_modelaveraged, ClassName_ClassIndex_Dict = PredictionWrapperC.Run(aux_id = aux_id)
        result_df.to_pickle("%s/Result_IndividualDf.pkl" %(self.DIR_SxprFolder), protocol = 4 )
        result_df_modelaveraged.to_pickle("%s/Result_EnsembleAvDf.pkl"  %(self.DIR_SxprFolder), protocol = 4 )
        with open("%s/ClassName_ClassIndex_Dict.pkl"  %(self.DIR_SxprFolder), 'wb' )as fn:
            pickle.dump(ClassName_ClassIndex_Dict, fn, protocol = 4)

        return 

    def MakeAUCG(self,  Checkpointlist = [  "../Models/AUCG-9CV_AUCG-9CV/294_295/checkpoints/epoch=7-step=26400-hp_metric=0.485228568315506.ckpt",
                                            "../Models/AUCG-9CV_AUCG-9CV/296_297/checkpoints/epoch=7-step=25924-hp_metric=0.48591428995132446.ckpt",
                                            "../Models/AUCG-9CV_AUCG-9CV/298_299/checkpoints/epoch=7-step=24326-hp_metric=0.5161714553833008.ckpt",],
                        aux_id = 0,
                        ):


        PredictionWrapperC = PredictionWrapper(
                DIR_Benchmark = self.DIR_ServerFolder + '/',
                DIR_Models = "../Models/",
                DIR_DerivedData = "../Database-PDB/DerivedData/",
                User_Task = "AUCG", 
                Checkpointlist = Checkpointlist)

        result_df, result_df_modelaveraged, ClassName_ClassIndex_Dict = PredictionWrapperC.Run(aux_id = aux_id)
        result_df.to_pickle("%s/Result_IndividualDf.pkl" %(self.DIR_AucgFolder), protocol = 4 )
        result_df_modelaveraged.to_pickle("%s/Result_EnsembleAvDf.pkl"  %(self.DIR_AucgFolder), protocol = 4 )
        with open("%s/ClassName_ClassIndex_Dict.pkl"  %(self.DIR_AucgFolder), 'wb' )as fn:
            pickle.dump(ClassName_ClassIndex_Dict, fn, protocol = 4)

        return 


    # =========================
    # Downstream Usage
    # =========================

    def Downstream_VisualisePse(self,              
                    AtomNameDict = {  "P":"P", 
                                    "R": "Re",
                                    "G": "Ge","U":"U", "A":"Ar","C":"Co", 
                                    "Pyr":"Y", "Pur":"Pu"}, 
                    User_Sxpr_VisualiseThreshold_Base = 0.6,
                    User_Sxpr_VisualiseThreshold_P = 0.9,
                    User_Sxpr_VisualiseThreshold_R = 0.7,
                    User_ProteinContactThreshold = 5.0,
                    aux_id = 0):


        DIR_ServerFolder = self.DIR_ServerFolder
        # ================================
        # Preprocess Ensemble Predictions
        # ================================
        sxpr_df = pd.read_pickle(DIR_ServerFolder+"/sxpr/Result_EnsembleAvDf.pkl")
        aucg_df = pd.read_pickle(DIR_ServerFolder+"/aucg/Result_EnsembleAvDf.pkl")



        with open(DIR_ServerFolder+"/aucg/ClassName_ClassIndex_Dict.pkl",'rb') as fn:
            aucg_classname_str2int = pickle.load(fn)
            aucg_classname_intstr2str = {str(v):k for k,v in aucg_classname_str2int.items()}
            aucg_classname_int2str = {v:k for k,v in aucg_classname_str2int.items()}


        with open(DIR_ServerFolder+"/sxpr/ClassName_ClassIndex_Dict.pkl",'rb') as fn:
            sxpr_classname_str2int = pickle.load(fn)
            sxpr_classname_intstr2str = {str(v):k for k,v in sxpr_classname_str2int.items()}
            sxpr_classname_int2str = {v:k for k,v in sxpr_classname_str2int.items()}



        aucg_df.loc[:,"User_Task"] = 'AUCG'
        sxpr_df.loc[:,"User_Task"] = 'SXPR'

        # NOTE Finding Dominant Class and predicted value
        PredictionTypeInterested = ['Raw', 'Smoothened', 'SmoothenedDominantTop1', 'SmoothenedDominantTop2']

        for predictiontype in PredictionTypeInterested:
                tempdf = sxpr_df[['%s_0'%(predictiontype), '%s_1'%(predictiontype),'%s_2'%(predictiontype), '%s_3' %(predictiontype)]]
                arr = np.argsort(tempdf.values, axis=1)[:,::-1][:,:2]
                tempdf_twoplaces_name  = pd.DataFrame(tempdf.columns[arr], 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_name'%(predictiontype), '%s_Dominant1_name'%(predictiontype)])
                tempdf_twoplaces_value = pd.DataFrame(tempdf.values[np.arange(len(arr)), arr.T].T, 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_value'%(predictiontype), '%s_Dominant1_value'%(predictiontype)])
                # NOTE class str
                for i in [0,1]:
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].str.replace("%s_"%(predictiontype),"")
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].replace(sxpr_classname_intstr2str)#,regex=True)

                # NOTE Incorporate into the df
                sxpr_df = pd.concat([sxpr_df, tempdf_twoplaces_name, tempdf_twoplaces_value], axis=1)


        for predictiontype in PredictionTypeInterested:
                tempdf = aucg_df[['%s_0'%(predictiontype), '%s_1'%(predictiontype),'%s_2'%(predictiontype), '%s_3' %(predictiontype)]]
                arr = np.argsort(tempdf.values, axis=1)[:,::-1][:,:2]
                tempdf_twoplaces_name  = pd.DataFrame(tempdf.columns[arr], 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_name'%(predictiontype), '%s_Dominant1_name'%(predictiontype)])
                tempdf_twoplaces_value = pd.DataFrame(tempdf.values[np.arange(len(arr)), arr.T].T, 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_value'%(predictiontype), '%s_Dominant1_value'%(predictiontype)])
                # NOTE class str
                for i in [0,1]:
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].str.replace("%s_"%(predictiontype),"")
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].replace(aucg_classname_intstr2str)#,regex=True)

                # NOTE Incorporate into the df
                aucg_df = pd.concat([aucg_df, tempdf_twoplaces_name, tempdf_twoplaces_value], axis=1)

        # NOTE For non site there is no ambiguity. Filter away all non-sites
        sxpr_df_filtered = sxpr_df.loc[~sxpr_df['SmoothenedDominantTop1_Dominant0_name'].isin(['Nonsite'])] # Nonsite
        aucg_df_filtered = aucg_df.loc[~sxpr_df['SmoothenedDominantTop1_Dominant0_name'].isin(['Nonsite'])]

        #print(sxpr_df_filtered)
        # NOTE Base Only. User_Sxpr_VisualiseBaseThreshold = 0.6
        aucg_df_filtered_baseonly = aucg_df_filtered.loc[(sxpr_df_filtered['SmoothenedDominantTop1_Dominant0_name'].isin(['Base'])) 
                                                            & (sxpr_df_filtered['SmoothenedDominantTop1_Dominant0_value'] > User_Sxpr_VisualiseThreshold_Base) ]
        # NOTE Backbone Only. We can  use Bit Fraction (Dominant Top1, Smoothened) as a ramp up
        sxpr_df_filtered_backonly = sxpr_df_filtered.loc[~sxpr_df_filtered['SmoothenedDominantTop1_Dominant0_name'].isin(['Base','Nonsite'])]



        
        # ==============================
        # Filtering
        # ===============================


        # NOTE Quantile of Bit fraction by top1. Not all of them are useful. 
        #      We will use these for filtering as SXPR is unlikely multilabel, but AUCG can be multilabel.
        QuantileOfInterest = [0.3,0.4,0.5,0.6,0.7,0.8] #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0]
        combined_quantile_dict = {}
        for qq in QuantileOfInterest:
            combined_quantile_dict[qq] = {  "SXPR" : sxpr_df_filtered_backonly['Bit Fraction (Dominant Top1, Smoothened)'].quantile(qq), 
                                            "AUCG" : aucg_df_filtered_baseonly['Bit Fraction (Dominant Top1, Smoothened)'].quantile(qq)}
        combinedresult_df = pd.concat([sxpr_df_filtered_backonly,aucg_df_filtered_baseonly],ignore_index=True)
        print(combined_quantile_dict)

        # ============================
        # File production
        # ======================

        # 1. Writing of predictions on grids
        for qq in QuantileOfInterest:
            #print(qq)
            # NOTE To be reset at new quantle
            #del ppdb.df['HETATM']['atom_number']
            ppdb = PandasPdb()
            # NOTE This is the uploaded protein we will append the prediction as HETATM
            #ppdb.read_pdb('%s/uploaded.pdb'%(DIR_ServerFolder)) 
            ppdb.read_pdb(self.DIR_SaveCleansedFolder + "/%s%08d.pdb" %('4tst',aux_id))

            hetatm_indexing = 1
            atom_name_ = []
            residue_name_ = []
            chain_id_ = []
            b_factor_ = [] # NOTE Store the Bit Fraction here
            occupancy_ = [] # NOTE Store the probability of the classes here
            element_symbol_ = []
            atom_number_ = []
            x_coord_ = []
            y_coord_ = []
            z_coord_ = []
            atom_number_ = []
            residue_number_ = []

            for i_row, row in combinedresult_df.iterrows():
                BitFractionQuantile = combined_quantile_dict[qq][row["User_Task"]]


                if row['Bit Fraction (Dominant Top1, Smoothened)'] < BitFractionQuantile:
                    continue

                # NOTE Top two labels if base
                if row['User_Task'] == 'AUCG':
                    topNlabel = [0,1]
                else:
                    topNlabel = [0]


                for top_index in topNlabel:
                    NameOfLabel = row['SmoothenedDominantTop1_Dominant%s_name' %(top_index)]
                    if NameOfLabel in ['Base', 'Nonsite']: # NOTE skipping the base and nonsite in sxpr when they are at second place
                        continue

                    if NameOfLabel in ['P']:
                        if row['SmoothenedDominantTop1_Dominant%s_value'%(top_index)] < User_Sxpr_VisualiseThreshold_P: 
                            continue
                    if NameOfLabel in ['R']:
                        if row['SmoothenedDominantTop1_Dominant%s_value'%(top_index)] < User_Sxpr_VisualiseThreshold_R:
                            continue

                    if NameOfLabel in ['A', 'U', 'C', 'G']:
                        # NOTE Just in case e.g. (0.8, 0.2, 0.0,0.0) which is pretty weak for the second strongest
                        if row['SmoothenedDominantTop1_Dominant%s_value'%(top_index)] < 0.25:
                            continue


                    #print(NameOfLabel)
                    AtomName = AtomNameDict[NameOfLabel]
                    atom_name_.append(AtomName)
                    residue_name_.append(AtomName.upper())
                    element_symbol_.append(AtomName)
                    chain_id_.append("Z")#%s"%(NameOfLabel))
                    b_factor_.append(row['Bit Fraction (Dominant Top1, Smoothened)']) # NOTE If we are going to set transparency by bitfraction? https://pymolwiki.org/index.php/B2transparency
                    occupancy_.append(row['SmoothenedDominantTop1_Dominant%s_value'%(top_index)])
                    x_coord_.append(row['x'])
                    y_coord_.append(row['y'])
                    z_coord_.append(row['z'])
                    atom_number_.append(hetatm_indexing)
                    residue_number_.append(hetatm_indexing%100) # NOTE This resid is meaningless, but we need a place holder
                    hetatm_indexing +=1

            ppdb.df['HETATM']['atom_name'] = atom_name_
            ppdb.df['HETATM']['residue_name'] = residue_name_
            ppdb.df['HETATM']['chain_id'] = chain_id_
            ppdb.df['HETATM']['b_factor'] = b_factor_ # NOTE Store the Bit Fraction here
            ppdb.df['HETATM']['occupancy'] = occupancy_ # NOTE Store the probability of the classes here
            ppdb.df['HETATM']['element_symbol'] = element_symbol_
            ppdb.df['HETATM']['atom_number'] = atom_number_
            ppdb.df['HETATM']['x_coord'] = x_coord_
            ppdb.df['HETATM']['y_coord'] = y_coord_
            ppdb.df['HETATM']['z_coord'] = z_coord_
            ppdb.df['HETATM']['atom_number'] = atom_number_
            ppdb.df['HETATM']['residue_number'] = residue_number_


            hetatm_indexing -=1
            # Patch up the unused flags
            ppdb.df['HETATM']['blank_1'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['blank_2'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['blank_3'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['blank_4'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['insertion'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['alt_loc'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['segment_id'] = ['' for i in range(hetatm_indexing)]
            ppdb.df['HETATM']['record_name'] = ['ATOM' for i in range(hetatm_indexing)]

            tempdf = ppdb.df['HETATM']
            # NOTE Filter perclass by dbscan here
            FilteredCoord = {}
            for lab in ppdb.df['HETATM']['atom_name'].unique():
                RemainingCoord_df = tempdf.loc[ppdb.df['HETATM']['atom_name'] == lab].reset_index()
                #print(RemainingCoord_df)
                RemainingCoord = RemainingCoord_df[['x_coord','y_coord','z_coord']].values

                # DBScan 
                X = RemainingCoord
                db = DBSCAN(eps=2.0, min_samples=6).fit(X)
                labels = db.labels_
                #print(np.unique(labels), lab)
                FilteredCoord[lab] = RemainingCoord_df.iloc[np.where(labels >= 0)[0]]#X[np.where(labels > 0)].tolist()

        
            ppdb.to_pdb(path='%s/%s_%s.pdb' %(DIR_ServerFolder, 'Result', qq) , records=['HETATM'], gz=False, append_newline=True)

            # NOTE DBSCAN Filtered
            ppdb.df['HETATM'] = pd.concat([FilteredCoord[lab] for lab in ppdb.df['HETATM']['atom_name'].unique()],ignore_index = True)
            ppdb.to_pdb(path='%s/%s_%s_Filtered.pdb' %(DIR_ServerFolder, 'Result', qq) , records=['HETATM'], gz=False, append_newline=True)
            del ppdb

        # ================================
        # Collect Orange Sites
        # ================================
        # NOTE This tries to answer 'apple-to-orange' comparison to protein sequence site projection
        # TODO Possibly optimizable either by heuristics or simple ramping. Leave as an exercise for the reader.

        likely_site_xyz = []
        for qq in [0.4,0.6]:
            ppdb = PandasPdb()
            # NOTE This is the uploaded protein we will append the prediction as HETATM
            #ppdb.read_pdb('%s/uploaded.pdb'%(DIR_ServerFolder)) 
            ppdb.read_pdb('%s/%s_%s_Filtered.pdb' %(DIR_ServerFolder, 'Result', qq))
            ppdbdf = ppdb.df['ATOM']

            if qq == 0.6:
                ppdbdf = ppdbdf.loc[ppdbdf['atom_name'].isin(['Re','P'])]
            else:
                ppdbdf = ppdbdf.loc[~ppdbdf['atom_name'].isin(['Re','P'])] # NOTE assumed aucg
            
            xyz = ppdbdf[['x_coord','y_coord','z_coord']].values
            #print(xyz)
            xyz = np.unique(xyz, axis=0)
            likely_site_xyz.append(xyz)
            del ppdb
        #ppdb.to_pdb(path='%s/%s_%s_Filtered.pdb' %(DIR_ServerFolder, 'Result', qq) , records=['HETATM'], gz=False, append_newline=True)
        likely_site_xyz = np.concatenate(likely_site_xyz,axis=0)

        #print(likely_site_xyz)
        ppdb = PandasPdb()
        ppdb.read_pdb("%s/%s%08d.pdb"% (self.DIR_SaveCleansedFolder,'4tst',aux_id))
        proteinpoint  = ppdb.df['ATOM'][['x_coord','y_coord','z_coord']].values
        proteintree = spatial.cKDTree(proteinpoint)

        site_on_protein_atom_lol = proteintree.query_ball_point(likely_site_xyz , User_ProteinContactThreshold, p=2., 
                                                    eps=0, workers=1, return_sorted=None, return_length=False)
        site_on_protein_atom_count = Counter(list(itertools.chain(*site_on_protein_atom_lol)))
        #print(site_on_protein_atom_count)
        max_n_neighboring_likely_site_xyz = max(site_on_protein_atom_count.values())
        site_on_protein_atom_occupancy_dict = {k: v /max_n_neighboring_likely_site_xyz  for k,v in site_on_protein_atom_count.items()}

        # NOTE Write to the CA summing up the count 
        site_occupancy_ = [0.0 for i in range(proteinpoint.shape[0])]
        ppdb.df['ATOM']['occupancy'] = site_occupancy_ # NOTE I will leave bfactor untouched.

        for i in range(proteinpoint.shape[0]):
            if i in site_on_protein_atom_occupancy_dict.keys():
                site_occupancy_[i] = site_on_protein_atom_occupancy_dict[i]
        ppdb.df['ATOM']['occupancy'] = site_occupancy_ # NOTE I will leave bfactor untouched.
        ppdb.to_pdb(path='%s/Result_OrangeSitePerAtom.pdb' %(DIR_ServerFolder) , records=['ATOM'], gz=False, append_newline=True)


        # NOTE a Seq only prediciton will likely need to take max of these values on each residue because the nucleotide can be in touvh with 
        #      sidechain or backbone of the protein residue and it makes no sense to produce a single number?
        #      e.g. a protein residue with a buried sidechain and only the backbone interact with an adenosine (similar to classic kinase binding)
        #      then taking mean will destroy the prediction



        # ===============================
        # Create pse by subprocess
        # ================================
        with open('%s/uploaded.pml' %(DIR_ServerFolder),'w') as fn:
            fn.write("""# ========================
    # Display Set up
    # ========================

    set cartoon_rect_width, 0.1
    set cartoon_rect_length, 0.7
    set cartoon_transparency, 0.2
    set line_radius = 0.1

    set transparency_mode, 1
    set gamma=1.5
    set mesh_radius = 0.01 
    set antialias = 1 
    set stick_radius = 0.22
    set dash_radius=0.07
    set ribbon_radius =0.1 
    #set direct =0.0
    set cartoon_fancy_helices=1
    bg_color white
    set orthoscopic = on
    #util.ray_shadows('none')
    set transparency, 0.5
    set solvent_radius, 0.5
    set cartoon_highlight_color =grey50
    set ray_trace_mode, 0
    set sphere_scale=0.1


    set transparency_mode, 2
    set ray_transparency_contrast, 0.50
    set ray_transparency_oblique, 0.99







    # =========================
    # File Loading
    # =======================
    # NOTE This is the user loaded pdb
    load %s
    # NOTE A recommended level of display for bit fraction level
    #      P and R : 0.6 but for AUCG : 0.4
    #      This avoids displaying low bitfraction predictions (i.e. highly varying predictions in neighborhood of a grid)
    load %s/Result_0.4_Filtered.pdb
    load %s/Result_0.6_Filtered.pdb
    load %s/Result_OrangeSitePerAtom.pdb
    color white, (/4tst00000000/)
    color white, (/Result_OrangeSitePerAtom/)





    # ======================================
    # Retrieve by level of bit fraction
    # ==========================================
    # NOTE Retrieve by level of bit fraction.

    create C, ((Result_0.4_Filtered) & name 'Co')
    create U, ((Result_0.4_Filtered) & name 'U')
    create A, ((Result_0.4_Filtered) & name 'Ar')
    create G, ((Result_0.4_Filtered) & name 'Ge')
    create P, ((Result_0.6_Filtered) & name 'P')
    create R, ((Result_0.6_Filtered) & name 'Re')

    # ================================================================
    # Show degree of feed-forward probability by transparency
    # =================================================================
    # NOTE Occupancy now stores the feed-forward probability 
    #      This is an improved feature for multilabel display.
    set transparency, 0.9, (q > 0.4 & q < 0.5 & name 'P')
    set transparency, 0.85, (q > 0.5 & q < 0.6 & name 'P') 
    set transparency, 0.80, (q > 0.6 & q < 0.7 & name 'P')
    set transparency, 0.75, (q > 0.7 & q < 0.8 & name 'P')
    set transparency, 0.4, (q > 0.8 & q < 0.9 & name 'P')
    set transparency, 0.3, (q > 0.9 & q < 0.95 & name 'P')
    set transparency, 0.2, (q > 0.95 & q < 1.0 & name 'P')

    set transparency, 0.9, (q > 0.4 & q < 0.5 & name 'Re')
    set transparency, 0.85, (q > 0.5 & q < 0.6 & name 'Re') 
    set transparency, 0.80, (q > 0.6 & q < 0.7 & name 'Re')
    set transparency, 0.75, (q > 0.7 & q < 0.8 & name 'Re')
    set transparency, 0.4, (q > 0.8 & q < 0.9 & name 'Re')
    set transparency, 0.3, (q > 0.9 & q < 0.95 & name 'Re')
    set transparency, 0.2, (q > 0.95 & q < 1.0 & name 'Re')

    set transparency, 0.9, (q > 0.0 & q < 0.5 & name 'Ar')
    set transparency, 0.85, (q > 0.5 & q < 0.6 & name 'Ar') 
    set transparency, 0.80, (q > 0.6 & q < 0.7 & name 'Ar')
    set transparency, 0.75, (q > 0.7 & q < 0.8 & name 'Ar')
    set transparency, 0.4, (q > 0.8 & q < 0.9 & name 'Ar')
    set transparency, 0.3, (q > 0.9 & q < 0.95 & name 'Ar')
    set transparency, 0.2, (q > 0.95 & q < 1.0 & name 'Ar')

    set transparency, 0.9, (q > 0.0 & q < 0.5 & name 'U')
    set transparency, 0.85, (q > 0.5 & q < 0.6 & name 'U') 
    set transparency, 0.80, (q > 0.6 & q < 0.7 & name 'U')
    set transparency, 0.75, (q > 0.7 & q < 0.8 & name 'U')
    set transparency, 0.4, (q > 0.8 & q < 0.9 & name 'U')
    set transparency, 0.3, (q > 0.9 & q < 0.95 & name 'U')
    set transparency, 0.2, (q > 0.95 & q < 1.0 & name 'U')

    set transparency, 0.9, (q > 0.0 & q < 0.5 & name 'Ge')
    set transparency, 0.85, (q > 0.5 & q < 0.6 & name 'Ge') 
    set transparency, 0.80, (q > 0.6 & q < 0.7 & name 'Ge')
    set transparency, 0.75, (q > 0.7 & q < 0.8 & name 'Ge')
    set transparency, 0.4, (q > 0.8 & q < 0.9 & name 'Ge')
    set transparency, 0.3, (q > 0.9 & q < 0.95 & name 'Ge')
    set transparency, 0.2, (q > 0.95 & q < 1.0 & name 'Ge')

    set transparency, 0.9, (q > 0.0 & q < 0.5 & name 'Co')
    set transparency, 0.85, (q > 0.5 & q < 0.6 & name 'Co') 
    set transparency, 0.80, (q > 0.6 & q < 0.7 & name 'Co')
    set transparency, 0.75, (q > 0.7 & q < 0.8 & name 'Co')
    set transparency, 0.4, (q > 0.8 & q < 0.9 & name 'Co')
    set transparency, 0.3, (q > 0.9 & q < 0.95 & name 'Co')
    set transparency, 0.2, (q > 0.95 & q < 1.0 & name 'Co')


    # =========================
    # Show Surface
    # =========================
    show surface, (/A/)
    show surface, (/G/)
    show surface, (/U/)
    show surface, (/C/)
    show surface, (/P/)
    show surface, (/R/)

    color red, name Co
    color magenta, name U
    color blue, name Ar
    color cyan, name Ge
    color yellow, name P
    color green, name Re


    # ===========================
    # Orange site occupancy
    # ===========================
    spectrum q, rainbow, (/Result_OrangeSitePerAtom/)
    show surface, (/Result_OrangeSitePerAtom/)
    disable (/Result_OrangeSitePerAtom/)




    hide spheres
    save %s/uploaded.pse
    """ %(os.path.abspath(self.DIR_SaveCleansedFolder + "/%s%08d.pdb" %('4tst',aux_id)),
            os.path.abspath(DIR_ServerFolder),
            os.path.abspath(DIR_ServerFolder),
            os.path.abspath(DIR_ServerFolder),
            os.path.abspath(DIR_ServerFolder)
            )
        )

        # NOTE If pymol at least 2.4 is not installed it will print not found and user has to run it manually
        subprocess.call('pymol -cq %s/uploaded.pml' %(DIR_ServerFolder), shell = True)


        return

    def Downstream_Logo(self,
                        DIR_InputPdbFile = "uploaded/uploaded.pdb",
                        User_ProteinContactThreshold = 5.0,
                        User_SummarisingSphereRadius = 3.0):

        import matplotlib as mpl
        mpl.use('TkAgg')  # or whatever other backend that you want
        import seaborn
        import matplotlib.pyplot as plt
        import matplotlib.patheffects
        from matplotlib import transforms
        from math import sin, cos, acos, sqrt, log
        DIR_ServerFolder = self.DIR_ServerFolder
        # ================================
        # Preprocess Ensemble Predictions
        # ================================
        sxpr_df = pd.read_pickle(DIR_ServerFolder+"/sxpr/Result_EnsembleAvDf.pkl")
        aucg_df = pd.read_pickle(DIR_ServerFolder+"/aucg/Result_EnsembleAvDf.pkl")



        with open(DIR_ServerFolder+"/aucg/ClassName_ClassIndex_Dict.pkl",'rb') as fn:
            aucg_classname_str2int = pickle.load(fn)
            aucg_classname_intstr2str = {str(v):k for k,v in aucg_classname_str2int.items()}
            aucg_classname_int2str = {v:k for k,v in aucg_classname_str2int.items()}


        with open(DIR_ServerFolder+"/sxpr/ClassName_ClassIndex_Dict.pkl",'rb') as fn:
            sxpr_classname_str2int = pickle.load(fn)
            sxpr_classname_intstr2str = {str(v):k for k,v in sxpr_classname_str2int.items()}
            sxpr_classname_int2str = {v:k for k,v in sxpr_classname_str2int.items()}



        aucg_df.loc[:,"User_Task"] = 'AUCG'
        sxpr_df.loc[:,"User_Task"] = 'SXPR'


        # NOTE Finding Dominant Class and predicted value
        PredictionTypeInterested = ['Raw', 'Smoothened', 'SmoothenedDominantTop1', 'SmoothenedDominantTop2']

        for predictiontype in PredictionTypeInterested:
                tempdf = sxpr_df[['%s_0'%(predictiontype), '%s_1'%(predictiontype),'%s_2'%(predictiontype), '%s_3' %(predictiontype)]]
                arr = np.argsort(tempdf.values, axis=1)[:,::-1][:,:2]
                tempdf_twoplaces_name  = pd.DataFrame(tempdf.columns[arr], 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_name'%(predictiontype), '%s_Dominant1_name'%(predictiontype)])
                tempdf_twoplaces_value = pd.DataFrame(tempdf.values[np.arange(len(arr)), arr.T].T, 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_value'%(predictiontype), '%s_Dominant1_value'%(predictiontype)])
                # NOTE class str
                for i in [0,1]:
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].str.replace("%s_"%(predictiontype),"")
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].replace(sxpr_classname_intstr2str)#,regex=True)

                # NOTE Incorporate into the df
                sxpr_df = pd.concat([sxpr_df, tempdf_twoplaces_name, tempdf_twoplaces_value], axis=1)


        for predictiontype in PredictionTypeInterested:
                tempdf = aucg_df[['%s_0'%(predictiontype), '%s_1'%(predictiontype),'%s_2'%(predictiontype), '%s_3' %(predictiontype)]]
                arr = np.argsort(tempdf.values, axis=1)[:,::-1][:,:2]
                tempdf_twoplaces_name  = pd.DataFrame(tempdf.columns[arr], 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_name'%(predictiontype), '%s_Dominant1_name'%(predictiontype)])
                tempdf_twoplaces_value = pd.DataFrame(tempdf.values[np.arange(len(arr)), arr.T].T, 
                                                        #index=tempdf.index, 
                                                        columns= ['%s_Dominant0_value'%(predictiontype), '%s_Dominant1_value'%(predictiontype)])
                # NOTE class str
                for i in [0,1]:
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].str.replace("%s_"%(predictiontype),"")
                    tempdf_twoplaces_name.loc[:,'%s_Dominant%s_name' %(predictiontype, i)] = tempdf_twoplaces_name['%s_Dominant%s_name' %(predictiontype,i)].replace(aucg_classname_intstr2str)#,regex=True)

                # NOTE Incorporate into the df
                aucg_df = pd.concat([aucg_df, tempdf_twoplaces_name, tempdf_twoplaces_value], axis=1)
        

        grid_coord = aucg_df[['x','y','z']].values
        grid_tree = spatial.cKDTree(grid_coord)




        # TODO Move to util
        BackboneGroup = {   "P": ["P", "OP1", "OP2", "O5\'", "O3\'"], 
                            "R": ["C1\'","C2\'","C3\'","C4\'","C5\'", "O3\'", "O4\'"]#,
                            #"D": ["C1\'","C2\'","C3\'","C4\'","C5\'", "O3\'", "O4\'"] 
                        }  
        BaseGroup     = {   "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2",       "N3", "C4"],
                            "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],    
                            "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
                            "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                            "DA": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2",       "N3", "C4"],
                            "DG": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],    
                            "DT": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"],
                            "DC": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                        }


        # ===========================
        # Load User Template structure
        # =====================================
        # NOTE Simply the uploaded file. Not the cleansed one.
        ppdb = PandasPdb()
        CurrentPdbStructure = ppdb.read_pdb(DIR_InputPdbFile)
        
        assert ppdb.df['ATOM'].shape[0] != 0, "ABORTED. User submitted a empty pdb file"

        #print(CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(self.Select_Resname)].columns)
        proteinpoint  = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(self.Select_Resname)][['x_coord','y_coord','z_coord']].values
        proteintree = spatial.cKDTree(proteinpoint)
        nadf = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","C","G","U"])]

        assert nadf.shape[0] != 0, "ABORTED. User submitted a pdb file without nucleotide."

        #NucleotideChainResidTupleList = sorted(set(zip(Nucleic_df["chain_id"].tolist(), Nucleic_df["residue_number"].tolist())))

        # NOTE we can safely assume that each nucleotide takes a unique chainid_resid
        na_chainresdf = [pd.DataFrame(y) for x, y in nadf.groupby(['chain_id','residue_number'], as_index=False)]
        centroid_id = 0
        landmark_df = []
        for local_nadf in na_chainresdf:

            local_nadf = local_nadf[
                ['atom_number',  'atom_name', 'residue_name', 
                'chain_id', 'residue_number', 
                'x_coord', 'y_coord', 'z_coord']]

            # NA name
            na_name = local_nadf["residue_name"].tolist()[0]

            if na_name not in ["A","G","C","U","DA","DT","DC","DG"]:
                na_name = "M" # NOTE Modified base
                continue 

            # Grep base
            basedf = local_nadf.loc[local_nadf['atom_name'].isin(BaseGroup[na_name])]
            # NOTE you may also want to check the completeness of the nucleotide by basedf.shape[0] == len(BaseGroup[na_name])
            if basedf.shape[0] > 0:
                basedf = basedf.assign(centroid_id = centroid_id, 
                                        occupancy = 1.0, 
                                        residue_name = "%s"%(na_name[-1]), # NOTE The last character. This covers DNA base name overlap. 
                                        residue_name_ = "%s" %(na_name))
                landmark_df.append(basedf)
                centroid_id +=1

        try:          
            landmark_df = pd.concat(landmark_df, ignore_index=True)
        except:
            print("ABORTED. There is no base in the supplied template. Made entirely of Modified Base? Check. ABORTED." )
            return
        #print(landmark_df)

        # NOTE Calculate centroid
        landmark_df = landmark_df.groupby(by=[
                "chain_id","residue_number", "residue_name"
                ]).mean().reset_index()

        #centroid = landmark_df[['x_coord', 'y_coord', 'z_coord']].values()
        PART1_CalculateVote = True
        if PART1_CalculateVote:        
            Probability_dict_list =  []
            Logo_dict_list = []
            Logo_dict_list_4 = []
            for i_row, row in landmark_df.iterrows():
                centroid = row[['x_coord', 'y_coord', 'z_coord']].values
                chainres = (row['chain_id'], row['residue_number'])

                if proteintree.query(centroid,1)[0] > User_ProteinContactThreshold:
                        continue

                # NOTE Grid tree here from aucg_df
                grid_within_idx = grid_tree.query_ball_point(centroid, User_SummarisingSphereRadius) # NOTE as we take smoothened.

                # Calculate density within the grid point
                grid_within_aucg = aucg_df.iloc[grid_within_idx]
                #grid_within_sxpr = sxpr_df.iloc[grid_within_idx]

                # NOTE skip when there is none
                if grid_within_aucg.shape[0] == 0:
                    continue


                logo_ = grid_within_aucg[['Bit Fraction (Dominant Top1, Smoothened)',                                                                     # NOTE height4
                                            'SmoothenedDominantTop1_0', 'SmoothenedDominantTop1_1','SmoothenedDominantTop1_2','SmoothenedDominantTop1_3',]  # NOTE p
                                            ].mean(axis = 0).to_frame().T
                logo_ = logo_.rename(columns = {"SmoothenedDominantTop1_%s" %(k):"%s" %(v) for k,v in aucg_classname_intstr2str.items()})


                height4 = logo_['Bit Fraction (Dominant Top1, Smoothened)'].values[0]


                p_U_ = logo_['U'].values[0]
                p_C_ = logo_['C'].values[0]
                p_A_ = logo_['A'].values[0]
                p_G_ = logo_['G'].values[0]
                Logo_dict_list_4.append( {'ChainID':chainres[0], 'ResidueID':chainres[1] , 
                                        'U':height4*p_U_,'C':height4*p_C_,'A':height4*p_A_,'G': height4*p_G_})

        #print(Logo_dict_list_4)
        # =======================
        # Save logo score
        # =======================
        
        performancedf = pd.DataFrame(columns = ['ChainID','ResidueID', 'U','C','A','G'])
        for i in Logo_dict_list_4:
            performancedf = performancedf.append(i, ignore_index=True)
        performancedf['Pyr'] = performancedf['C'] + performancedf['U']
        performancedf['Pur'] = performancedf['A'] + performancedf['G']
        performancedf['Y-R_specificity'] = performancedf['Pyr'] - performancedf['Pur'] 


        for chain in sorted(set(performancedf["ChainID"].tolist())):
            
            temp_performancedf = performancedf.loc[performancedf['ChainID'] == chain]

            maxy = 0.0
            minx = sorted(temp_performancedf['ResidueID'].tolist())[0]
            maxx = sorted(temp_performancedf['ResidueID'].tolist())[-1]

            fig = plt.figure(figsize = (len(range(minx,maxx))+2.4,6))

            #print(len(range(minx,maxx))+1)
            ax = fig.add_subplot(111)
            for index,row in temp_performancedf.iterrows():
                y = 0.0
                for component in ['A','G','U','C']:
                    letterAt(component, row['ResidueID'],y, row['%s' %(component)], ax)

                    y += row['%s' %(component)] 
                    #print(row['%s' %(component)])
                maxy = np.maximum(float(maxy), float(y))
                #print(maxy, index)
                #if (index in [21,22,23,24]):
                    #print(row)
            plt.xticks(range(minx,maxx + 1))
            plt.xlim((minx-1, maxx + 1)) 
            plt.ylim((0, np.minimum(maxy + 0.1, log(6,2))))
            plt.tight_layout()      
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: "{:.1f}".format(x)))
            ax.tick_params(axis = 'x', which = 'major', labelsize = 24)
            #plt.rcParams["figure.figsize"] = (len(list(temp_performancedf.iterrows())),3)
            #plt.gcf().set_size_inches(len(list(temp_performancedf.iterrows())),3)
            plt.savefig('%s/%s_%s_logo_RNACColor.png'%(self.DIR_ServerFolder,'uploaded', chain), dpi = 800)
            plt.savefig('%s/%s_%s_logo_RNACColor.svg'%(self.DIR_ServerFolder,'uploaded', chain), dpi = 800)
            plt.clf()

            #print(Probabilitydf)


        return







    # =========================
    # Misc. Some light utils
    # =========================
    def UTIL_PdbDfDictToStr(self, df, RemoveLinesStartingWith = [], DIR_Save = "", records = []):

        """
        https://github.com/rasbt/biopandas/blob/main/biopandas/pdb/pandas_pdb.py
        """
        if len(records) == 0:
            records = df.keys()

        dfs = {r: df[r].copy() for r in records if not df[r].empty}

        for r in dfs.keys():
            for col in PandasPdbEngines.pdb_records[r]:
                dfs[r][col['id']] = dfs[r][col['id']].apply(col['strf'])
                dfs[r]['OUT'] = pd.Series('', index=dfs[r].index)

            for c in dfs[r].columns:
                # fix issue where coordinates with four or more digits would
                # cause issues because the columns become too wide
                if c in {'x_coord', 'y_coord', 'z_coord'}:
                    for idx in range(dfs[r][c].values.shape[0]): 
                        if len(dfs[r][c].values[idx]) > 8: # NOTE 8 digits including decimal point
                            dfs[r][c].values[idx] = \
                                str(dfs[r][c].values[idx]).strip()
                if c in {'line_idx', 'OUT'}:
                    pass
                elif r in {'ATOM', 'HETATM'} and c not in PandasPdbEngines.pdb_df_columns:
                    warn('Column %s is not an expected column and'
                         ' will be skipped.' % c)
                else:
                    dfs[r]['OUT'] = dfs[r]['OUT'] + dfs[r][c]


        df = pd.concat(dfs, sort=False)
        df.sort_values(by='line_idx', inplace=True)

        s = df['OUT'].tolist()
        for idx in range(len(s)):
            if len(s[idx]) < 80:
                s[idx] = '%s%s' % (s[idx], ' ' * (80 - len(s[idx])))
        if len(RemoveLinesStartingWith) > 0:
            s = [i for i in s if not str(i).startswith(tuple(RemoveLinesStartingWith))]

        to_write = '\n'.join(s)
        del df, dfs
        gc.collect()

        if len(DIR_Save) > 0:
            with open(DIR_Save, 'w') as f:

                for idx in range(len(s)):
                    if len(s[idx]) < 80:
                        s[idx] = '%s%s' % (s[idx], ' ' * (80 - len(s[idx])))

                to_write = '\n'.join(s)
                f.write(to_write)
                f.write('\n')



        return to_write



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