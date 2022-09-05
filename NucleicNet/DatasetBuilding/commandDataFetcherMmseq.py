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



import subprocess
#from biopandas.pdb import PandasPdb
from scipy import spatial
#from scipy.spatial.distance import euclidean
import numpy as np
import scipy
from scipy import sparse
import networkx as nx

import time
import tempfile
import itertools
import functools
import collections
import pickle
import gc
import tqdm
from functools import partial
import multiprocessing

from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandTypify import OOC_RetrieveHaloCoords
import NucleicNet.Fuel.T1
import NucleicNet.Fuel.DS2
import NucleicNet.Fuel.DS3


sys.path.append('../')
# =========================================
# Reciprocal of Std for IPCA component projection
# ==========================================
# NOTE Elements with Std < 1e-3 are zeroed. They show very narrow variance which are prone to distract/overfit
with open('../NucleicNet/DatasetBuilding/util_ReciprocalStd_Projected.pkl','rb') as fn:
    ReciprocalStd_Projected = pickle.load(fn)

# =========================
# Index Fetcher
# =========================
class FetchIndex():
    def __init__(self,  DIR_Feature = "../Database-PDB/feature/",
                        DIR_Typi = "../Database-PDB/typi/", # NOTE This is a dict of classstr to classindex in Typi\
                        n_row_maxhold = 10000, # NOTE This is the segment max size when you save the feature
                        ):  
        """
        This class collects function that return index by default

        """
        self.n_row_maxhold = n_row_maxhold
        self.DIR_Feature = DIR_Feature
        self.DIR_Typi = DIR_Typi
        with open(DIR_Typi + '/ClassIndex.pkl', 'rb') as fn:
            self.TypiContent = pickle.load(fn)

        

    # ==========================
    # Some utils
    # ==========================
    def UTIL_Index_HeadDict(self, index_):
        index_head = ((index_ / self.n_row_maxhold).astype(int) * self.n_row_maxhold).tolist()
        index_localindex = (index_ % self.n_row_maxhold).tolist()
        #print(index_head[0:1000:334], index_[0:1000:334], index_localindex[0:1000:334], "\n")
        # Group by file head 
        headcontain = collections.defaultdict(list)
        for i in range(len(index_head)):
            headcontain[index_head[i]].append(index_localindex[i])
        return headcontain

    # NOTE index_ is a list sorted within the same label
    def UTIL_Index_Feature(self, index_, pdbid = "", featuretype = "altman", ReturnDense = False):

        headdict = self.UTIL_Index_HeadDict(index_)
        featurevectors = []
        for headindex in sorted(headdict.keys()):
            localindex = headdict[headindex]
            fn = self.DIR_Feature + pdbid + ".%s.%s.npz" %(featuretype, headindex)
            with np.load(fn, 
                            mmap_mode='r', allow_pickle=False) as f:
                    ff = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])   
                    if ReturnDense:
                        featurevectors.append(ff[localindex, :].todense())
                    else:
                        featurevectors.append(ff[localindex, :])
        if len(featurevectors) == 0:
            return None
            
        if ReturnDense:
            return np.asarray(np.concatenate(featurevectors, axis=0))
        else:
            return sparse.vstack(featurevectors)


    def UTIL_PdbidList_ClassSizeDict(self,PdbidList = [], LabelTreeLogic = {"RedunSite":  {'union': ["A","U","C","G","P","R","T","D"],
                                                                                'exclu': [],
                                                                                'intersect':['nucsite_']},
                                                                    "Nonsite":  {'union': ["nonsite_"],
                                                                                'exclu': [],
                                                                                 'intersect':[]}
                                                                },):
        datasizedf = {}# []
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
                    pdbid = PdbidList[pdbid_i]
                    with np.load("../Database-PDB/typi/%s.typi.npz" %(pdbid)) as f:
                            typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
                    datapoint_count = len(self.Typi_TreeLogicClass(typi, LabelTreeLogic = LabelTreeLogic, 
                    ReturnSeperateInDict = False))
                    #datasizedf.append([pdbid, datapoint_count])
                    datasizedf[pdbid] = datapoint_count
        #datasizedf = sorted(datasizedf, key = lambda x: x[1])
        #return pd.DataFrame(datasizedf, columns= ['Pdbid', 'Datasize'])

        return datasizedf
    def UTIL_PdbfnList_TotalSizeAllClassesDf(self,PdbfnList = [], LabelTreeLogic = {"RedunSite":  {'union': ["A","U","C","G","P","R","T","D"],
                                                                                'exclu': [],
                                                                                'intersect': ["nucsite_"]},
                                                                    "Nonsite":  {'union': ["nonsite_"],
                                                                                'exclu': [],
                                                                                'intersect': []}
                                                                },):
        #datasizedf = []
        datasizedf = defaultdict(int)
        pdbidfndict = defaultdict(list)
        for pdbfn_i in tqdm.tqdm(range(len(PdbfnList))):
            pdbfn = PdbfnList[pdbfn_i]
            pdbid = pdbfn[:4]
            with np.load("../Database-PDB/typi/%s.typi.npz" %(pdbfn)) as f:
                    typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
            datapoint_count = len(self.Typi_TreeLogicClass(typi, LabelTreeLogic = LabelTreeLogic, 
                        ReturnSeperateInDict = False))
            datasizedf[pdbid] += datapoint_count
            pdbidfndict[pdbid].append(pdbfn)

        datasizedf = [[k,v]for k,v in datasizedf.items()]
        datasizedf = pd.DataFrame(datasizedf, columns= ['Pdbid', 'Datasize']) 
        datasizedf.loc[:,"Pdbfn"] =  datasizedf["Pdbid"].map(pdbidfndict)

        return datasizedf

    def UTIL_PdbidList_SizeEachClass(self,PdbidList = [], LabelTreeLogic = {"RedunSite":  {'union': ["A","U","C","G","P","R","T","D"],
                                                                                'exclu': [],
                                                                                'intersect': ["nucsite_"]},
                                                                    "Nonsite":  {'union': ["nonsite_"],
                                                                                'exclu': [],
                                                                                'intersect': []}
                                                                },):
        datasizedf = {k:0 for k in LabelTreeLogic.keys()}
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
                    pdbid = PdbidList[pdbid_i]
                    with np.load("../Database-PDB/typi/%s.typi.npz" %(pdbid)) as f:
                            typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
                    datapoint_count = self.Typi_TreeLogicClass(typi, LabelTreeLogic = LabelTreeLogic,
                     ReturnSeperateInDict = True)
                    #datasizedf.append([pdbid, datapoint_count])
                    for k,v in datapoint_count.items():
                        print(k,v)
                        datasizedf[k] += len(datapoint_count[k])
                    #datasizedf[pdbid] = datapoint_count
        #datasizedf = sorted(datasizedf, key = lambda x: x[1])
        return datasizedf #pd.DataFrame(datasizedf, columns= ['Pdbid', 'Datasize'])



    def UTIL_PdbidList_ClassSizeAwarePdbidBatches(self,  PdbidList = [], 
                                                LabelTreeLogic = {"RedunSite":  {'union': ["A","U","C","G","P","R","T","D"],
                                                                                'exclu': [ ],
                                                                                'intersect':['nucsite_']},
                                                                    "Nonsite":  {'union': ["nonsite_"],
                                                                                'exclu': [],
                                                                                'intersect':[]},
                                                                                
                                                                },
                                                DesiredDatasize = 1000000,
                                                                ):

        datasizedf = self.UTIL_PdbidList_ClassSizeDict(PdbidList = PdbidList, LabelTreeLogic = LabelTreeLogic)
        datasizedf = [[k,v] for k,v in datasizedf.items()]
        datasizedf = sorted(datasizedf, key = lambda x: x[1])

        PdbidLol = []
        tempcount = 0
        temppdbidlist = []
        for pdbid_i in range(len(datasizedf) + 1):
            if (tempcount > DesiredDatasize) | (pdbid_i == len(datasizedf)):
                if len(temppdbidlist) > 0:
                    PdbidLol.append(temppdbidlist)
                    temppdbidlist = []
                    tempcount = 0
            if (pdbid_i == len(datasizedf)):
                continue

            temppdbidlist.append(datasizedf[pdbid_i][0])
            tempcount += datasizedf[pdbid_i][1]

        return PdbidLol



    # ===========================
    # Body
    # ===========================

    def Typi_SingleClass(self, typi, ClassStr = "nucsite_", ReturnHeadDict = False):
        classindex = self.TypiContent[ClassStr]
        index_ = np.sort(scipy.sparse.find(typi[:,classindex])[0])
        if ReturnHeadDict:
            return self.UTIL_Index_HeadDict(index_)
        else:
            return index_

    def Typi_LeafLogicClass(self, typi, ClassUnion = ["A","U","C","G","P","R","T","D"], 
                                        ClassExclusion = [], 
                                        ClassIntersect = [], 
                                        ReturnHeadDict = False): 

        ClassUnion = [self.TypiContent[i] for i in ClassUnion]
        ClassExclusion = [self.TypiContent[i] for i in ClassExclusion]
        ClassIntersect = [self.TypiContent[i] for i in ClassIntersect]


        index_union = []
        for i in ClassUnion:
            index_union.extend(np.sort(scipy.sparse.find(typi[:,i])[0]).tolist())
        index_union = set(index_union)

        index_exclu = []
        for i in ClassExclusion:
            index_exclu.extend(np.sort(scipy.sparse.find(typi[:,i])[0]).tolist())    
        index_exclu = set(index_exclu)

        index_intersect = []
        for i in ClassIntersect:
            index_intersect.extend(np.sort(scipy.sparse.find(typi[:,i])[0]).tolist())
        index_intersect = set(index_intersect)
        
        #print(len(index_intersect)/len(index_union))# NOTE There are phsopoate and ribose hence > 1
        candidate = (index_union - index_exclu) #& (index_intersect) # NOTE intersect is turned off 
        #if len((index_union - index_exclu)) > 0.0:
        #    print(len(candidate)/ len((index_union - index_exclu)))



        index_ = np.sort(list(candidate))
        if ReturnHeadDict:
            return self.UTIL_Index_HeadDict(index_)
        else:
            return index_

    def Typi_TreeLogicClass(self, typi , LabelTreeLogic =  {"RedunSite":{'union': ["A","U","C","G","P","R","T","D"],
                                                                        'exclu': [],
                                                                        'intersect':["nucsite_"]},
                                                            "Nonsite":  {'union': ["nonsite_"],
                                                                        'exclu': [],
                                                                        'intersect':[]}
                                                            },
                                                            ReturnSeperateInDict = True # NOTE if false return as a mixed set of index
                                                            ):
        # This is the culmulated class

        # TODO assert all written name in TypiContent.keys()
        culmclass_index_allmixed = []
        culmclass_index = {}
        for culmclass , instruction in LabelTreeLogic.items():
            """
            instruction_ = {}
            for k,v in instruction.items():
                if len(v) > 0:
                    instruction_[k] = [self.TypiContent[i] for i in v]
                else:
                    instruction_[k] = []
            """
            index_ = self.Typi_LeafLogicClass(typi, ClassUnion = instruction['union'], 
                                                    ClassExclusion = instruction['exclu'],
                                                    ClassIntersect = instruction['intersect'])
            if ReturnSeperateInDict:
                culmclass_index[culmclass] = index_
            else:
                culmclass_index_allmixed.extend(index_.tolist())
        if ReturnSeperateInDict:
            return culmclass_index
        else:
            #print(set(culmclass_index_allmixed))
            return np.sort(list(set(culmclass_index_allmixed)))





# ==========================
# Define Task
# ============================
def TEMPLATE_TaskNameLabelLogicDict():
        LabelLogic_level0 = {  "Base":  {'union': ["A","U","C","G"],
                                         'exclu': [],
                                         'intersect':[] 
                                        },
                    "Nonsite":  {'union': ["nonsite_"],
                                'exclu': ['F'],
                                'intersect':[]},
                        "P":    {'union': ["P"],
                                'exclu': [],
                                 'intersect':[]},
                        "R":    {'union': ["R"],
                                 'exclu': [],
                                  'intersect':[]}
                                }

        LabelLogic_level1 = {   "A":  {'union': ["A"],
                                        'exclu': [],
                                        'intersect':["nucsite_"]}, # NOTE the intersect with non internal redundant base sites is relaxed with the introduction of weighted pytorch dataloader
                                "U":  {'union': ["U"],
                                        'exclu': [],
                                        'intersect':["nucsite_"]},
                                "C":  {'union': ["C"],
                                        'exclu': [],
                                        'intersect':["nucsite_"]},
                                "G":  {'union': ["G"],
                                        'exclu': [],
                                        'intersect':["nucsite_"]},        
                                        }
        TaskNameLabelLogicDict = {      "SXPR":LabelLogic_level0, 
                                        "AUCG": LabelLogic_level1,
                                }
        return TaskNameLabelLogicDict

def OBSOLETE_TaskNameLabelLogicDict():
    # NOTE A finer class focusing on accuracy in SX. But it needs to train 4 models
    LabelLogic_SX   =       {   "S":    {'union': ["A","U","C","G","P","R"],
                                        'exclu': [],
                                        'intersect':["nucsite_"]},
                                "X":    {'union': ["nonsite_"],
                                        'exclu': [],
                                        'intersect':[]},
                                    }
    LabelLogic_BoneBase   = {   "Bone":     {'union': ["P","R"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                "Base":     {'union': ["A","U","C","G"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                }

    LabelLogic_AUCG   =     {   "A":        {'union': ["A"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                "U":        {'union': ["U"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                "C":        {'union': ["C"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                "G":        {'union': ["G"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                }

    LabelLogic_PR   =       {   "P":        {'union': ["P"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                "R":        {'union': ["R"],
                                            'exclu': [],
                                            'intersect':["nucsite_"]},
                                }

    TaskNameLabelLogicDict = {      "SX":LabelLogic_SX, 
                                    "BoneBase" : LabelLogic_BoneBase,
                                    "AUCG": LabelLogic_AUCG,
                                    "PR": LabelLogic_PR,
                            }    
    return TaskNameLabelLogicDict

class FetchTask():
    def __init__(self,  DIR_DerivedData = "../Database-PDB/DerivedData/",
                        DIR_Typi = "../Database-PDB/typi/",
                        DIR_FuelInput = "../Database-PDB/feature/",


                        Df_grand = None,
                        TaskNameLabelLogicDict = None,

                        n_row_maxhold = 10000):


        import networkx as nx

        self.Df_grand = Df_grand
        self.DIR_FuelInput = DIR_FuelInput
        self.DIR_Typi = DIR_Typi
        self.DIR_DerivedData = DIR_DerivedData
        if TaskNameLabelLogicDict is None:
            self.TaskNameLabelLogicDict = TEMPLATE_TaskNameLabelLogicDict()
        else:
            self.TaskNameLabelLogicDict = TaskNameLabelLogicDict
        self.n_row_maxhold = n_row_maxhold

    # NOTE THis is the default task list
    

    def Return_TaskNameLabelLogicDict(self):
        return TEMPLATE_TaskNameLabelLogicDict()

    def Return_CrossFoldDfList(self, n_CrossFold = 5, ClanGraphBcPercent = 90, 
                                    User_Task = "AUCG", 
                                    Factor_ClampOnMaxSize = 10000,  # NOTE Constraint on datasize of a clan
                                    Factor_ClampOnMultistate = 20,   # NOTE Constriant on number of multistate file read
                                    NmrStates = [],
                                    ):
        
        assert n_CrossFold >= 5
        # NOTE This chekc if the task clan fold etc are present if not create it
        PART1_CheckFileExist = True
        if PART1_CheckFileExist:
            for bc in [100, 95, 90, 70, 50, 40, 30]:
                self.Make_TaskClanFoldDf(n_CrossFold = n_CrossFold, ClanGraphBcPercent = bc, 
                            Factor_ClampOnMaxSize = Factor_ClampOnMaxSize,  # NOTE Constraint on datasize of a clan
                            Factor_ClampOnMultistate = Factor_ClampOnMultistate,
                            NmrStates = NmrStates,
                            User_Task = User_Task)
                #self.Make_SamplingWeightDf(n_CrossFold = n_CrossFold, ClanGraphBcPercent = bc) # NOTE OBSOLETE

        # ========================
        # Task clan fold selection
        # ========================
        with open(self.DIR_DerivedData + "/TaskClanFoldDf_Task%s_Mmseq%s_Fold%s.pkl" %(User_Task, ClanGraphBcPercent,n_CrossFold ), "rb") as fn:
            TaskClanFoldDf = pickle.load(fn)

        TaskClanFoldDf_SelectedTask = TaskClanFoldDf.loc[(TaskClanFoldDf['Task'] == User_Task)]
        TaskClanFoldDf_SelectedTask = TaskClanFoldDf_SelectedTask.sort_values(by= ['Task','CrossFold'], ignore_index= True)
        TaskClanFoldDf_SelectedTask.loc[:,'N_pdbids'] = TaskClanFoldDf_SelectedTask['PdbidList'].map(lambda x: len(x))
        TaskClanFoldDf_SelectedTask.loc[:,'Averagesize'] = TaskClanFoldDf_SelectedTask['Datasize']/TaskClanFoldDf_SelectedTask['N_pdbids']
        # NOTE This is a filter to remove cases where class intereseted only marginally touch the protein.
        TaskClanFoldDf_SelectedTask = TaskClanFoldDf_SelectedTask.loc[ (TaskClanFoldDf_SelectedTask['Averagesize'] > 400) ]





        # NOTE In general, reserve one fold for testing one fold for validation 
        Crossfoldlist = []
        for i in range(n_CrossFold):

            # NOTE Fold i for testing; Fold (i+1,i+2)%n_CrossFold for validation TODO This can be relaxed TODO [[p1 p2 p3],[pa pb]]
            Testing_i = [i]
            Val_i = [(i+1)%n_CrossFold, (i+2)%n_CrossFold]
            Train_i = sorted((set(list(range(n_CrossFold))) - set(Testing_i) ) - set(Val_i))


            Train_PdbidSizedf = TaskClanFoldDf_SelectedTask.loc[TaskClanFoldDf_SelectedTask['CrossFold'].isin(Train_i)][['PdbidList', 'Datasize', 'Averagesize']]
            TrainFold = []
            TrainFold_PdbidSamplingWeight = []
            for _, row in Train_PdbidSizedf.iterrows():
                TrainFold.extend(row['PdbidList'])
                TrainFold_PdbidSamplingWeight.extend( [1/ row['Datasize']]* len(row['PdbidList']))

            Val_PdbidSizedf = TaskClanFoldDf_SelectedTask.loc[TaskClanFoldDf_SelectedTask['CrossFold'].isin(Val_i)][['PdbidList', 'Datasize', 'Averagesize']]
            ValFold = []
            ValFold_PdbidSamplingWeight = []
            for _, row in Val_PdbidSizedf.iterrows():
                ValFold.extend(row['PdbidList'])
                ValFold_PdbidSamplingWeight.extend( [1/ row['Datasize']]* len(row['PdbidList']))

            Testing_PdbidSizedf = TaskClanFoldDf_SelectedTask.loc[TaskClanFoldDf_SelectedTask['CrossFold'].isin(Testing_i)][['PdbidList', 'Datasize', 'Averagesize']]
            TestingFold = []
            TestingFold_PdbidSamplingWeight = []
            for _, row in Testing_PdbidSizedf.iterrows():
                TestingFold.extend(row['PdbidList'])
                TestingFold_PdbidSamplingWeight.extend( [1/ row['Datasize']]* len(row['PdbidList']))




            TrainFold_PdbidSamplingWeight = pd.DataFrame(list(zip(TrainFold, TrainFold_PdbidSamplingWeight)), columns = ["Pdbid", "PdbidSamplingWeight"])
            TrainFold_PdbidSamplingWeight.loc[:,"Datasize"] = Train_PdbidSizedf['Averagesize'] #1/TrainFold_PdbidSamplingWeight["PdbidSamplingWeight"]
            ValFold_PdbidSamplingWeight = pd.DataFrame(list(zip(ValFold, ValFold_PdbidSamplingWeight)), columns = ["Pdbid", "PdbidSamplingWeight"])
            ValFold_PdbidSamplingWeight.loc[:,"Datasize"] = Val_PdbidSizedf['Averagesize'] #1/ValFold_PdbidSamplingWeight["PdbidSamplingWeight"]
            TestingFold_PdbidSamplingWeight = pd.DataFrame(list(zip(TestingFold, TestingFold_PdbidSamplingWeight)), columns = ["Pdbid", "PdbidSamplingWeight"])
            TestingFold_PdbidSamplingWeight.loc[:,"Datasize"] = Testing_PdbidSizedf['Averagesize'] #1/TestingFold_PdbidSamplingWeight["PdbidSamplingWeight"]

            Crossfoldlist.append([
                                (TrainFold, TrainFold_PdbidSamplingWeight),
                                (ValFold, ValFold_PdbidSamplingWeight),
                                (TestingFold, TestingFold_PdbidSamplingWeight)
                                ])

        return Crossfoldlist



    def Make_TaskClanFoldDf(self,n_CrossFold = 5, 
                            ClanGraphBcPercent = 90, 
                            Factor_ClampOnMaxSize = 10000,  # NOTE Constraint on datasize of a clan
                            Factor_ClampOnMultistate = 5, # NOTE Constraint on the number of state for each pdbid
                            NmrStates = [],
                            User_Task = "",
                            ):


        assert len(User_Task) > 0, "ABORTED. Please oprovide User_Task to Make_TaskClanFoldDf"

        NmrPdbidTobeChecked = defaultdict(list)
        for i in NmrStates:
            NmrPdbidTobeChecked[i[:4]].append(i)


        self.ClanGraphBcPercent = ClanGraphBcPercent
        PARTA_MakeClampedTaskClanFold = True
        if PARTA_MakeClampedTaskClanFold:
            if os.path.exists(self.DIR_DerivedData + '/TaskClanFoldDf_Task%s_Mmseq%s_Fold%s.pkl' %(User_Task,ClanGraphBcPercent,n_CrossFold)):
                TaskClanSizeDf_save = pd.read_pickle(self.DIR_DerivedData + "/TaskClanFoldDf_Task%s_Mmseq%s_Fold%s.pkl" %(User_Task,self.ClanGraphBcPercent, n_CrossFold))
            else:
                # =================================================
                # Obtain a List of Pdbid where all info are avail
                # =====================================================
                PART0_CheckPbdidAvailabilityFromMultipleSources = True
                if PART0_CheckPbdidAvailabilityFromMultipleSources:
                    # NOTE BC Clan
                    with open(self.DIR_DerivedData + "/MmseqClanGraph_%s.pkl" %(self.ClanGraphBcPercent), 'rb') as fn:
                        ClanG = pickle.load(fn)
                        ClanG = ClanG.subgraph([i for i in ClanG.nodes if type(i) is str]) 
                        Clans = sorted(list(nx.connected_components(ClanG)) , key=len, reverse=True)
                        Clans = [sorted(i) for i in Clans]



                    # Check available pdbid from various sources
                    pdbavail_Dfgrand = self.Df_grand['Pdbid'].tolist()
                    pdbavail_ClanG = sorted(ClanG.nodes())
                    pdbavail_AnnotationIntersect = sorted(set(pdbavail_Dfgrand) & set(pdbavail_ClanG))

                    pdbavail_Typi = [i.split("/")[-1].split(".")[0] for i in sorted(glob.glob(self.DIR_Typi + '/*.typi.npz'))]
                    pdbavail_Feature = [i.split("/")[-1].split(".")[0] for i in sorted(glob.glob(self.DIR_FuelInput + '/*.0.npz'))]
                    pdbavail_DatasetIntersect = sorted(set(pdbavail_Typi) & set(pdbavail_Feature))


                    # Prefiltering the clans
                    Clans_ = []
                    for c in Clans:
                        cc = sorted(set(c) & set(pdbavail_AnnotationIntersect))
                        if len(cc) == 0:
                            continue
                        Clans_.append(cc)
                    Clans = Clans_

                    # Mapping Pdbfn Clan membership
                    ClanToPdbidDict = {}
                    ClanToPdbfnDict = {}
                    PdbfnToClanDict = {}
                    PdbidToClanDict = {}
                    clan_i = 0
                    for c in Clans:
                        # Find pdbfn in clan
                        matched_pdbfn_ = sorted([i for i in pdbavail_DatasetIntersect if str(i).startswith(tuple(c))])
                        if len(matched_pdbfn_) == 0:
                            continue                      

                        # Find any match with NMR state provided
                        if len(NmrStates) > 0:
                            matched_pdbfn__ = []
                            for i in matched_pdbfn_:
                                if i[:4] in NmrPdbidTobeChecked.keys():
                                    if i in NmrPdbidTobeChecked[i[:4]]:
                                        matched_pdbfn__.append(i)
                                        #print(i)
                                    else:
                                        pass
                                else:
                                    matched_pdbfn__.append(i)
                            matched_pdbfn_ = matched_pdbfn__

                        # Clamp on multistate here
                        matched_pdbfn  =[]
                        tempcount = {i[:4]:0 for i in matched_pdbfn_}
                        for i in matched_pdbfn_:
                            if tempcount[i[:4]] > Factor_ClampOnMultistate:
                                continue
                            matched_pdbfn.append(i)
                            tempcount[i[:4]] +=1

                        matched_pdbid = sorted(set([i[:4] for i in matched_pdbfn]))

                        ClanToPdbfnDict[clan_i] = matched_pdbfn
                        ClanToPdbidDict[clan_i] = matched_pdbid
                        for i in matched_pdbfn:
                            PdbfnToClanDict[i] = clan_i

                        for i in matched_pdbid:
                            PdbidToClanDict[i] = clan_i

                        clan_i += 1


                    pdbfnavail = sorted(PdbfnToClanDict.keys())
                    pdbidavail = sorted(PdbidToClanDict.keys())
                # ======================================================
                # Bound on Clan Size 
                # ======================================================
                # NOTE This prevents extremely abundant clan e.g. histone bound dna from dominating the data
                #      We set a max datasize for each clan at max of the largest pdbentry at around 2000000
                FetchIndexC = FetchIndex(DIR_Feature = self.DIR_FuelInput, 
                                        DIR_Typi = self.DIR_Typi, n_row_maxhold = self.n_row_maxhold)

                PART1_GetClassPdbidDatasize = True
                if PART1_GetClassPdbidDatasize:
                    TaskNameDatasize = {}
                    for taskname, labellogic in self.TaskNameLabelLogicDict.items():
                        if taskname != User_Task: # NOTE This accomodates situation where a less selective set of pdbid is tolerated by an easier task e.g. SXPR
                            continue
                        print(labellogic)
                        datasizedf = FetchIndexC.UTIL_PdbfnList_TotalSizeAllClassesDf(PdbfnList = pdbfnavail, 
                                                                                    LabelTreeLogic = labellogic)
                        datasizedf = datasizedf.rename(columns={"Datasize": "Datasize_%s" %(taskname)})
                        TaskNameDatasize[taskname] = datasizedf



                    #print(TaskNameDatasize)
                    DatasizeDfGrand = functools.reduce(lambda x, y: pd.merge(x, y, on = 'Pdbid'), 
                                                                    [v for v in TaskNameDatasize.values()])

                    DatasizeDfGrand = DatasizeDfGrand.rename(columns={"Pdbfn_x": "Pdbfn"})
                    Df_grand = pd.read_pickle(self.DIR_DerivedData + "/DataframeGrand.pkl")
                    Df_grand = Df_grand.loc[Df_grand["Pdbid"].isin(pdbidavail)]
                    Df_grand = pd.merge(Df_grand, DatasizeDfGrand, on ='Pdbid')
                    Df_grand.loc[:,"Clan%s" %(self.ClanGraphBcPercent)] = Df_grand["Pdbid"].map(PdbidToClanDict)






                PART2_GroupPdbidIntoClanClampOnSize = True
                if PART2_GroupPdbidIntoClanClampOnSize:
                    TaskClanDict = {}
                    #for task in ["Datasize_AUCG", "Datasize_PR", "Datasize_BoneBase", "Datasize_SX"]:
                    for taskname,labellogic in self.TaskNameLabelLogicDict.items():
                            if taskname != User_Task:
                                continue
                            task = "Datasize_%s" %(taskname)
                            tempclandict = collections.defaultdict(list)

                            totaldatasize = 0
                            for clanid, tempinfodf in tqdm.tqdm(Df_grand.groupby("Clan%s" %(self.ClanGraphBcPercent))):
                                    #tempinfodf = tempinfodf.round({'Resolution': 0})    # NOTE Round to integer
                                    tempinfodf.loc[:,"Resolution"] = ((tempinfodf["Resolution"]/0.25).astype(int))*0.25

                                    tempinfodf = tempinfodf.sort_values(by = ["Resolution", task], 
                                                                    ascending=[True, False], ignore_index = True)
                                    #print(tempinfodf)
                                    datasizecount = 0
                                    for row_i, row in tempinfodf.iterrows():

                                            if datasizecount > Factor_ClampOnMaxSize :
                                                    break
                                            
                                            if row[task] == 0: # NOTE This means there are no such task for this particular pdbid. e.g. only making backbone contact
                                                    continue

                                            #tempclandict[clanid].append(str(row['Pdbid']))
                                            tempclandict[clanid].append(row['Pdbfn'])
                                            datasizecount += row[task]

                                    totaldatasize+=datasizecount

                                    # NOTE I will use the final entry as a record of datasize 
                                    tempclandict[clanid].append(datasizecount)

                            print(task, totaldatasize)
                            TaskClanDict[taskname] = tempclandict





                # ================================
                # Separate into fold
                # ================================
                PART3_SeparateClanIntoFold = True
                if PART3_SeparateClanIntoFold:
                    TaskClanSizeDf = []
                    for taskname, clandict in TaskClanDict.items():
                        if taskname != User_Task:
                            continue
                        for k, v in clandict.items():
                            TaskClanSizeDf.append([k, v[-1], taskname, v[:-1]])
                    # NOTE THe PdbidList here is actually PdbfnList
                    TaskClanSizeDf = pd.DataFrame(TaskClanSizeDf, columns = ["Clan", "Datasize", "Task", "PdbidList"])
                    TaskClanSizeDf.loc[:,'PdbidList'] = TaskClanSizeDf['PdbidList'].map(lambda x: list(itertools.chain(*x)))

                    TaskFoldClanDict = {}
                    print("Breakdown of fold datasize")
                    for task, tempinfodf in TaskClanSizeDf.groupby("Task"):
                        tempinfodf = tempinfodf.sort_values(by=["Datasize"], ignore_index = True)
                        tempinfodf = tempinfodf.loc[tempinfodf['Datasize'] > 0]

                        clan_SortBySize = tempinfodf['Clan'].tolist()
                        size_SortBySize = tempinfodf['Datasize'].tolist()
                        #print(sorted(size_SortBySize)[0])
                        crossfolddict = collections.defaultdict(list)
                        foldsizedict = {}
                        for i in range(n_CrossFold):
                            foldsizedict[i] = 0
                        for i in range(len(clan_SortBySize)):
                            crossfolddict[i%n_CrossFold].append(clan_SortBySize[i])
                            foldsizedict[i%n_CrossFold] += size_SortBySize[i]


                        clanidfolddict = {}
                        for k,v in crossfolddict.items():
                            for clanid in v:
                                clanidfolddict[clanid] = int(k)

                        TaskFoldClanDict[task] = clanidfolddict
                        print(task, foldsizedict)

                    TaskClanSizeDf_save = []
                    for task, tempinfodf in TaskClanSizeDf.groupby("Task"):
                        tempinfodf.loc[:,"CrossFold"] = tempinfodf["Clan"].map(TaskFoldClanDict[task]) 
                        # TODO When datasize == 0 this produce nan because dictionary mapping cannot find key
                        TaskClanSizeDf_save.append(tempinfodf)
                    TaskClanSizeDf_save = pd.concat(TaskClanSizeDf_save, ignore_index= True)
                    TaskClanSizeDf_save = TaskClanSizeDf_save.loc[TaskClanSizeDf_save['Datasize'] > 0] 
                    # Related to TODO above






                # =================
                # Save
                # ===================
                
                TaskClanSizeDf_save.to_pickle(self.DIR_DerivedData + "/TaskClanFoldDf_Task%s_Mmseq%s_Fold%s.pkl" %(User_Task, self.ClanGraphBcPercent, n_CrossFold),protocol = 4)



        return TaskClanSizeDf_save


    def OBSOLETE_Make_SamplingWeightDf(self,n_CrossFold = 5, ClanGraphBcPercent = 90, Factor_ClampOnMaxSize = 2.0):    
        if os.path.exists(self.DIR_DerivedData + '/TaskClanFoldDf_Mmseq%s_Fold%s.pkl' %(ClanGraphBcPercent,n_CrossFold)):
            TaskClanSizeDf_save = pd.read_pickle(self.DIR_DerivedData + "/TaskClanFoldDf_Mmseq%s_Fold%s.pkl" %(self.ClanGraphBcPercent, n_CrossFold))
        else:
            print("ABRODTED. Run Make_TaskClanFoldDf")
            return


        FetchIndexC = FetchIndex(DIR_Feature = self.DIR_FuelInput, 
                                DIR_Typi = self.DIR_Typi, n_row_maxhold = self.n_row_maxhold)

        PARTB_StatOnClass = True
        if PARTB_StatOnClass:
            if os.path.exists(self.DIR_DerivedData + "/SamplingWeightDf_Mmseq%s_Fold%s.pkl" %(self.ClanGraphBcPercent, n_CrossFold)):
                samplingweightdf = pd.read_pickle(self.DIR_DerivedData + "/SamplingWeightDf_Mmseq%s_Fold%s.pkl" %(self.ClanGraphBcPercent, n_CrossFold))
            else:
                # Also make the stat on class
                TaskClanFoldDf = TaskClanSizeDf_save
                samplingweightdf = []
                for User_Task in pd.unique(TaskClanFoldDf['Task']):
                    TaskClanFoldDf_SelectedTask = TaskClanFoldDf.loc[(TaskClanFoldDf['Task'] == User_Task)]
                    TaskClanFoldDf_SelectedTask = TaskClanFoldDf_SelectedTask.sort_values(by= ['Task','CrossFold'], ignore_index= True)
                    Size_EachClass_Dict = FetchIndexC.UTIL_PdbidList_SizeEachClass(
                                            PdbidList = sorted(itertools.chain(*TaskClanFoldDf_SelectedTask['PdbidList'].tolist())),
                                            LabelTreeLogic = self.TaskNameLabelLogicDict[User_Task]
                                            )

                    #sampling_weight = [1.0/Size_EachClass_Dict[k] for k in sorted(Size_EachClass_Dict.keys())]
                    classidx = 0
                    for k in sorted(Size_EachClass_Dict.keys()):
                        try:
                            samplingweightdf.append([ClanGraphBcPercent, n_CrossFold, User_Task, 
                                                k, classidx, 
                                                1.0/Size_EachClass_Dict[k], Size_EachClass_Dict[k]])
                        except KeyError:
                            print(Size_EachClass_Dict)
                        classidx +=1

                samplingweightdf = pd.DataFrame(samplingweightdf, columns=['BC Percent', 'Number Cross Folds', "Task",
                                                                    "Class Name", "Class Index",
                                                                    "Sampling Weight", "Datasize"])
                samplingweightdf.to_pickle(self.DIR_DerivedData + "/SamplingWeightDf_Mmseq%s_Fold%s.pkl" %(self.ClanGraphBcPercent, n_CrossFold))

        return samplingweightdf


    def Return_SamplingWeightDf(self,n_CrossFold = 5, ClanGraphBcPercent = 90, User_Task = "AUCG"):

        if os.path.exists(self.DIR_DerivedData + "/SamplingWeightDf_Mmseq%s_Fold%s.pkl" %(ClanGraphBcPercent, n_CrossFold)):
            df = pd.read_pickle(self.DIR_DerivedData + "/SamplingWeightDf_Mmseq%s_Fold%s.pkl" %(ClanGraphBcPercent, n_CrossFold))
        else:
            print("ABORTED. File not found. Run Make_SamplingWeightDf")
            return

        if User_Task is not None:
            return df.loc[df['Task'] == User_Task]
        else:
            return df


    def Return_DictClassToHaloIndex(self, pdbid, 
        User_Task = ""):

        # ======================================================
        # Bound on Clan Size 
        # ======================================================
        # NOTE This prevents extremely abundant clan e.g. histone bound dna from dominating the data
        #      We set a max datasize for each clan at max of the largest pdbentry at around 2000000
        FetchIndexC = FetchIndex(DIR_Feature = self.DIR_FuelInput, 
                                DIR_Typi = self.DIR_Typi, n_row_maxhold = self.n_row_maxhold)

        classindex_str = sorted(self.TaskNameLabelLogicDict[User_Task].keys()) 
        classindex_int = dict(zip(classindex_str, range(len(classindex_str))))
        # =====================
        # Load Typi (Fast)
        # =====================
        with np.load("%s/%s.typi.npz" %(self.DIR_Typi, pdbid)) as f:
                typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])

        # NOTE obtain Index where class is available
        Class_To_Idx_Dict = FetchIndexC.Typi_TreeLogicClass(typi , 
                                LabelTreeLogic = self.TaskNameLabelLogicDict[User_Task],
                                ReturnSeperateInDict = True)
        # ====================
        # Get target (Fast)
        # ====================
        # NOTE both should be list of list
        #idx_y = {}
        idx_x = {} 
        for c in sorted(Class_To_Idx_Dict.keys()):

            if len(Class_To_Idx_Dict[c]) == 0: # NOTE This handles when a class is missing in the pdb entry
                continue

            idx_x[c] = Class_To_Idx_Dict[c] # NOTE THis is the halo index
            #idx_y[c] = [classindex_int[c]] * len(Class_To_Idx_Dict[c])
        return idx_x



# ==========================
# Fetch Feature
# ===========================
import NucleicNet.Fuel.T1
import NucleicNet.Fuel.DS2

def OOC_Pdbid_TorchDataset(pdbid, classindex_int = {}, 
                                Assigned_PdbidWeight = {},
                                feature_mean = None, #IPCA_Mean,
                                feature_std = None, #IPCA_Std,
                                feature_resizeshape = None, #(6,80)
                                feature_component_size = 60,
                                feature_component = None,
                                feature_einsum_mean = None,
                                feature_einsum_std = None,
                                data_stride = 1, 
                                FetchIndexC = None,
                                LabelTreeLogic = None,
                                ReturnRawAltman = False,
                                User_NeighborLabelSmoothAngstrom = 1.5,
                                DIR_Typi = "../Database-PDB/typi/",
                                ):

        # =====================
        # Load Typi (Fast)
        # =====================
        
        with np.load("%s/%s.typi.npz" %(DIR_Typi, pdbid)) as f:
                typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
    
        # NOTE obtain Index where class is available
        Class_To_Idx_Dict = FetchIndexC.Typi_TreeLogicClass(typi , 
                                LabelTreeLogic = LabelTreeLogic,
                                ReturnSeperateInDict = True)

        # ====================
        # Get target (Fast)
        # ====================
        # NOTE both should be list of list
        idx_y = {}
        idx_x = {} 
        for c in sorted(Class_To_Idx_Dict.keys()):

            if len(Class_To_Idx_Dict[c]) == 0: # NOTE This handles when a class is missing in the pdb entry
                continue

            idx_x[c] = Class_To_Idx_Dict[c]
            idx_y[c] = [classindex_int[c]] * len(Class_To_Idx_Dict[c])

        # NOTE apply stride
        if data_stride > 1:
            for c in sorted(Class_To_Idx_Dict.keys()):
                if len(Class_To_Idx_Dict[c]) == 0: # NOTE This handles when a class is missing in the pdb entry
                    continue
                #print("classsss ", c) 
                if c not in ["Nonsite"]:
                    continue
                idx_y[c] = idx_y[c][::data_stride]
                idx_x[c] = idx_x[c][::data_stride]
            #sys.exit()

        # =======================
        # Get feature (Slow)
        # ========================
        train_x = []
        train_y = []
        total_leng = 0
        #st = time.time()
        for c in sorted(Class_To_Idx_Dict.keys()):
            if len(Class_To_Idx_Dict[c]) == 0: # NOTE This handles when a class is missing in the pdb entry
                continue
            train_x__ = FetchIndexC.UTIL_Index_Feature(idx_x[c], 
                                    pdbid = pdbid, featuretype = "altman", ReturnDense = True)
            train_x.append(train_x__)
            train_y.extend(idx_y[c])
            total_leng += len(idx_y[c])

        try:
            train_x = np.concatenate(train_x, axis=0)
        except:
            print(pdbid, 'no data! Is cuda out of memory?')
            return None, None, None, pdbid

        train_y_hot = torch.nn.functional.one_hot(torch.LongTensor(train_y).type(torch.int64), num_classes=len(sorted(Class_To_Idx_Dict.keys()))).type(torch.float32)
        train_yyyy = torch.zeros(total_leng, len(sorted(Class_To_Idx_Dict.keys())))
        #print(train_yyyy.shape)

        if User_NeighborLabelSmoothAngstrom > 0.0:

            # index on the lattice
            avail_idx = []
            for c in sorted(Class_To_Idx_Dict.keys()):
                if len(Class_To_Idx_Dict[c]) == 0: # NOTE This handles when a class is missing in the pdb entry
                    continue
                # Get the index of all its 9 neighbors which is registered in one of the classes defined by current task. 
                avail_idx.extend(idx_x[c])
            avail_idx = np.array(avail_idx).astype(int)
            #print(idx_x[c])
            with open("../Database-PDB/halo/%s.halotup"%(pdbid),'rb') as fn:
                halo_tuple = pickle.load(fn)
            halo_coord = OOC_RetrieveHaloCoords(halo_tuple)
            halo_coord_selected = halo_coord[avail_idx]
            halo_tree = spatial.cKDTree(halo_coord_selected)
            # NOTE These are index starting 0 to len(halo_coord_selected)-1
            neighbor_selectedindex_lol = halo_tree.query_ball_point(halo_coord_selected, User_NeighborLabelSmoothAngstrom)


            #print(len(neighbor_selectedindex_lol))
            #print(len(neighbor_selectedindex_lol), 'total datapoint')
            nnnn = []
            for i in range(len(neighbor_selectedindex_lol)):

                self_label = train_y_hot[i].detach().clone()
                n_nei = len(neighbor_selectedindex_lol[i])
                nnnn.append(n_nei)
                for j in neighbor_selectedindex_lol[i]:
                    #print(train_y_hot[j])
                    # This has the same index as train_y
                    self_label += train_y_hot[j]
                #print(self_label, n_nei)
                self_label = self_label / (n_nei + 1)
                #print(self_label)

                train_yyyy[i] += self_label
            #print(nnnn)
            #print(pdbid, 
            #        train_yyyy[(train_yyyy < 1.0) & (train_yyyy > 0.0)].shape, 
            #            train_yyyy.shape, torch.mean(train_yyyy[(train_yyyy < 0.5) & (train_yyyy > 0.0)]))

        else:
            train_yyyy = train_y_hot






        # Find out those vectors with first two shells largely empty
        nonzerocount  = np.count_nonzero(train_x[:,:160], axis = 1)
        nonzerocount = np.where(nonzerocount > 16)[0] # Require at least 10% non-zero

        oldshape = train_x.shape[0]

        # Only use those with nonzerocount for training
        #train_x = train_x[nonzerocount]

        train_y = np.array(train_y, dtype= int)
        #train_y = train_y[nonzerocount]
        #train_yyyy = train_yyyy[nonzerocount]
        if train_x.shape[0]/oldshape < 0.25:
            print("This pdbid has less than 25 percent datapoint left. Check it. Is it a candidate for removal?", pdbid)
            print(train_x.shape[0]/oldshape)

        train_x_size = train_x.shape[0]
        train_x = torch.FloatTensor(train_x)
        train_y = torch.LongTensor(train_y)


        if ReturnRawAltman:
            return train_x, train_y, train_yyyy, pdbid

        # ==========================
        # Process feature (Fast)
        # ==========================
        #if feature_mean is not None:                                   # NOTE Close to zero. But the global mean is tested batch sensitive. I will leave this to Batch norm.
        #    train_x -= torch.FloatTensor(feature_mean)
            
        if feature_std is not None: 
            train_x /= torch.FloatTensor(feature_std)

        train_x = torch.nan_to_num(train_x, nan = 0.0, posinf = 0.0, neginf = 0.0)




        if feature_resizeshape is not None:
            train_x.resize_((train_x_size, *feature_resizeshape)).unsqueeze_(1)

        if feature_component is not None:
            assert (feature_resizeshape is not None)
            #assert (feature_einsum_mean is not None)
            #assert (feature_einsum_std is not None)
            feature_component = feature_component[:feature_component_size,:]
            feature_component = torch.FloatTensor(feature_component)
            feature_component.resize_((feature_component_size, *feature_resizeshape))
            #train_x[:,:,:,2] = 0.0 # NOTE Remove c-alpha as a feature 
            #train_x[:,:,:,6] = 0.0 # NOTE Remove nitrogen sp2 with hydrogen i.e. histidine as a feature
            train_x = torch.einsum('bcsf,gsf->bcsg', train_x, feature_component)

            Reduction_reciprocalstd = torch.tensor(ReciprocalStd_Projected[:,:feature_component_size], dtype = torch.float32)
            Reduction_reciprocalstd.unsqueeze_(0)

            train_x = torch.einsum('bsf,bcsf->bcsf', Reduction_reciprocalstd,train_x)
            
        train_x = torch.nan_to_num(train_x, nan = 0.0, posinf = 0.0, neginf = 0.0)

        # TODO Clamp
        sign = train_x.sign()
        train_x.abs_()
        train_x[train_x <= 1e-6] = 0.0
        train_x *= sign



        #ds = NucleicNet.Fuel.DS2.BasicMap(train_x, train_y)
        #return ds, pdbid
        return train_x, train_y, train_yyyy, pdbid


def OOC_Pdbid_TorchNoOpSingleDataset(pdbid, classindex_int = {}, 
                                Assigned_PdbidWeight = {},
                                feature_mean = None, #IPCA_Mean,
                                feature_std = None, #IPCA_Std,
                                feature_resizeshape = None, #(6,80)
                                feature_component_size = 60,
                                feature_component = None,
                                feature_einsum_mean = None,
                                feature_einsum_std = None,
                                data_stride = 1, 
                                FetchIndexC = None,
                                LabelTreeLogic = None,
                                ReturnRawAltman = False,
                                User_NeighborLabelSmoothAngstrom = 1.5,
                                DIR_Typi = "../Database-PDB/typi/",
                                ):

        # =====================
        # Load Typi (Fast)
        # =====================
        
        with np.load("%s/%s.typi.npz" %(DIR_Typi, pdbid)) as f:
                typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
    
        # NOTE obtain Index where class is available
        Class_To_Idx_Dict = FetchIndexC.Typi_TreeLogicClass(typi , 
                                LabelTreeLogic = LabelTreeLogic,
                                ReturnSeperateInDict = True)
        TotalNumberOfX = typi.shape[0]
        idx_x = np.array(list(range(TotalNumberOfX))).astype(int)

        if data_stride > 1 :
            idx_x = idx_x[::data_stride]
        
        # =======================
        # Get feature (Slow)
        # ========================
        train_x = FetchIndexC.UTIL_Index_Feature(idx_x, 
                                    pdbid = pdbid, featuretype = "altman", ReturnDense = True)
        train_y = [0 for i in idx_x]
        total_leng = len(idx_x)
        train_y_hot = torch.nn.functional.one_hot(torch.LongTensor(train_y).type(torch.int64), num_classes=len(sorted(Class_To_Idx_Dict.keys()))).type(torch.float32)
        train_yyyy = torch.zeros(total_leng, len(sorted(Class_To_Idx_Dict.keys())))
        #print(train_yyyy.shape)

        train_yyyy = train_y_hot




        train_y = np.array(train_y, dtype= int)



        train_x_size = train_x.shape[0]
        train_x = torch.FloatTensor(train_x)
        train_y = torch.LongTensor(train_y)


        if ReturnRawAltman:
            return train_x, train_y, train_yyyy, pdbid

        # ==========================
        # Process feature (Fast)
        # ==========================
        #if feature_mean is not None:                                   # NOTE Close to zero. But the global mean is tested batch sensitive. I will leave this to Batch norm.
        #    train_x -= torch.FloatTensor(feature_mean)
            
        if feature_std is not None: 
            train_x /= torch.FloatTensor(feature_std)

        train_x = torch.nan_to_num(train_x, nan = 0.0, posinf = 0.0, neginf = 0.0)




        if feature_resizeshape is not None:
            train_x.resize_((train_x_size, *feature_resizeshape)).unsqueeze_(1)

        if feature_component is not None:
            assert (feature_resizeshape is not None)
            #assert (feature_einsum_mean is not None)
            #assert (feature_einsum_std is not None)
            feature_component = feature_component[:feature_component_size,:]
            feature_component = torch.FloatTensor(feature_component)
            feature_component.resize_((feature_component_size, *feature_resizeshape))
            #train_x[:,:,:,2] = 0.0 # NOTE Remove c-alpha as a feature 
            #train_x[:,:,:,6] = 0.0 # NOTE Remove nitrogen sp2 with hydrogen i.e. histidine as a feature
            train_x = torch.einsum('bcsf,gsf->bcsg', train_x, feature_component)

            Reduction_reciprocalstd = torch.tensor(ReciprocalStd_Projected[:,:feature_component_size], dtype = torch.float32)
            Reduction_reciprocalstd.unsqueeze_(0)

            train_x = torch.einsum('bsf,bcsf->bcsf', Reduction_reciprocalstd,train_x)
            
        train_x = torch.nan_to_num(train_x, nan = 0.0, posinf = 0.0, neginf = 0.0)

        # TODO Clamp
        sign = train_x.sign()
        train_x.abs_()
        train_x[train_x <= 1e-6] = 0.0
        train_x *= sign


        #print(train_x, train_y, train_yyyy, pdbid)
        #ds = NucleicNet.Fuel.DS2.BasicMap(train_x, train_y)
        #return ds, pdbid
        return train_x, train_y, train_yyyy, pdbid


class FetchDataset():
    def __init__(self,
        DIR_DerivedData = "../Database-PDB/DerivedData/",
        DIR_Typi = "../Database-PDB/typi/",
        DIR_FuelInput = "../Database-PDB/feature/",


        User_DesiredDatasize    = 6000000, # NOTE This controls the number of new batch of dataset-dataloader being reloaded into memory
        User_SampleSizePerEpoch_Factor = 1.0, # NOTE This controls how much sample enters into an epoch
        User_featuretype = 'altman',

        n_datasetworker = 24,
        n_row_maxhold = 10000,
        TaskNameLabelLogicDict = None,
        ClanGraphBcPercent = 90):



        self.DIR_DerivedData = DIR_DerivedData
        self.DIR_FuelInput = DIR_FuelInput
        self.DIR_Typi = DIR_Typi
        self.User_featuretype = User_featuretype

        self.n_row_maxhold = n_row_maxhold

        self.n_datasetworker = n_datasetworker


        if TaskNameLabelLogicDict is None:
            self.TaskNameLabelLogicDict = TEMPLATE_TaskNameLabelLogicDict()

        # NOTE Resampling will be done within batch to balance the class size
        #      NOTE That this is separated from the minibatch mechansim
        self.User_SampleSizePerEpoch = int(User_DesiredDatasize * User_SampleSizePerEpoch_Factor)

    def GetIpca(self, User_Task):
        #with open(self.DIR_DerivedData + "Feature.IPCA.%s.%s.pkl" %(self.User_featuretype, User_Task), "rb") as fn:
        with open(self.DIR_DerivedData + "Feature.IPCA.%s.%s.pkl" %(self.User_featuretype, "AUCG"), "rb") as fn:
            IPCA = pickle.load(fn)
        if help:
            """
            IPCA_Mean = IPCA.mean_
            IPCA_Std = (IPCA.var_)**0.5
            IPCA_Components = IPCA.components_
            """
        return IPCA




    def GetDataset(self,Assigned_PdbidBatch = [],
                        Assigned_PdbidWeight = {},
                        ClassName_ClassIndex_Dict = {"A":0, "C": 1, "G": 2, "U": 3},
                        User_Task = "AUCG",
                        User_datastride = 1,
                        User_NumReductionComponent = None,
                        PerformZscoring = True, PerformReduction = False,
                        StringentValidation = True,
                        ReturnRawAltman = False,
                        TestingPhase = False,
                        User_NeighborLabelSmoothAngstrom = 0.0,
                        ):
        FetchIndexC = FetchIndex(DIR_Feature = self.DIR_FuelInput, 
                                    DIR_Typi = self.DIR_Typi, n_row_maxhold = self.n_row_maxhold)
        #print(Assigned_PdbidWeight)
        #print(Assigned_PdbidBatch)
        n_feature_per_shell = 80
        IPCA = self.GetIpca(User_Task)
        if PerformZscoring:
            IPCA_Mean = IPCA.mean_
            IPCA_Std = (IPCA.var_)**0.5
        else:
            IPCA_Mean = None
            IPCA_Std = None
        
        if PerformReduction:
            assert (User_NumReductionComponent is not None), "ABORTED. Specify User_NumReductionComponent"
            n_feature_per_shell = User_NumReductionComponent
            IPCA_Mean = IPCA.mean_
            IPCA_Std = (IPCA.var_)**0.5
            IPCA_Components = IPCA.components_[:n_feature_per_shell,:]
            IPCA_Components[np.abs(IPCA_Components[:n_feature_per_shell,:]) <= 1e-5] = 0.0

        else:
            IPCA_Components = None

        if TestingPhase is False:
            print("Concating Dataset")
        pool = multiprocessing.Pool(self.n_datasetworker)
        OOC_PdbidListFeature_partial = functools.partial(OOC_Pdbid_TorchDataset, 
                                classindex_int = ClassName_ClassIndex_Dict, 
                                #Assigned_PdbidWeight  = Assigned_PdbidWeight,
                                feature_mean = IPCA_Mean,
                                feature_std = IPCA_Std,
                                feature_component = IPCA_Components,
                                feature_component_size = n_feature_per_shell,
                                feature_resizeshape = (6,80), # TODO Here assume altman
                                FetchIndexC = FetchIndexC,  
                                ReturnRawAltman = ReturnRawAltman,
                                data_stride = User_datastride,
                                LabelTreeLogic = self.TaskNameLabelLogicDict[User_Task],
                                User_NeighborLabelSmoothAngstrom = User_NeighborLabelSmoothAngstrom,
                                DIR_Typi = self.DIR_Typi)
        #st = time.time()
        ds_temp = pool.map(OOC_PdbidListFeature_partial, Assigned_PdbidBatch)


        if TestingPhase is False:
            ds_pdbidweight = [] # NOTE reciprocal of count
            ds_labels = []
            for t_x, t_y, t_yyyy, pdbid in tqdm.tqdm(ds_temp):
                if t_x is None:
                    continue
                totalsize = t_x.shape[0]
                reciprocal_sizecount_pdbid=  [Assigned_PdbidWeight[pdbid]]*totalsize
                ds_labels.extend(t_y.numpy().tolist())
                ds_pdbidweight.extend(reciprocal_sizecount_pdbid)




        # Remove the pdbid
        #ds_temp = map (lambda x:x[0],ds_temp)
        if TestingPhase is False:
            print("Finished Concat data. Cooling down")
            time.sleep(10)

        pool.close()
        pool.join()

        del OOC_PdbidListFeature_partial
        gc.collect()

        KillInactiveKernels(cpu_threshold = 0.1)
        if TestingPhase is False:
            time.sleep(10)


        ds_temp = NucleicNet.Fuel.DS3.BasicMap(
                  torch.cat([x[0] for x in ds_temp if x[0] is not None], 0), 
                  torch.cat([x[1] for x in ds_temp if x[0] is not None], 0),
                  torch.cat([x[2] for x in ds_temp if x[0] is not None], 0)
                  )

        if TestingPhase:
             return ds_temp


        ds_pdbidweight = np.array(ds_pdbidweight)
        ds_labels = np.array(ds_labels).astype(int)

        print(len(ds_pdbidweight), ds_temp.__len__())

        # =============================
        # Class label weight
        # =============================
        # NOTE Unfortunately we cannot rely on the global sampling weight. We need to recount it after splitting...
        #      count += pbdidweight
        count_class = {i:0 for i in range(len(ClassName_ClassIndex_Dict.keys()))}
        for label_i in range(len(ClassName_ClassIndex_Dict.keys())):
            index_label_i = np.where(ds_labels == label_i)[0]
            for i in index_label_i:
                count_class[label_i] += ds_pdbidweight[i]
        print(count_class)

        # NOTE This is the class label weight after reweighed by pdbid
        ds_classlabelsamplingweight = np.array([1/count_class[i] for i in range(len(ClassName_ClassIndex_Dict.keys()))])*1000 # For the sake of reading
        ds_classlabelweight = np.array([ds_classlabelsamplingweight[i] for i in ds_labels])
        ds_samplingweight = ds_classlabelweight * ds_pdbidweight
        

        return ds_temp, ds_samplingweight
                

    def GetNoOpSingleDataset(self,Assigned_PdbidBatch = [],
                        Assigned_PdbidWeight = {},
                        ClassName_ClassIndex_Dict = {"A":0, "C": 1, "G": 2, "U": 3},
                        User_Task = "AUCG",
                        User_datastride = 1,
                        User_NumReductionComponent = None,
                        PerformZscoring = True, PerformReduction = False,
                        StringentValidation = True,
                        ReturnRawAltman = False,
                        TestingPhase = True,
                        User_NeighborLabelSmoothAngstrom = 0.0,
                        ):
        FetchIndexC = FetchIndex(DIR_Feature = self.DIR_FuelInput, 
                                    DIR_Typi = self.DIR_Typi, n_row_maxhold = self.n_row_maxhold)
        #print(Assigned_PdbidWeight)
        #print(Assigned_PdbidBatch)

        assert len(Assigned_PdbidBatch) == 1, "ABORTED. Supply a list (with only one member) of pdbidbatch"

        n_feature_per_shell = 80
        IPCA = self.GetIpca(User_Task)
        if PerformZscoring:
            IPCA_Mean = IPCA.mean_
            IPCA_Std = (IPCA.var_)**0.5
        else:
            IPCA_Mean = None
            IPCA_Std = None
        
        if PerformReduction:
            assert (User_NumReductionComponent is not None), "ABORTED. Specify User_NumReductionComponent"
            n_feature_per_shell = User_NumReductionComponent
            IPCA_Mean = IPCA.mean_
            IPCA_Std = (IPCA.var_)**0.5
            IPCA_Components = IPCA.components_[:n_feature_per_shell,:]
            IPCA_Components[np.abs(IPCA_Components[:n_feature_per_shell,:]) <= 1e-5] = 0.0

        else:
            IPCA_Components = None

        
        ds_temp = OOC_Pdbid_TorchNoOpSingleDataset(Assigned_PdbidBatch[0],
                                classindex_int = ClassName_ClassIndex_Dict, 
                                #Assigned_PdbidWeight  = Assigned_PdbidWeight,
                                feature_mean = IPCA_Mean,
                                feature_std = IPCA_Std,
                                feature_component = IPCA_Components,
                                feature_component_size = n_feature_per_shell,
                                feature_resizeshape = (6,80), # TODO Here assume altman
                                FetchIndexC = FetchIndexC,  
                                ReturnRawAltman = ReturnRawAltman,
                                data_stride = User_datastride,
                                LabelTreeLogic = self.TaskNameLabelLogicDict[User_Task],
                                User_NeighborLabelSmoothAngstrom = User_NeighborLabelSmoothAngstrom,
                                DIR_Typi = self.DIR_Typi
                    )



        ds_temp = NucleicNet.Fuel.DS3.BasicMap(ds_temp[0], ds_temp[1], ds_temp[2])

        return ds_temp




# =============================
# Template Task Set up
# =============================

"""
# ===================================
# NOTE OBSOLETE Getting a constant testing fold to memory
# ====================================
# NOTE This is obsolete as we should have some faith in validation. 
#   We will delay this after the training
pool = multiprocessing.Pool(30)
OOC_PdbidListFeature_partial = functools.partial(OOC_PdbidListFeature, classindex_int = classindex_int, 
                        feature_mean = IPCA_Mean,
                        feature_std = IPCA_Std,
                        feature_resizeshape = (80,6),
                        FetchIndexC = FetchIndexC,  
                        LabelTreeLogic = TaskNameLabelLogicDict[User_Task],
                        data_stride = 2
                        )

ds_testing = pool.map(OOC_PdbidListFeature_partial, TestingFold)


pool.close()
pool.join()

ds_testing = torch.utils.data.ConcatDataset(ds_testing)
testing_culmsize = ds_testing.__len__()
print(testing_culmsize)
test_dataloader = torch.utils.data.DataLoader(ds_testing, 
                    batch_size=User_SizeMinibatch, 
                    drop_last=True, num_workers=4, 
                    pin_memory=True,worker_init_fn=None, 
                    prefetch_factor=3, persistent_workers=False)
#del ds_testing, pool
#gc.collect()

"""
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