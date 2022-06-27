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

import sys
import time
import glob
import tqdm
import pickle
import pandas as pd
import numpy as np
import scipy
from scipy import sparse

import seaborn as sns
import matplotlib.pyplot as plt
import random




#import dask
import linecache
import shutil
import tempfile
import multiprocessing
from functools import partial
import gc
from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandHalo import Halo



import collections



# =====================
# Some simple util
# =====================
# TODO OBSOLETE Move to DataFetcher
def FindContainer_ByClass(typi, classindex = 10, n_row_maxhold = 10000):
    index_ = np.sort(scipy.sparse.find(typi[:,classindex])[0])
    index_head = ((index_ / n_row_maxhold).astype(int) * n_row_maxhold).tolist()
    index_localindex = (index_ % n_row_maxhold).tolist()
    # Group by file head 
    headcontain = collections.defaultdict(list)
    for i in range(len(index_head)):
        headcontain[index_head[i]].append(index_localindex[i])
    return headcontain, index_


# TODO OBSOLETE Move to DataFetcher
def FindFeature_ByContainer(containerdict, 
                                DIR_Feature = "../Database-PDB/feature/",
                                pdbid = "", featuretype = "altman", returndense = False):
        featurevectors = []
        for headindex in sorted(containerdict.keys()):
                localindex = containerdict[headindex]
                fn = DIR_Feature + pdbid + ".%s.%s.npz" %(featuretype, headindex)
                with np.load(fn, 
                                mmap_mode='r', allow_pickle=False) as f:
                        ff = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])   
                        if returndense:
                            featurevectors.append(ff[localindex, :].todense())
                        else:
                            featurevectors.append(ff[localindex, :])
        if len(featurevectors) == 0:
            return None
            
        if returndense:
            return np.concatenate(featurevectors, axis=0)
        else:
            return sparse.vstack(featurevectors)

# =========================
# Altman
# ==========================

def OOC_MakeDssp(f, DsspFolder = "", DsspExecutable = "../NucleicNet/util/dssp"):
    subprocess.call("%s -i %s -o %s/%s.dssp" %(
        DsspExecutable,
        f,
        DsspFolder,
        f.split("/")[-1].split(".")[0][:4] # NOTE THe feature program only recognize first 4 charcters....
        ), shell = True)

    return

def OOC_Altman_MakePtf(tmpdirname, pdbid, halo_coord):
    with open("%s/ptf/%s.ptf"%(tmpdirname, pdbid),'w+') as f:
        for i in range(halo_coord.shape[0]):
            f.write('%s\t%.3f\t%.3f\t%.3f\t#\t%s000:X@XX:xxxx\n' 
                    %(pdbid,halo_coord[i][0], 
                            halo_coord[i][1],
                            halo_coord[i][2],
                    str("X")))

def OOC_Altman_MakeFF(DIR_InputPdbFolder, tmpdirname, DIR_AltmanFolder, n_AltmanShell, pdbid):
    subprocess.call(
        # TODO Possibly we need to find the pid and pkill it to avoid memory leak?
        "export PDB_DIR=%s ; export DSSP_DIR=%s ; export FEATURE_DIR=%s ; %s/featurize -n %s -P %s/ptf/%s.ptf -s %s > %s/ff/%s.ff; sleep 2"
                    %(os.path.abspath(DIR_InputPdbFolder),
                    os.path.abspath("%s/dssp/" %(tmpdirname)),
                    os.path.abspath(DIR_AltmanFolder) + "/data/",
                    os.path.abspath(DIR_AltmanFolder) + "/src",
                    n_AltmanShell,
                    os.path.abspath(tmpdirname),
                    pdbid,
                    os.path.abspath(DIR_InputPdbFolder)+ "/%s.pdb" %(pdbid), 
                    os.path.abspath(tmpdirname), 
                    pdbid), shell=True)

def OOC_Altman_ProcessFF(ff_filename, outputfolder = "../Database-PDB/trial/", 
n_altmanfeature_total = 640, feature_excluded = list(range(160)), feature_setzero = [0,3,45], n_row_maxhold = 10000):
    
    pdbid = ff_filename.split("/")[-1].split(".")[0]
    feature_included = [i for i in range(n_altmanfeature_total) if i not in feature_excluded]
    n_line_total = sum(1 for line in open(ff_filename))

    # Batched sparse matrices
    SparseClassMatrix = scipy.sparse.lil_matrix((n_row_maxhold,n_altmanfeature_total), dtype=np.float32, copy=False)    

    ever_encountered_rows = 0
    start_index = 0
    local_row_index = 0

    # TODO Make it a for loop
    for line_i in range(n_line_total):   
            linetext = linecache.getline(ff_filename, line_i+1)

            if linetext.startswith("#"):
                continue

            ll = np.array(linetext.split("\t")[1:-6], dtype = np.float32)

            index = np.where(ll != 0.0)[0]
            SparseClassMatrix[local_row_index,index] = ll[index]
            local_row_index += 1
            ever_encountered_rows += 1

            if (((ever_encountered_rows % n_row_maxhold) == 0 ) & (ever_encountered_rows != 0)) | (line_i == n_line_total-1):

                # Just in case there are nan
                SparseClassMatrix  = np.nan_to_num(SparseClassMatrix, copy=True, nan=0.0, posinf=None, neginf=None)

                # To csr
                SparseClassMatrix = SparseClassMatrix.tocsr(copy = False)

                # Set certain column zero
                SparseClassMatrix[:,feature_setzero] = 0.0

                # Filter the column
                SparseClassMatrix = SparseClassMatrix[:,feature_included]

                # Cut cookie in shape
                if (line_i == n_line_total-1):
                    SparseClassMatrix = SparseClassMatrix[:local_row_index,:]

                # Save matrix 
                # NOTE Float 16 vs float 32 is around 1.7 MB vs 2.2 MB after compression.
                #      Compression is necessary as uncompressed data are 11 MB vs 2.1MB
                scipy.sparse.save_npz(outputfolder+'/%s.altman.%s.npz' %(pdbid, start_index),
                            SparseClassMatrix,compressed = True)

                # NOTE I try to make this safer to prevent storage fault when it write alot of files in parallel
                time.sleep(random.uniform(0.5, 1.5))

                start_index += n_row_maxhold

                # Reset matrix
                local_row_index = 0
                SparseClassMatrix = scipy.sparse.lil_matrix((n_row_maxhold,n_altmanfeature_total), dtype=np.float32, copy=False)
                gc.collect()
            #del linetext
            #gc.collect()
    
    del SparseClassMatrix
    gc.collect()


def OOC_Altman(pdbfn, DIR_InputPdbFolder= "", DIR_InputHaloFolder = "", 
                        DIR_DsspExecutable = "", DIR_AltmanFolder = "", 
                        DIR_OutputFeatureVector = "", 
                        n_AltmanShell = 8, n_altmanfeature_total = 640, 
                        feature_remove = list(range(160)), feature_setzero = [0,3,45],
                        n_row_maxhold = 10000):

    pdbid = pdbfn.split("/")[-1].split(".")[0]

    # Get halo coord. TODO Replace this with mesh and start with halo coord
    with open(DIR_InputHaloFolder + "/%s.halotup"%(pdbid), "rb") as fn:
        halo_tuple = pickle.load(fn)
    HaloC = Halo(CallMkdirList = False)
    halo_coord = HaloC.RetrieveHaloCoords(halo_tuple)

    # Make ptf,ff,dssp
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Separate the output to prevent messing up with FEATURE
        MkdirList([ "%s/ptf" %(tmpdirname), 
                    "%s/dssp" %(tmpdirname),
                    "%s/ff" %(tmpdirname)])

        # Make dssp
        OOC_MakeDssp(DIR_InputPdbFolder + "/%s.pdb" %(pdbid), DsspFolder="%s/dssp/" %(tmpdirname), DsspExecutable=DIR_DsspExecutable)

        # Make Ptf
        OOC_Altman_MakePtf(tmpdirname, pdbid, halo_coord)

        # Make ff
        OOC_Altman_MakeFF(DIR_InputPdbFolder, tmpdirname, DIR_AltmanFolder, n_AltmanShell, pdbid)

        # Process the output file of altman's FEATURE
        OOC_Altman_ProcessFF("%s/ff/%s.ff"%(os.path.abspath(tmpdirname), 
                                    pdbid), 
                        outputfolder = DIR_OutputFeatureVector, 
                        n_altmanfeature_total = n_altmanfeature_total, feature_excluded = feature_remove, 
                        feature_setzero = feature_setzero, n_row_maxhold = n_row_maxhold)

        # NOTE The with statement does not seem to erase from memory.
        #print(os.path.abspath(tmpdirname)+ "/ff/%s.ff" %(pdbid))
        try:
            os.remove(os.path.abspath(tmpdirname)+ "/ff/%s.ff" %(pdbid))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.abspath(tmpdirname)+"/dssp/%s.dssp" %(pdbid))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.abspath(tmpdirname)+"/ptf/%s.ptf" %(pdbid))
        except FileNotFoundError:
            pass
        #shutil.rmtree("%s/"%(os.path.abspath(tmpdirname)), ignore_errors=True, onerror=None)

    del HaloC, halo_coord, halo_tuple
    gc.collect()
    return

def OOC_AltmanProperties():
    return


# ===============================
# Body
# ===============================


class Featurisation:
    def __init__(self, 
        DIR_OutputFeatureVector = "/home/homingla/Project-NucleicNet/Database-PDB/feature",
        DIR_InputPdbFolder = "/home/homingla/Project-NucleicNet/Database-PDB/apo", 
        DIR_InputHaloFolder = "/home/homingla/Project-NucleicNet/Database-PDB/halo", 
        DIR_DsspExecutable = "../NucleicNet/util/dssp",
        DIR_AltmanFolder = "../NucleicNet/util/feature-3.1.0/",
        HaloLowerBound = 2.5,
        HaloUpperBound = 5.5,
        n_multiprocessingworkers = 8,
        n_row_maxhold = 10000,
        UpdateExist = False,SavingFeature = True):

        self.DIR_InputPdbFolder = DIR_InputPdbFolder
        self.DIR_OutputFeatureVector = DIR_OutputFeatureVector
        self.DIR_InputHaloFolder = DIR_InputHaloFolder
        self.DIR_DsspExecutable = DIR_DsspExecutable
        self.DIR_AltmanFolder = DIR_AltmanFolder
        self.UpdateExist = UpdateExist
        self.SavingFeature = SavingFeature
        self.HaloLowerBound = HaloLowerBound
        self.HaloUpperBound = HaloUpperBound
        self.n_multiprocessingworkers = n_multiprocessingworkers

        self.n_row_maxhold = n_row_maxhold
        MkdirList([self.DIR_OutputFeatureVector])

    def Altman(self, RemoveInnerShells = True):

        DIR_DsspExecutable = self.DIR_DsspExecutable 
        DIR_AltmanFolder = self.DIR_AltmanFolder 
        DIR_DsspExecutable = self.DIR_DsspExecutable
        DIR_InputPdbFolder = self.DIR_InputPdbFolder
        DIR_InputHaloFolder = self.DIR_InputHaloFolder
        DIR_OutputFeatureVector = self.DIR_OutputFeatureVector


        # ==========================================
        # Prepare Altman feature interpretation
        # ==========================================
        PART0_CollectFeatureMeaning = True
        if PART0_CollectFeatureMeaning:

            AltmanShell_thickness = 1.25
            n_InnerShells_removed = int(self.HaloLowerBound/1.25)
            n_AltmanShell = 6+n_InnerShells_removed

            # A. Prepare column names for dataframe
            # Read in the property name
            altman_propertiesremoved = ["RESIDUE_NAME_IS_HOH", 
                                        "RESIDUE_NAME_IS_OTHER",
                                        "RESIDUE_CLASS1_IS_UNKNOWN",
                                        "RESIDUE_CLASS2_IS_UNKNOWN", 
                                        "MOBILITY",
                                        "ATOM_TYPE_IS_OTHER", # NOTE Removed in 3.1.0. replaced as ELEMENT_IS_OTHER
                                        "ELEMENT_IS_OTHER",
                                        "ATOM_TYPE_IS_Na", # NOTE Small case. Removed in 3.1.0. 
                                        "ATOM_TYPE_IS_Ca"] 
            altman_propnamelist=[]
            propfile = open("%s/data/%s"%(DIR_AltmanFolder,"proteins.properties"), 'r')
            propdata = propfile.read().splitlines()
            for line in propdata:
                if line:
                    if not line.startswith("#"):
                        altman_propnamelist.append(str(line))
            propfile.close()

            # This is the list of all possible column names
            ColumnnameToIndex = {}
            i=0
            for shell in range(n_AltmanShell): # NOTE The default of altman feature has 6 shells
                for prop in altman_propnamelist:
                        #columnlist.append(str(shell)+str(prop))
                        ColumnnameToIndex[str(shell)+str(prop)] = i
                        i+=1
            n_altmanfeature_total = i
            #print(ColumnnameToIndex)

            # Set zero for unused features
            feature_setzero = []
            for i in range(n_AltmanShell):
                for j in altman_propertiesremoved:
                    if "%s%s" %(i,j) in ColumnnameToIndex.keys():
                        feature_setzero.append(ColumnnameToIndex["%s%s" %(i,j)])

            # Remove frontal shells
            if RemoveInnerShells:
                feature_remove = []
                for i in range(n_InnerShells_removed):
                    for j in altman_propnamelist:
                        feature_remove.append(ColumnnameToIndex["%s%s" %(i,j)])

            else:
                # Keep at 6 shells
                feature_remove = []
                for i in range(n_AltmanShell):
                    if i <= 5:
                        continue
                    for j in altman_propnamelist:
                        feature_remove.append(ColumnnameToIndex["%s%s" %(i,j)])                




        # =================================================
        # Look for available/remaining structure files
        # ==================================================
        # TODO It is assumed both halo and apo pdb exist
        pdbavail = sorted(glob.glob( self.DIR_InputHaloFolder + "/*.halotup"))
        print("There are %s structure PDB files ready for analysis" %(len(pdbavail)))

        if self.UpdateExist:
            pass
        else:
            pdbavail_done = []
            for i in pdbavail:
                pdbid = i.split("/")[-1].split(".")[0]
                if os.path.exists(DIR_OutputFeatureVector + "/%s.altman.0.npz" %(pdbid)):
                    pdbavail_done.append(i)
            pdbavail = [i for i in pdbavail if i not in pdbavail_done] 


        filesize = [os.path.getsize(i) for i in pdbavail]
        pdbavail = [pdbavail[i] for i in np.argsort(filesize).tolist()]#[::-1]
        pdbavail_allowmultiprocessing = pdbavail
        #print(pdbavail_allowmultiprocessing)
        print("There are %s structure PDB files remained to be analysed" %(len(pdbavail_allowmultiprocessing)))


        # ======================================
        # Multiprocessing for Altman feature
        # =======================================
        print("Start Multiprocessing")
        pdbavail_allowmultiprocessing_chunks = ChunkList(pdbavail_allowmultiprocessing,self.n_multiprocessingworkers*2 )
        for chunk in tqdm.tqdm(pdbavail_allowmultiprocessing_chunks):
            pool = multiprocessing.Pool(self.n_multiprocessingworkers)
            partial_OOC_Altman = partial(OOC_Altman, DIR_InputPdbFolder= DIR_InputPdbFolder, DIR_InputHaloFolder = DIR_InputHaloFolder, 
                        DIR_DsspExecutable = DIR_DsspExecutable, DIR_AltmanFolder = DIR_AltmanFolder, 
                        DIR_OutputFeatureVector = DIR_OutputFeatureVector, 
                        n_AltmanShell = n_AltmanShell, n_altmanfeature_total = n_altmanfeature_total, 
                        feature_remove = feature_remove, feature_setzero = feature_setzero,
                        n_row_maxhold = self.n_row_maxhold)
            pool.map(partial_OOC_Altman, chunk)
            pool.close()
            del partial_OOC_Altman
            gc.collect()

            KillInactiveKernels(cpu_threshold = 0.1)


        
        
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