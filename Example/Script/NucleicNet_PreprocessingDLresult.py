
"""
Preprocessing DL results
"""




import subprocess
import os
import sys
import re
import glob
from io import StringIO
from argparse import ArgumentParser
import shutil
import itertools
from functools import partial


import gc
import copy

import pandas as pd
from biopandas.pdb import PandasPdb
from collections import defaultdict
from collections import Counter

import numpy as np
from scipy import spatial
from scipy.spatial.distance import euclidean
import scipy


import urllib.request

import gzip
import pickle

import multiprocessing
from random import random
from numpy import sqrt, loadtxt, savetxt, argmax, floor, abs, max, exp, mod, zeros



from sklearn.cluster import DBSCAN
from sklearn import metrics


import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import sin, cos, acos, sqrt




from Supporting import *



parser = ArgumentParser(description="This script will dock the mer in pdb onto the target")
parser.add_argument("--TargetPdbFolder", type=str, dest="targetpdbfolder",
                    help="The Protein structure analysed")
parser.add_argument("--OutputFolder", type=str, dest="out",
                    help="Store all outputs")
parser.add_argument("--PredictionFolder", type=str, dest="predictionfolder",
                    help="folder that contains the blind pdb")
parser.add_argument("--Pdbid", type=str, dest="pdbid",
                    help="folder that contains the blind pdb")
args = parser.parse_args()

##########################################
MkdirList([args.out])



def XYZ(listofarray,label,fn):
    XYZTrial = []
    if listofarray:
        Points=listofarray
        for point in Points:
            XYZTrial.append('%s       %.5f        %.5f        %.5f\n' %(label, point[0], point[1], point[2]))
        with open("%s" %(fn),'a') as f:
            for point in XYZTrial:
                f.write(point)
    del XYZTrial


def Inspection(RemainingCoord, name):
    # Print into a single xyz file for inspection
    XYZ(RemainingCoord['R'],"Re","%s/%s.xyz" %(args.out,name))
    XYZ(RemainingCoord['P'],"P","%s/%s.xyz" %(args.out,name))
    XYZ(RemainingCoord['G'],"Ge","%s/%s.xyz" %(args.out ,name))
    XYZ(RemainingCoord['U'],"U","%s/%s.xyz" %(args.out  ,name))
    XYZ(RemainingCoord['A'],"Ar","%s/%s.xyz" %(args.out ,name))
    XYZ(RemainingCoord['C'],"Co","%s/%s.xyz" %(args.out ,name))
    XYZ(RemainingCoord['Pyr'],"Y","%s/%s.xyz" %(args.out,name))
    XYZ(RemainingCoord['Pur'],"Pu","%s/%s.xyz" %(args.out,name))

def pool_init():
    gc.collect()

class Parallel_dbscan(object):
 def __init__(self):
    print("DBscan start")
 def __call__(self, QQ):
        RemainingCoord, lab, Target_Pdb_Name = QQ
        FilteredCoord = {}
        CentroidPharmacophoreCoord = {}

        tree = spatial.KDTree(np.array(RemainingCoord[lab]))
        radius = 2.0
        LocalNeighborhood = np.array(list(map(len, tree.query_ball_tree(tree, radius))))
        PercentileNeighbour = np.percentile(LocalNeighborhood, 70.0)

        # DBScan 
        X = np.array(RemainingCoord[lab])
        db = DBSCAN(eps=2.0, min_samples=PercentileNeighbour*1.5).fit(X)
        labels = db.labels_

        # Number of clusters (ignoring noise if present.)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print("There are %s clusters in %s, each core point is surrounded by ~ %s points." %(n_clusters_, lab, int(PercentileNeighbour*1.5)))

        # Visualise the clusters and protein
        """        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax._axis3don = False
        visualizeDBPatch( X,ax,labels,lab, Target_Pdb_Name)
        plt.clf()
        """

        # Will be written into a xyz for visualisztion
        FilteredCoord[lab] = X[np.where(labels > 0)].tolist()

        # For each cluster write a centroid
        PharmaCentroidList = []
        for o in sorted(set(labels)-set([-1])):
            K = X[np.where(labels == o)]
            PharmaCentroidList.append(K.mean(axis=0))
        CentroidPharmacophoreCoord[lab] = PharmaCentroidList

        return FilteredCoord, CentroidPharmacophoreCoord




# Read Grid Prediction Result
AvailableGridPrediction = [i for i in glob.glob("%s/%s_result_reformat.pickle" %(args.predictionfolder, args.pdbid))]


# Will read and analyse each result one by one. Parallel on mer 
for targetdfdir in AvailableGridPrediction:

    # A. Preparation


    # 1. Print a Summary for each column
    name = targetdfdir.split("/")[-1].split("_")[0]
    df = pd.read_pickle("%s"%(targetdfdir))
    df['Pur'] = df['A'] + df['G']
    df['Pyr'] = df['C'] + df['U'] #+ df['T']
    print(df.describe())



    print("%s/%s.pdb" %(args.targetpdbfolder, name))


    # 2. Processing Deep Learning Probability 
    ListOfLabel = ['R','P','A','U','C','G','Pur','Pyr']
    """
    RemainingCoord = {}
    # Filtering by Dominant Label (Tend to give VERY imprecise grid points)
    df['DominantLabel'] = df[['NONSITE','R','P','A','U','C','G','Pur','Pyr']].idxmax(axis=1)
    for lab in ListOfLabel:
        RemainingCoord[lab] = [[float(g.split(":")[-3]), float(g.split(":")[-2]), float(g.split(":")[-1])] for g in df.loc[(df['DominantLabel'] == lab )]['Annotation'].values.tolist()]

    Inspection(RemainingCoord, name+"_dominantlabel")
    """
    
    # Filter away the highly improbables which distract visualisations
    RemainingCoord = {} 
    # The following parameter only affects visualisation of the binding pocket
    # For Ago: 
    #Percentile_Dict = {'R': 92.0, 'P':90.0 , 'A': 98.0, 'G':98.0 , 'U':98.0, 'C':98.0, 'Pur': 98.0, 'Pyr': 98.0}
    # For others:
    #Percentile_Dict = {'R': 92.0, 'P':90.0 , 'A': 98.0, 'G':96.0 , 'U':97.5, 'C':98.0, 'Pur': 98.0, 'Pyr': 98.0}
    # OR
    #Percentile_Dict = {'R': 92.0, 'P':90.0 , 'A': 98.0, 'G':98.0 , 'U':99.0, 'C':98.0, 'Pur': 98.0, 'Pyr': 98.0}
    # Parameter below only affects visualisation
    if name in ['4f3t'] :
       Percentile_Dict = {'R': 92.0, 'P':90.0 , 'A': 98.0, 'G':98.0 , 'U':98.0, 'C':98.0, 'Pur': 98.0, 'Pyr': 98.0}
    else:
       Percentile_Dict = {'R': 92.0, 'P':90.0 , 'A': 98.0, 'G':95.5 , 'U':97.5, 'C':98.0, 'Pur': 98.0, 'Pyr': 98.0}
       #Percentile_Dict = {'R': 92.0, 'P':90.0 , 'A': 98.0, 'G':98.0 , 'U':99.0, 'C':98.0, 'Pur': 98.0, 'Pyr': 98.0}
    for lab in ListOfLabel:
     BootstrapIndex = []
     for BootstrapTrial in range(500):
        # Filter by chance (i.e. grids show up by chance indicated by the predicted probability)
        df_temp = df[lab] - np.random.rand(*df[lab].shape)
        # Record Those remained after filtering
        Reamining_df_index = (df_temp > 0.0 )
        BootstrapIndex.extend(np.where(Reamining_df_index.tolist())[0].tolist())
     count = Counter(BootstrapIndex)
     percentile = np.percentile(list(count.values()), Percentile_Dict[lab])
     selected_index = []
     for k, v in count.items():
         if v >= percentile:
             selected_index.append(int(k))
     RemainingCoord[lab] = [[float(g.split(":")[-3]), float(g.split(":")[-2]), float(g.split(":")[-1])] for g in df.loc[np.array(selected_index)]['Annotation'].values.tolist()]


    # 3. Parallel DBscan 
    # This removes the sparse points if there are any
    pool=multiprocessing.Pool(processes=4, maxtasksperchild=10000)
    Target_Pdb_Name = "%s/%s.pdb" %(args.targetpdbfolder, name)
    zipped =  [(RemainingCoord,lab, Target_Pdb_Name) for lab in ListOfLabel]
    results = pool.map(Parallel_dbscan(), zipped)
    FilteredCoord = {}
    for d in results:
        FilteredCoord.update(d[0])
    CentroidPharmacophoreCoord = {}
    for d in results:
        CentroidPharmacophoreCoord.update(d[1])


    # 4. Write into a xyz for visualisation
    Inspection(FilteredCoord, name+"_strong_Bootstrap")
    Inspection(CentroidPharmacophoreCoord, name+"_dbscan_centroid_Bootstrap")
    XYZ([[float(g.split(":")[-3]), float(g.split(":")[-2]), float(g.split(":")[-1])] for g in df['Annotation'].values.tolist()],"Xe","%s/%s.xyz" %(args.out,'%s_all_grids'%(name)))    
    pickle.dump(FilteredCoord, open("%s/%s_DBScanBootstrapGrid.dict"%(args.out,name) ,"wb"))









    







