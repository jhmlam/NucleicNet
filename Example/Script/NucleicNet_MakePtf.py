import subprocess
import os
import sys
import re
import glob
from io import StringIO
from argparse import ArgumentParser
import shutil
import itertools

import gc
import copy

import pandas as pd
from biopandas.pdb import PandasPdb
from collections import defaultdict

import numpy as np
from scipy import spatial
import random
from scipy.spatial.distance import euclidean



import urllib.request

import gzip
import pickle

import multiprocessing

from Supporting import *

parser = ArgumentParser(description="This script will define sites as Ptf files for each pdb")
parser.add_argument("--Resolution", type=float, dest="resolution",
                    help="A dataframe created in Database-Pdb by merging avaialble information around Pdbids")
parser.add_argument("--Distance", type=float, dest="distance",
                    help="A distance cutoff for making of training data set")
parser.add_argument("--VoronoiCellTruncation", type=float, dest="truncation",
                    help="Truncation of Voronoi Cells")
parser.add_argument("--MidlineHalo", type=str, dest="halo",
                    help="Upper and lower radius limit for Halo of Midline")
parser.add_argument("--AwayPocketNucleic", type=float, dest="away",
                    help="the distance where non site grid point should be away from pocket and nucleic points.")
parser.add_argument("--BlindPdbFolder", type=str, dest="blindfolder",
                    help="folder that contains the blind pdb")


args = parser.parse_args()

##########################################
halo = [float(i) for i in args.halo.split(",")]

#MkdirList([args.surface, args.pocket])

# Grab pdbs with NA
#DfGrandNA = pickle.load(open("%s"%(args.granddfdirNA),"rb"))
#DfGrandNA = pd.read_pickle("%s"%(args.granddfdirNA))
#ListOfProNucComplexes = DfGrandNA["Pdbid"].tolist()
#print(len(ListOfProNucComplexes))




AtDict = {"A":"C4","T":"C4","C":"C4","G":"C4","U":"C4","R":"C1\'","D":"C1\'","P":"P"}

def ReturnCentroid(pdbid, df, proteintree, label, distance):
    Centroid = []
    if not df.empty:
      zipped=sorted(set([(resid, chainid) for resid,chainid in zip(df["residue_number"].tolist(), df["chain_id"].tolist())]))
      for resid,chainid in zipped:
        x=df.loc[(df["residue_number"] == int(resid)) & (df["chain_id"] == str(chainid))]["x_coord"].tolist()
        y=df.loc[(df["residue_number"] == int(resid)) & (df["chain_id"] == str(chainid))]["y_coord"].tolist()
        z=df.loc[(df["residue_number"] == int(resid)) & (df["chain_id"] == str(chainid))]["z_coord"].tolist()
        centroid=[sum(x)/len(x), sum(y)/len(y), sum(z)/len(z)]
        # 7.5 is the default radius for FEATURE vector
        if proteintree.query(centroid,1)[0] < distance:
          Centroid.append([centroid[0], centroid[1], centroid[2], str(label),resid, chainid, AtDict[label]])
    else:
      pass
    return Centroid


# For visualisation in pymol for points e.g. site and non sites
def XYZ(listofarray,label,fn):
    XYZTrial = []
    if listofarray:
        Points=sorted(listofarray)
        for point in Points:
            XYZTrial.append('%s       %.5f        %.5f        %.5f\n' %(label, point[0], point[1], point[2]))
        with open("%s" %(fn),'w+') as f:
            for point in XYZTrial:
                f.write(point)
    del XYZTrial





def pool_init():
    gc.collect()

class MakePtf(object):
 def __init__(self):
    print("Starting To Generate Blind Grid")
 def __call__(self, pdbid):

    print(pdbid)
    # A. Pdb reading
    # Reading in the pdb for the current conformation
    ppdb = PandasPdb()
    CurrentPdbStructure = ppdb.read_pdb("%s/%s.pdb" %(args.blindfolder, str(pdbid)))

    proteindf = CurrentPdbStructure.df['ATOM'][~CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U","DA","DT","DC","DG","DU"])]
    proteinpoint =  np.array([proteindf["x_coord"].tolist(),proteindf["y_coord"].tolist(),proteindf["z_coord"].tolist()]).T
    proteintree = spatial.cKDTree(proteinpoint)



    # B. Grid Creation
    # Define Dimension of Grid box
    maxx=max(CurrentPdbStructure.df['ATOM']["x_coord"].tolist())+5
    minx=min(CurrentPdbStructure.df['ATOM']["x_coord"].tolist())-5
    maxy=max(CurrentPdbStructure.df['ATOM']["y_coord"].tolist())+5
    miny=min(CurrentPdbStructure.df['ATOM']["y_coord"].tolist())-5
    maxz=max(CurrentPdbStructure.df['ATOM']["z_coord"].tolist())+5
    minz=min(CurrentPdbStructure.df['ATOM']["z_coord"].tolist())-5



    # Surface Grid Points
    points = np.mgrid[minx:maxx, miny:maxy, minz:maxz]
    points = np.matrix(points.reshape(3, -1).T)
    tree = spatial.cKDTree(points)
    # Index of points within cutoff
    pointswithincutoff1 = set(itertools.chain.from_iterable(list(tree.query_ball_point(proteinpoint, halo[0]))))
    # Index of points within cutoff
    pointswithincutoff2 = set(itertools.chain.from_iterable(list(tree.query_ball_point(proteinpoint, halo[1]))))
    # Surface points within the midline and finalise tree for Surface accordingly
    midlineindex = sorted(pointswithincutoff2 - pointswithincutoff1)
    
    print (pdbid,len(points),len(midlineindex))

    print (points[1].tolist()[0][0])



    with open("%s/%s_Grid.ptf"%(args.blindfolder, pdbid),'w+') as f:
        for i in midlineindex:
          f.write('%s\t%.3f\t%.3f\t%.3f\t#\t%s000:X@XX:grid\n' %(pdbid, points[i].tolist()[0][0], points[i].tolist()[0][1], points[i].tolist()[0][2], str("X")))
    #print(points[np.array(midlineindex)].tolist())

    XYZ(points[np.array(midlineindex)].tolist(),"Ge","%s/%s_Grid.xyz" %(args.blindfolder,pdbid))






ListBlindTest = [i.split('/')[-1].split(".")[0] for i in glob.glob("%s/*.pdb"%(args.blindfolder))]
#print(ListBlindTest)
pool=multiprocessing.Pool(processes=12, initializer=pool_init, maxtasksperchild=10000)
results = pool.map(MakePtf(), ListBlindTest)
pool.close
gc.collect()


