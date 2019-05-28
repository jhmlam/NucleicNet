
"""
Produce Sequence Logo for the Known Sites.
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
from scipy.spatial import cKDTree



import gzip
import pickle

import multiprocessing
from random import random
from numpy import sqrt, loadtxt, savetxt, argmax, floor, abs, max, exp, mod, zeros



import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib import transforms
from math import sin, cos, acos, sqrt, log




#from Supporting_GeometricFittings import *
from Supporting import *

from biopandas.pdb import PandasPdb


parser = ArgumentParser(description="This script will dock the mer in pdb onto the target")
#parser.add_argument("--RNAMerFolder", type=str, dest="merfolder",
#                    help="Records of Mers within some Pdb")
#parser.add_argument("--PdbFolder", type=str, dest="pdbfolder",
#                    help="source of Mer coordinates")
#parser.add_argument("--TargetPdbFolder", type=str, dest="targetpdbfolder",
#                    help="The Apo Proteins where docking will happen on its surface")
parser.add_argument("--OutputFolder", type=str, dest="out",
                    help="Store all outputs")
#parser.add_argument("--GrandDataframeNA", type=str, dest="granddfdirNA",
#                    help="A dataframe created in Database-Pdb by merging avaialble information around Pdbids")
#parser.add_argument("--Resolution", type=float, dest="resolution",
#                    help="A dataframe created in Database-Pdb by merging avaialble information around Pdbids")
#parser.add_argument("--Tolerance", type=float, dest="tolerance",
#                    help="Acceptable range +- tolerance for pharmacophore distance")
parser.add_argument("--PredictionFolder", type=str, dest="predictionfolder",
                    help="folder that contains the blind pdb")
#parser.add_argument("--MinimalClique", type=int, dest="minclique",
#                    help="This is the minimal size of clique considered")
parser.add_argument("--Pdbid", type=str, dest="pdbid",
                    help="filename in study")
#parser.add_argument("--MaxDistance", type=float, dest="maxdistance",
#                    help="Maximum distance in the graph accepting as an edge")
parser.add_argument("--KnownSitePDBFolder", type=str, dest="knownsitefolder",
                    help="the folder that holds pdb files with known binding site (RNA-loaded)")

args = parser.parse_args()

##########################################
MkdirList([args.out])


def pool_init():
    gc.collect()



# Read Grid Prediction Result
AvailableGridPrediction = [i for i in glob.glob("%s/%s_result_reformat.pickle" %(args.predictionfolder, args.pdbid))]
#AvailableGridPrediction = [i for i in glob.glob("%s/*_result_reformat.pickle" %(args.predictionfolder))]


# ============================
# Configure Matplotlib 
# ============================
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

fp = FontProperties(family="sans-serif", weight="bold") 
globscale = 1.35
LETTERS = { "U" : TextPath((-0.365, 0), "U", size=1, prop=fp),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.325, 0), "C", size=1, prop=fp) }
# RNAC color scheme
COLOR_SCHEME = {'G': '#FFAF00', 
                'A': '#06C806', 
                'C': '#0003C6', 
                'U': '#C90101'}

# Our NucleicNet Color scheme
#COLOR_SCHEME = {'G': '#AEF3EB', 
#                'A': '#5B9AE8', 
#                'C': '#E30F38', 
#                'U': '#F1BBC1'}

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

print(AvailableGridPrediction)






# =============================
# Main
# =============================

for targetdfdir in AvailableGridPrediction:

    print(targetdfdir)
    # =========================
    # Preparation
    # =========================

    # 1. Read in the predicted Grid point
    name = targetdfdir.split("/")[-1].split("_")[0]
    df = pd.read_pickle("%s"%(targetdfdir))
    df['Pur'] = df['A'] + df['G']
    df['Pyr'] = df['C'] + df['U'] #+ df['T']

    # Dominant Label
    df['DominantBase'] = df[['A','U', 'C', 'G']].idxmax(axis=1)


    # Grid Coordinate tree
    grid_coord = []
    for i in df['Annotation'].tolist():
        splitted = i.split(":")
        x = float(splitted[-3])
        y = float(splitted[-2])
        z = float(splitted[-1])
        grid_coord.append([x,y,z])
    grid_coord = np.array(grid_coord)
    grid_tree = cKDTree(grid_coord)

    print("%s.pdb" %( name))


    # 2. Read in the Control (RNA loaded pdb file)
    ppdb= PandasPdb()
    CurrentPdbStructure = ppdb.read_pdb("%s/%s.pdb" %(args.knownsitefolder, args.pdbid))

    # Protein tree
    proteinpoint  = np.array([CurrentPdbStructure.df['ATOM'][~CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U"])]["x_coord"].tolist(),CurrentPdbStructure.df['ATOM'][~CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U"])]["y_coord"].tolist(),CurrentPdbStructure.df['ATOM'][~CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U"])]["z_coord"].tolist()]).T
    proteintree = cKDTree(proteinpoint)


    # Find RNA residues
    Nucleic_df = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U"])]
    NucleotideChainResidTupleList = sorted(set(zip(Nucleic_df["chain_id"].tolist(), Nucleic_df["residue_number"].tolist())))


    


    # =================================
    # Calculate sequence logo
    # =================================
    Probability_dict_list =  []
    Logo_dict_list = []
    Logo_dict_list_4 = []
    for chainres in NucleotideChainResidTupleList:

      # Calculate centroid for the base
      Nucleotide = CurrentPdbStructure.df['ATOM'].loc[(CurrentPdbStructure.df['ATOM']["chain_id"] == chainres[0]) & (CurrentPdbStructure.df['ATOM']["residue_number"] == chainres[1])]
      Nucleotide = Nucleotide.loc[CurrentPdbStructure.df['ATOM']["atom_name"].isin(CentroidAttributeDict[sorted(set(Nucleotide["residue_name"]))[0]])]
      x=Nucleotide["x_coord"].tolist()
      y=Nucleotide["y_coord"].tolist()
      z=Nucleotide["z_coord"].tolist()
      centroid=np.array([sum(x)/len(x), sum(y)/len(y), sum(z)/len(z)])

      # Reject RNA residue that is not close to protein
      # Usual 6.5; Ago 5.0
      if args.pdbid == '4f3t':
         protein_lim = 5.0
      else:
         protein_lim = 6.5

      if proteintree.query(centroid,1)[0] > protein_lim:
          continue

      # Find grid point within 3.0 of the centroid
      grid_within_idx = grid_tree.query_ball_point(centroid, 3.0)

      # Calculate density within the grid point
      grid_within = df.iloc[grid_within_idx]

      # Count the dominant label (+0.01 such that denominator is not zero)
      U_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'U']) + 0.01
      A_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'A']) + 0.01
      C_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'C']) + 0.01
      G_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'G']) + 0.01
      P_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'P']) + 0.01
      R_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'R']) + 0.01 
      #X_Sum = sum([1 for i in grid_within['DominantBase'].tolist() if i == 'X']) + 0.01

      """
      # Calculate Averaged grid point density
      U_Sum = sum(grid_within["U"].tolist())
      A_Sum = sum(grid_within["A"].tolist())
      C_Sum = sum(grid_within["C"].tolist())
      G_Sum = sum(grid_within["G"].tolist())
      P_Sum = sum(grid_within["P"].tolist())
      R_Sum = sum(grid_within["R"].tolist())
      #Pur_Sum = sum(grid_within["Pur"].tolist())
      #Pyr_Sum = sum(grid_within["Pyr"].tolist())
      """
      Denom_Sum = sum([U_Sum, A_Sum, C_Sum, G_Sum, P_Sum, R_Sum])#, X_Sum])

      if Denom_Sum > 0.0:
         U_Sum /= Denom_Sum
         A_Sum /= Denom_Sum
         C_Sum /= Denom_Sum
         G_Sum /= Denom_Sum
         P_Sum /= Denom_Sum
         R_Sum /= Denom_Sum
      
      # logo height 6
      height6 = log(6,2)
      for i in [U_Sum, A_Sum, C_Sum, G_Sum, P_Sum, R_Sum]:#, X_Sum]:
        height6 += i*log(i,2)
      
      Logo_dict_list.append( {'ChainID':chainres[0], 'ResidueID':chainres[1] , 'U':height6*U_Sum,'C':height6*C_Sum,'A':height6*A_Sum,'G': height6*G_Sum, 'P': height6*P_Sum, 'R':height6*R_Sum})
      Probability_dict_list.append({'ChainID':chainres[0], 'ResidueID':chainres[1] , 'U':U_Sum,'C':C_Sum,'A':A_Sum,'G': G_Sum, 'P': P_Sum, 'R':R_Sum})


      # logo height 4
      U_Sum_4 = U_Sum * Denom_Sum
      A_Sum_4 = A_Sum * Denom_Sum
      C_Sum_4 = C_Sum * Denom_Sum
      G_Sum_4 = G_Sum * Denom_Sum

      Denom_Sum_4 = sum([U_Sum_4, A_Sum_4, C_Sum_4, G_Sum_4])
      if Denom_Sum_4 > 0.0:
         U_Sum_4 /= Denom_Sum_4
         A_Sum_4 /= Denom_Sum_4
         C_Sum_4 /= Denom_Sum_4
         G_Sum_4 /= Denom_Sum_4


      height4 = log(4,2)
      for i in [U_Sum_4, A_Sum_4, C_Sum_4, G_Sum_4]:
        height4 += i*log(i,2)

      # Logo score.
      Logo_dict_list_4.append( {'ChainID':chainres[0], 'ResidueID':chainres[1] , 'U':height4*U_Sum_4,'C':height4*C_Sum_4,'A':height4*A_Sum_4,'G': height4*G_Sum_4})


    # =======================
    # Save logo score
    # =======================
    performancedf = pd.DataFrame(columns = ['ChainID','ResidueID', 'U','C','A','G', 'P', 'R'])
    for i in Logo_dict_list:
         performancedf = performancedf.append(i, ignore_index=True)
    performancedf['Pyr'] = performancedf['C'] + performancedf['U']
    performancedf['Pur'] = performancedf['A'] + performancedf['G']
    performancedf['Y-R_specificity'] = performancedf['Pyr'] - performancedf['Pur']

    Probabilitydf = pd.DataFrame(columns = ['ChainID','ResidueID', 'U','C','A','G', 'P', 'R'])
    for j in Probability_dict_list:
        Probabilitydf = Probabilitydf.append(j, ignore_index = True)

    #Save logo score and prob
    #pickle.dump(performancedf, open('%s/%s_logoscore.df'%(args.out, args.pdbid),'wb'))
    #pickle.dump(Probabilitydf, open('%s/%s_knownsiteprobability.df'%(args.out, args.pdbid),'wb'))
    

    # ========================
    # Plot Logo
    # ========================
    del performancedf
    performancedf = pd.DataFrame(columns = ['ChainID','ResidueID', 'U','C','A','G', 'P', 'R'])
    for i in Logo_dict_list_4:
         performancedf = performancedf.append(i, ignore_index=True)
    performancedf['Pyr'] = performancedf['C'] + performancedf['U']
    performancedf['Pur'] = performancedf['A'] + performancedf['G']
    performancedf['Y-R_specificity'] = performancedf['Pyr'] - performancedf['Pur'] 


    for chain in sorted(set(performancedf["ChainID"].tolist())):
      subperformancedf = performancedf.loc[performancedf['ChainID'] == chain]
      #fig, ax = plt.subplots(figsize = (len(list(subperformancedf.iterrows())),3))


      #fig.set_size_inches(len(list(subperformancedf.iterrows())),3, forward=True)
      #fig, ax = plt.figure(figsize=(len(list(subperformancedf.iterrows())),3)) 
      maxy = 0.0
      minx = sorted(subperformancedf['ResidueID'].tolist())[0]
      maxx = sorted(subperformancedf['ResidueID'].tolist())[-1]
      fig = plt.figure(figsize = (len(range(minx,maxx))+2.4,6))

      print(len(range(minx,maxx))+1)
      ax = fig.add_subplot(111)
      for index,row in subperformancedf.iterrows():
        y = 0.0
        for component in ['A','G','U','C']:#,'P','R']:
            letterAt(component, row['ResidueID'],y, row['%s' %(component)], ax)
            y += row['%s' %(component)]
        maxy = np.maximum(float(maxy), float(y))

      plt.xticks(range(minx,maxx + 1))
      plt.xlim((minx-1, maxx + 1)) 
      plt.ylim((0, np.minimum(maxy + 0.1, log(6,2))))
      plt.tight_layout()      
      ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: "{:.1f}".format(x)))
      ax.tick_params(axis = 'x', which = 'major', labelsize = 24)
      #plt.rcParams["figure.figsize"] = (len(list(subperformancedf.iterrows())),3)
      #plt.gcf().set_size_inches(len(list(subperformancedf.iterrows())),3)
      plt.savefig('%s/%s_%s_logo_RNACColor.png'%(args.out,args.pdbid, chain), dpi = 800)
      plt.clf()

    print(Probabilitydf)

    """
    # ======================================
    # Write Specificity into B factor slot
    # ======================================

    CurrentPdbStructure.df['ATOM']['b_factor'] = 0.0    
    for chain, res in zip(performancedf['ChainID'].tolist(), performancedf['ResidueID'].tolist()):
        matching_idx = CurrentPdbStructure.df['ATOM'].loc[(CurrentPdbStructure.df['ATOM']['chain_id'] == chain) & (CurrentPdbStructure.df['ATOM']['residue_number'] == res)].index.values
        spec = performancedf.loc[(performancedf['ResidueID'] == res) &(performancedf['ChainID'] == chain)]['Y-R_specificity'].tolist()[0]
        CurrentPdbStructure.df['ATOM'].iloc[matching_idx, CurrentPdbStructure.df['ATOM'].columns.get_loc('b_factor')] = spec

    CurrentPdbStructure.to_pdb(path='%s/%s_logo_loaded.pdb' %(args.out, args.pdbid), records=None, gz=False, append_newline=True)
    """







