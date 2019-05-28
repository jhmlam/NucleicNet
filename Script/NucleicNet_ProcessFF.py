import subprocess
import os
import sys
import re
import glob
from io import StringIO
from argparse import ArgumentParser

import gc
import copy

import pandas as pd
from biopandas.pdb import PandasPdb
from collections import defaultdict

import numpy as np
from scipy import spatial
import random


import gzip
import pickle
#import cPickle as pickle

import multiprocessing

#from Supporting import *
#import Supporting
#import Supporting_NucleicParts






parser = ArgumentParser(description="This script will define sites as Ptf files for each pdb")
parser.add_argument("--TrainingFolder", type=str, dest="training",
                    help="All available training data")
parser.add_argument("--FeatureFolder", type=str, dest="feature",
                    help="All available training data")
parser.add_argument("--PropertyNameFile", type=str, dest="propertynamefile",
                    help="folder holding the XXXX.properties")
parser.add_argument("--Shells", type=int, dest="shells",
                    help="The number of shells")
parser.add_argument("--UserKeyword", type=str, dest="userkey",
                    help="A User Keyword to append to the end of outputfilename to denote that some special parameter were used. E.g. A very close to atom halo. Please Do Not contain . in the filename")
args = parser.parse_args()

#####################################################################


if not args.userkey:
    userkeyword = ''
else:
    userkeyword = "_%s"%(args.userkey)



# A. Prepare column names for dataframe
# Read in the property name
propnamelist=[]
propfile = open("%s/%s"%(args.feature,args.propertynamefile), 'r')
propdata = propfile.read().splitlines()
for line in propdata:
    if line:
        if not line.startswith("#"):
            propnamelist.append(str(line))
propfile.close()

# This is the list of column names for the Dataframe
columnlist = ["Annotation"]
for shell in range(args.shells):
   for prop in propnamelist:
       columnlist.append(str(shell)+str(prop))


# B. Processing the ff files

def pool_init():
    gc.collect()

class ProcessFF(object):
 def __init__(self):
    print("Starting To Generate Training")
 def __call__(self, pdbid):

        print (pdbid)

        # Read in the raw ff file
        ffraw=[]
        fffile = open("%s/%s.ff"%(args.training, str(pdbid)), 'r')
        ffdata = fffile.read().splitlines()
        for line in ffdata:
            if line.startswith("Env_"):
                ffraw.append(str(line))
        fffile.close()
        del ffdata
        gc.collect


        fflistoflist=[]
        for item in ffraw:
            renamed = [str("%s:%s:%s:%s:%s:%s:%s:%s" %(item.split("\t")[0].split("_")[1], str(item.split("\t")[-1][0]), item.split("\t")[-1].split(":")[0].replace(str(item.split("\t")[-1][0]), str("")), item.split("\t")[-1].split(":")[1].split("@")[0], item.split("\t")[-1].split(":")[2], item.split("\t")[-5], item.split("\t")[-4], item.split("\t")[-3]))]
            renamed.extend(item.split("\t")[1:-6])
            fflistoflist.append(renamed)
        

        df = pd.DataFrame(columns=columnlist, data=fflistoflist)
        #pd.set_option('display.max_columns', len(columnlist))
        #pd.set_option('display.max_rows', len(df))
        #print(df)

        df.to_pickle("%s/%s.pkl" %(args.training, str(pdbid)))




Pdbid = [i.split("/")[-1].split(".")[0] for i in glob.glob("%s/*%s.ff"%(args.training,userkeyword))]

pool=multiprocessing.Pool(processes=12, initializer=pool_init, maxtasksperchild=10000)
results = pool.map(ProcessFF(), Pdbid)
#results = pool.map(ProcessFF(), ['4do9_SiteCentroid'])
pool.close
gc.collect()

