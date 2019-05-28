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

import urllib.request

import gzip
import pickle

import multiprocessing

from Supporting import *
import Supporting
import Supporting_NucleicParts






parser = ArgumentParser(description="This script will define sites as Ptf files for each pdb")
parser.add_argument("--DsspFolder", type=str, dest="dssp",
                    help="folder of output files")
parser.add_argument("--ApoFolder", type=str, dest="apo",
                    help="folder of output files")
parser.add_argument("--TrainingFolder", type=str, dest="training",
                    help="All available training data")
parser.add_argument("--FeatureFolder", type=str, dest="feature",
                    help="All available training data")
parser.add_argument("--UserKeyword", type=str, dest="userkey",
                    help="A User Keyword to append to the end of outputfilename to denote that some special parameter were used. E.g. A very close to atom halo. Please Do Not contain . in the filename")
args = parser.parse_args()

#####################################################################




if not args.userkey:
    userkeyword = ''
else:
    userkeyword = "_%s"%(args.userkey)



def pool_init():
    gc.collect()

class MakeFF(object):
 def __init__(self):
    print("Starting To Generate Training")
 def __call__(self, pdbid):

        print (pdbid)
        subprocess.call("export PDB_DIR=%s ; export DSSP_DIR=%s; export FEATURE_DIR=%s; featurize -P %s/%s.ptf -s %s > %s/%s.ff"%(args.apo,args.dssp,args.feature,args.training,pdbid,args.training, args.training, pdbid), shell=True)





Pdbid = [i.split("/")[-1].split(".")[0] for i in glob.glob("%s/*%s.ptf"%(args.training,userkeyword))]

pool=multiprocessing.Pool(processes=12, initializer=pool_init, maxtasksperchild=10000)
results = pool.map(MakeFF(), Pdbid)
pool.close
gc.collect()

