import subprocess
import multiprocessing
import os
from argparse import ArgumentParser
import sys
import re
import pandas as pd
from io import StringIO
import urllib.request
import glob
import gzip
import gc

from Supporting import *



parser = ArgumentParser(description="This script will make dssp for all pdb files within indicated folder in a parallel fashion")
parser.add_argument("--DsspFolder", type=str, dest="dssp",
                    help="folder of output files")
parser.add_argument("--PdbFolders", type=str, dest="pdb",
                    help="folder of output files")
args = parser.parse_args()



Pdbfolders = str(args.pdb).split(",")


MkdirList([args.dssp])


def pool_init():
    gc.collect()

class MakeDssp(object):
 def __init__(self):
    print("Starting To Generate Dssp")
 def __call__(self, j):
    print(j)
    subprocess.call("dssp -i %s -o %s/%s.dssp" %(j,args.dssp,j.split("/")[-1].split(".")[0]), shell = True)


jobs=[]
for i in Pdbfolders:
    for j in glob.glob("%s/*.pdb" %(i)):
      if not os.path.exists("%s/%s.dssp"%(args.dssp,j.split("/")[-1].split(".")[0])):
        jobs.append(j)


pool=multiprocessing.Pool(processes=24, initializer=pool_init, maxtasksperchild=10000)
results = pool.map(MakeDssp(), jobs)
pool.close
gc.collect()



