
"""
Supporting Code 
"""


# You may copy here
import subprocess
import os
import sys
import re
import glob
from io import StringIO
from argparse import ArgumentParser
import shutil

import gc
import copy

import pandas as pd
from biopandas.pdb import PandasPdb
from collections import defaultdict

import numpy as np
from scipy import spatial
import random

#import urllib.request

import gzip
import pickle

import multiprocessing









# Mkdir if not exist
def MkdirList(folderlist):
 import os
 for i in folderlist:
  if not os.path.exists('%s' %(i)):
   os.mkdir('%s' %(i))

# Dict for Atom Attribute
CentroidAttributeDict = {"A":["N1", "C2", "C5", "C4", "N3", "C6", "N6", "N7", "C8", "N9"],"T":["N3", "C2", "C6", "C5", "N1", "C4", "O4", "O2", "C7"],"C":["N3", "C2", "C6", "C5", "N1", "C4", "O2", "N4"],"G":["N1", "C2", "C5", "C4", "N3", "C6", "O6", "N2", "N7", "C8", "N9"],"U":["N3", "C2", "C6", "C5", "N1", "C4", "O4", "O2"],"P":["P","OP1","OP2","O3\'"],"R":["O4\'", "C1\'", "C2\'", "C3\'", "C4\'", "O2\'"],"D":["O4\'", "C1\'", "C2\'", "C3\'", "C4\'"]}

# Visualisation of points
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
