
"""
Score some miRNA sequences from a saved HMM.
"""


import time
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
from operator import itemgetter
import gc
import copy

import pandas as pd
from biopandas.pdb import PandasPdb
from collections import defaultdict
from collections import Counter
# some nice settings
pd.set_option('precision', 5)

import random
import numpy as np
import scipy



import multiprocessing
from random import random





import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = 'k'
from mpl_toolkits.mplot3d import Axes3D



import seaborn as sns





from math import sin, cos, acos, sqrt
from scipy import sparse,signal, interpolate
from sklearn.neighbors import NearestNeighbors

from Supporting import *





parser = ArgumentParser(description="This script will dock the mer in pdb onto the target")
parser.add_argument("--TestSequences", type=str, dest="seq",
                    help="a typed csv containing the mi/siRNA sequence letters of length at least 8 e.g. AUGGUGUGUGUUUUUUU,UGUUGUGUUUUAAAAAAACCCC")
parser.add_argument("--HMM", type=str, dest="hmmfile",
                    help="folder that contains the blind pdb", default = "Scripts/NucleicNet_4f3t_HMM_Model.pickle")

args = parser.parse_args()

##########################################


Sequence_list = args.seq.split(',')


# HMM Model loading
#Target_name  = args.targetname
#HMM_Model = pickle.load( open("%s/%s_HMM_Model.pickle" %(args.predictionfolder, Target_name), 'rb'))
HMM_Model = pickle.load( open("%s" %(args.hmmfile) , 'rb'))


def HMM_test(seq, HMM_Model):
        test_synonym = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
        test_seq_translated_1 = np.array([[test_synonym[i] for i in seq]]).T
        result_1 = HMM_Model.score_samples(test_seq_translated_1)[0]
        return result_1

# HMM Score
for seq in Sequence_list:
       score = HMM_test(seq[0:8],HMM_Model)
       print(seq[0:8], score)
