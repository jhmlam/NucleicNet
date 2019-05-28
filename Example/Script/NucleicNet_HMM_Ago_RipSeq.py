
"""
Ago In vivo RipSeq 
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
from scipy import spatial
from scipy.spatial.distance import euclidean
from scipy.linalg import eig 
import scipy
from numpy import sqrt, loadtxt, savetxt, argmax, floor, abs, max, exp, mod, zeros
from scipy import stats



#import urllib.request

import gzip
import pickle

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
parser.add_argument("--HMM", type=str, dest="hmmfile",
                    help="folder that contains the blind pdb", default = "Scripts/NucleicNet_4f3t_HMM_Model.pickle")
parser.add_argument("--SeqTxtFolder", type=str, dest="seqfolder",
                    help="folder that contains sequence given by Jayden from Prof. Chen")

args = parser.parse_args()

##########################################


# HMM Model loading
HMM_Model = pickle.load( open("%s" %(args.hmmfile) , 'rb'))


def HMM_test(seq, HMM_Model):
        test_synonym = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
        test_seq_translated_1 = np.array([[test_synonym[i] for i in seq]]).T
        result_1 = HMM_Model.score_samples(test_seq_translated_1)[0]
        return result_1





# =================================================================================================
# RipSeq results
# =================================================================================================




df_list = []
for f in glob.glob("%s/*.txt" %(args.seqfolder)):

   
   # Smoothed
   df = pd.read_csv(f, sep = '\t')
   df.loc[:,'minorRPM'] += 0.001
   df.loc[:,'majorRPM'] += 0.001


   # rename
   df["Reads Minor"] = df['minorRPM']
   df["Reads Major"] = df['majorRPM']


   # Log 10 
   df['Log10 Reads Major'] = np.log10(df.majorRPM)
   df['Log10 Reads Minor'] = np.log10(df.minorRPM)
   df['Log10 Reads Difference'] = df['Log10 Reads Major'] - df['Log10 Reads Minor']

   # Check that all major reads are always experimentally more abundant than minor reads. THis plot also allows us to see how many times it is.
   #df['Log10 Reads Difference'].plot.hist(bins=20, title= '%s' %(f.split("/")[-1].split(".")[0].split("_")[-1]))
   #plt.show()
   #plt.clf()


   # Retrieving Ago2 length 20-22
   mask = (df['minorSequence'].str.len() >= 20) & (df['majorSequence'].str.len() >= 20) & (df['minorSequence'].str.len() < 23) & (df['majorSequence'].str.len() < 23) 
   #mask = (df['minorSequence'].str.len() == 22) & (df['majorSequence'].str.len() == 22)
   df = df.loc[mask]
   
   # HMM logP
   for index, row in df.iterrows():
        df.loc[index,'HMM logP Minor'] = HMM_test(row['minorSequence'][0:8],HMM_Model)
        df.loc[index,'HMM logP Major'] = HMM_test(row['majorSequence'][0:8],HMM_Model)
        df.loc[index,'Minor LHS Sequence'] = row['minorSequence'][0:8]
        df.loc[index,'Major LHS Sequence'] = row['majorSequence'][0:8]
        df.loc[index, 'Source'] = f.split("/")[-1].split(".")[0].split("_")[-1]
        df.loc[index,'HMM Score Difference'] = df.loc[index,'HMM logP Major'] - df.loc[index,'HMM logP Minor']

   # Check repeats.
   #majidx = df.groupby(['Major LHS Sequence'])['Log10 Reads Major'].transform(min) == df['Log10 Reads Major']
   #majdf = df[majidx]


   # Check each txt file see if distribution similar or any outlier
   #majdf['HMM Score Difference'].plot(kind='hist', title= '%s' %(f.split("/")[-1].split(".")[0].split("_")[-1]), color = 'b', bins = 20)
   #plt.show()


   df_list.append(df)






# Seaborn Settings
sns.set(color_codes=True)
#sns.set_style("ticks")
#sns.set_style("whitegrid")
sns.set_style("white")

   
gdf = pd.concat(df_list)


# Check Repeats
majidx = gdf.groupby(['Major LHS Sequence'])['Log10 Reads Major'].transform(min) == gdf['Log10 Reads Major']
# Do not check repeats. This allows more samples.
majgdf = gdf#[majidx]



#print(majgdf)

# Check Experimental Read difference
majgdf.hist(column ='Log10 Reads Difference', bins=20, by = "Source", stacked = True, sharex = True, sharey = True)
#plt.savefig("%s/RipSeq_ExperimentalReadsDifference.png"%(args.seqfolder), dpi = 500)
plt.clf()


# ===========================
# Discarding poor reads
# ===========================
# As the accuracy of the Read deteriorates at lower values, all less than 50 RPM were discarded. 25: 1.397, 50: 1.69, 1: 0.0 (Full Window)
#majgdf = majgdf.loc[(majgdf['Log10 Reads Major'] > 1.397)]

# Discard by multiple instead of a fixed number
#majgdf = majgdf.loc[(majgdf['Log10 Reads Difference'] >= 2.0)]

# Double discard
majgdf = majgdf.loc[(majgdf['Log10 Reads Difference'] >= 2.0) & (majgdf['Log10 Reads Major'] > 1.39)]





# ========================
# Histogram of difference 
# ========================
#
majgdf["HMM Score Difference"].plot(kind='hist', title= 'HMM Score Difference (Major - Minor) Histogram', color = 'b', bins=np.linspace(-6.0, 6.0, num=13))

plt.savefig("%s/RipSeq_HMMlogPDifference_Combined.png"%(args.seqfolder), dpi = 500)
print("There are %s positives and %s negative values in HMM logP difference" %(np.sum(np.array(majgdf["HMM Score Difference"].tolist()) > 0), np.sum(np.array(majgdf["HMM Score Difference"].tolist()) < 0)))
print(majgdf["HMM Score Difference"].describe())
plt.clf()

# QQ plot to show normality
stats.probplot(majgdf["HMM Score Difference"], plot= plt)
plt.title('HMM Score Difference (Major - Minor) Q-Q Plot')
plt.savefig("%s/RipSeq_HMMlogPDifferenceQQPlot.png"%(args.seqfolder), dpi = 500)
plt.clf()

# Plot for each dataset
#majgdf.hist(column ='HMM Score Difference', bins=np.linspace(-6.0, 6.0, num=13), by = "Source", sharex = True, sharey = True, stacked = True)
#plt.savefig("%s/InVitro_HMMlogPDifference_Separated.png"%(args.seqfolder), dpi = 500)
#plt.clf()
performance = []
for i in majgdf["HMM Score Difference"].tolist():
    if i >= 0.0:
       performance.append("Positive")
    else:
       performance.append("Negative")

majgdf["Performance"] = performance
print(majgdf)

plt.rcParams["patch.force_edgecolor"] = True
g = sns.FacetGrid(majgdf, col='Source', hue='Performance', palette="husl", hue_order=['Negative', 'Positive'])
g = (g.map(sns.distplot, "HMM Score Difference", hist=True, rug=False, kde = False, bins=np.linspace(-6.0, 6.0, num=13),  hist_kws=dict(edgecolor="k", linewidth=2)).add_legend())
#plt.show()
plt.savefig("%s/RipSeq_HMMlogPDifference.png"%(args.seqfolder), dpi = 500)
plt.clf()



# =========================
# Stat test
# =========================

print("Overall Stat Test Result")

# T-test
print(stats.ttest_rel(majgdf['HMM logP Major'], majgdf['HMM logP Minor']))


# Wilcoxon Signed Rank test. Exclusion for zero division
print(stats.wilcoxon(majgdf['HMM logP Major'], majgdf['HMM logP Minor']))


# TODO Test per dataset
for s in sorted(set(majgdf['Source'].tolist())):

    print("%s Stat Test Result" %(s))

    # Descriptive stat
    print(majgdf.loc[majgdf['Source'] == s ]["HMM Score Difference"].describe())


    # T-test
    print(stats.ttest_rel(majgdf.loc[majgdf['Source'] == s ]['HMM logP Major'], majgdf.loc[majgdf['Source'] == s ]['HMM logP Minor']))


    # Wilcoxon Signed Rank test. Exclusion for zero division
    print(stats.wilcoxon(majgdf.loc[majgdf['Source'] == s ]['HMM logP Major'], majgdf.loc[majgdf['Source'] == s ]['HMM logP Minor']))




# Save the analysis
pickle.dump(majgdf, open("%s/RipSeq_Data_with_Prediction.df"%(args.seqfolder), 'wb'))




