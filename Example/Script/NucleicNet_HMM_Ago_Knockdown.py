
"""
In Vivo Knockdown
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



from Supporting import *





parser = ArgumentParser(description="This script will dock the mer in pdb onto the target")
parser.add_argument("--SeqTxtFolder", type=str, dest="seqfolder",
                    help="folder that contains sequence given by Jayden from Prof. Chen")
parser.add_argument("--HMM", type=str, dest="hmmfile",
                    help="folder that contains the blind pdb", default = "Scripts/NucleicNet_4f3t_HMM_Model.pickle")

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
# In vivo results
# =================================================================================================
for f in glob.glob("%s/siRNA_targeting_Top200_genes_in_hek293_linux.csv" %(args.seqfolder)):
    company_df = pd.read_csv(f, sep = ",")


# Check repeats
si_idx = company_df.groupby(['siRNA sequence'])['Mean KnockDown Level'].transform(max) == company_df['Mean KnockDown Level']
company_df = company_df[si_idx]

company_df['Mean KnockDown Level'] = [float(i) for i in company_df['Mean KnockDown Level'].tolist()]
company_df['Normalised Mean KnockDown Level'] = company_df.groupby('Cell Line')['Mean KnockDown Level'].apply(lambda x: (x)/(x.max() + 0.001))

# HMM Score
for index, row in company_df.iterrows():
        company_df.loc[index,'HMM Score'] = HMM_test(row['siRNA sequence'][::-1][0:8],HMM_Model)


# Define Apparent Outliers
Apparent_Outliers = ["RAN", "RPS5", "NPM1", "H2AFZ", "PKM", "HSPA8", "HIST1H2AG", "XRCC5", "PRDX1", "ATP5B", "RPS4X"]
# Flat lines: "ATP5B", "RPS4X"
sns.set_style("white")
invivo_dflist = []
# Analyse per cell line per gene
for cell in sorted(set(company_df['Cell Line'].tolist()) - set(["NIH/3T3", "3T3-L1", "297T/17", "A3"])):  # Rejected Cell Lines: Too few samples.
    


    # Plot A549 has high SD in HMM Score
    local_company_df = company_df.loc[company_df['Cell Line'].isin([cell])]
    local_company_df['Normalised Mean KnockDown Level'] = company_df.groupby('GENE')['Mean KnockDown Level'].apply(lambda x: (x)/(x.max() + 0.001))


    # Focus on Region CDS
    local_company_df = local_company_df.loc[(local_company_df["Region"] == "CDS")]

    # Focus on TRC version1
    local_company_df = local_company_df.loc[(local_company_df["TRC Version"] == 1)]

    # Remove datapoint with gene count == 1
    Gene_Member_dict = Counter(local_company_df['GENE'].tolist())
    for index, row in local_company_df.iterrows():
        local_company_df.loc[index,'Gene Count'] = Gene_Member_dict[row['GENE']]
    local_company_df = local_company_df.loc[local_company_df["Gene Count"] > 1]


    # Remove datapoint groups with narrow range of mean knockdown level (0.1)
    knockdown_range_dict = defaultdict(list)
    for index, row in local_company_df.iterrows():
        knockdown_range_dict[row['GENE']].append(row['Mean KnockDown Level'])
    for k,v in knockdown_range_dict.items():
        knockdown_range_dict[k] = np.max(knockdown_range_dict[k]) - np.min(knockdown_range_dict[k]) #np.std(knockdown_std_dict[k])
    for index, row in local_company_df.iterrows():
        local_company_df.loc[index,'Gene Knockdown Range'] = knockdown_range_dict[row['GENE']]
    local_company_df = local_company_df.loc[local_company_df["Gene Knockdown Range"] > 0.1]



    # Define Markers
    mks = itertools.cycle(['o', '^', '*', '8', 's', 'p', 'D'])
    markers = [next(mks) for i in local_company_df.loc[~(local_company_df["GENE"].isin(Apparent_Outliers))]["GENE"].unique()]

    invivo_dflist.append(local_company_df)


# Plot all in vivo data
invivo_gdf = pd.concat(invivo_dflist)

print(invivo_gdf)

# Define Markers for good
mks = itertools.cycle(['o', '^', '*', '8', 's', 'p', 'D'])
markers = [next(mks) for i in invivo_gdf.loc[~(invivo_gdf["GENE"].isin(Apparent_Outliers))]["GENE"].unique()]
sns.set(style="white", font_scale=1.5)

g = sns.lmplot("Mean KnockDown Level", "HMM Score",
data=invivo_gdf.loc[~(invivo_gdf["GENE"].isin(Apparent_Outliers)) ],
hue="GENE",col = 'Cell Line',
palette=sns.color_palette("husl", len(invivo_gdf.loc[~(invivo_gdf["GENE"].isin(Apparent_Outliers))]["GENE"].unique())),
markers=markers,
ci=None, n_boot=100, order=1, robust=False, truncate=True, scatter_kws={'linewidths':1,'edgecolor':'k', "s": 65}, line_kws = {'alpha':0.7}, legend_out=True, height = 6.01, aspect = 0.8)
g.savefig("%s/Knockdown_Relation_All_Positive_publication.png"%(args.seqfolder), dpi = 500)
plt.clf()

# ======================
# Show Negative
# ======================

# Define Markers for bad
mks = itertools.cycle(['o', '^', '*', '8', 's', 'p', 'D'])
markers = [next(mks) for i in invivo_gdf.loc[(invivo_gdf["GENE"].isin(Apparent_Outliers))]["GENE"].unique()]

g = sns.lmplot("Mean KnockDown Level", "HMM Score",
data=invivo_gdf.loc[(invivo_gdf["GENE"].isin(Apparent_Outliers)) ],
hue="GENE", col = 'Cell Line',
palette=sns.color_palette("husl", len(invivo_gdf.loc[(invivo_gdf["GENE"].isin(Apparent_Outliers))]["GENE"].unique())),
markers=markers,
ci=None, n_boot=100, order=1, robust=False, truncate=True, scatter_kws={'linewidths':1,'edgecolor':'k', "s": 65}, line_kws = {'alpha':0.7}, legend_out=True, height = 6.01, aspect = 0.8)
g.savefig("%s/Knockdown_Relation_All_Negative_publication.png"%(args.seqfolder), dpi = 500)
plt.clf()

pickle.dump(invivo_gdf, open("%s/InVivo_Data_with_Prediction.df"%(args.seqfolder), 'wb'))

print("There are %s negative correlation out of %s data"%(len(invivo_gdf.loc[(invivo_gdf["GENE"].isin(Apparent_Outliers))]["GENE"].tolist()), len(invivo_gdf["GENE"].tolist())))


