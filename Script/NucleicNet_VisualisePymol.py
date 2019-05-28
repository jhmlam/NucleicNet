
"""
Simple script to modify the pymol session template.
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


from Supporting import *



parser = ArgumentParser(description="This script will dock the mer in pdb onto the target")
parser.add_argument("--OutputFolder", type=str, dest="out",
                    help="Store all outputs")
parser.add_argument("--ApoFolder", type=str, dest="apo",
                    help="Store apo protein")
parser.add_argument("--Pdbid", type=str, dest="pdbid",
                    help="folder that contains the blind pdb")
args = parser.parse_args()

MkdirList([args.out])


# =====================================
# Create pymol session from template
# =====================================

# Copy the user input pdb file
shutil.copyfile('%s/%s.pdb' %(args.apo, args.pdbid),'%s/%s.pdb' %(args.out, args.pdbid))

# Read in the pml file
with open('Nucleic-Bind_VisualisePymol.pml', 'r') as file :
  filedata = file.read()

# Edit dummy filename string of the pymol script
filedata = filedata.replace('4f3t', '%s' %(args.pdbid))

# Write pml file
with open('%s/%s_temp.pml' %(args.out, args.pdbid), 'w') as file:
  file.write(filedata)

# Run pml to produce pse to downloaded
subprocess.call('cd %s/ ; pymol -cq %s_temp.pml; cd ../' %(args.out, args.pdbid), shell = True)


# ========================================
# convert xyz to pdb
# ========================================

xyz_df = pd.read_csv("%s/%s_strong_Bootstrap.xyz"%(args.out, args.pdbid), sep = '       ', names = ['Element','x','y','z'])

# Read in the apo protein
ppdb = PandasPdb()
ppdb.read_pdb('%s/%s.pdb' %(args.out, args.pdbid))

ppdb.df['HETATM']['record_name'] = ['HETATM' for i in range(len(xyz_df['Element'].tolist()))]
ppdb.df['HETATM']['atom_name'] = [i for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['residue_name'] = [i.upper() for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['chain_id'] = ['Z' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['blank_1'] = ['' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['blank_2'] = ['' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['alt_loc'] = ['' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['blank_3'] = ['' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['insertion'] = ['' for i in xyz_df['Element'].tolist()]


ppdb.df['HETATM']['occupancy'] = [0.00 for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['blank_4'] = ['' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['b_factor'] = [0.00 for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['segment_id'] = ['' for i in xyz_df['Element'].tolist()]
ppdb.df['HETATM']['element_symbol'] = [i for i in xyz_df['Element'].tolist()]
#ppdb.df['HETATM']['charge'] = ['' for i in xyz_df['Element'].tolist()]
#ppdb.df['HETATM']['line_idx'] = ['' for i in xyz_df['Element'].tolist()]


ppdb.df['HETATM']['residue_number'] = [i+1 for i in range(len(xyz_df['Element'].tolist()))]
ppdb.df['HETATM']['x_coord'] = [i for i in xyz_df['x'].tolist()]
ppdb.df['HETATM']['y_coord'] = [i for i in xyz_df['y'].tolist()]
ppdb.df['HETATM']['z_coord'] = [i for i in xyz_df['z'].tolist()]
ppdb.df['HETATM']['atom_number'] = [i+1 for i in range(len(xyz_df['Element'].tolist()))]


ppdb.to_pdb(path='%s/%s_strong_Bootstrap.pdb' %(args.out, args.pdbid), records=['HETATM'], gz=False, append_newline=True)

print(ppdb.df['HETATM'])
