#! /usr/bin/env python

# Given a hit file and PDB structure,
# locate the residues close to the hit

import sys,os
import getopt
import re
from feature.hitsfile import HitFile
from pdbutils import goodpdbfilename
from simplepdb import PDBFile

DEFAULT_SCORE_CUTOFF = 50.0
DEFAULT_RADIUS_CUTOFF = 5.0
DEFAULT_MAX_NUM_HITS = None
DEFAULT_MAX_NUM_RESIDUES = None

def eusage():
    print "Usage: %s [OPTIONS] HITFILE [PDBID]" % os.path.basename(sys.argv[0])
    print """
Options:
    -c SCORECUTOFF [Default: %(DEFAULT_SCORE_CUTOFF)s]
        Set the maximum score to be considered as hit
    -r RADIUSCUTOFF [Default: %(DEFAULT_RADIUS_CUTOFF)s]
        Set the maximum distance of residue from hit
    -k NUMHITS [Default: %(DEFAULT_MAX_NUM_HITS)s]
        Set the maximum number of hits to consider
    -n NUMRESIDUES [Default: %(DEFAULT_MAX_NUM_RESIDUES)s]
        Set the maximum number of residues reported per hit
    """ % globals()
    sys.exit(0)

class MyHits:
    def __init__(self,hfile):
        # make copy of hit list
        pass

    def FilterByCutoff(self,cutoff):
        pass

    def SortDescending(self):
        pass

    def FilterByTopHits(self,numhits):
        pass

class MyResidues:
    def __init__(self,hit,atomlist):
        # make copy of atom list
        # group atoms into residues

    def SortAscending(self):
        pass

    def FilterByTopResidues(self,numresidues):
        pass


# Check arguments
score_cutoff = DEFAULT_SCORE_CUTOFF
radius_cuotff = DEFAULT_RADIUS_CUTOFF
max_num_hits = DEFAULT_MAX_NUM_HITS
max_num_residues = DEFAULT_MAX_NUM_RESIDUES
try:
    optlist,args = getopt.getopt(args,"c:r:k:n:")
except getopt.GetoptError:
    eusage()
for opt,arg in optlist:
    if opt in ['-c']:
        score_cutoff = float(arg)
    elif opt in ['-r']:
        radius_cutoff = float(arg)
    elif opt in ['-k']:
        max_num_hits = int(arg)
    elif opt in ['-n']:
        max_num_residues = int(arg)
if len(args) not in [1,2]:
    eusage()
hitsfilename = args[0]
if len(args) > 1:
    pdbid = args[1]
else:
    pdbid = pdbidFromFilename(hitsfilename)
pdbfilename = goodpdbfilename(pdbid)

# Load hit file
hfile = HitsFile(hitsfilename)
hits = MyHits(hfile)
# If score cutoff:
if score_cutoff:
    # Filter scores by score cutoff
    hits.FilterByCutoff(score_cutoff)
# Sort hits by descending score
hits.SortDescending()
# If maxhits:
if max_num_hits:
    # Filter scores by max hits
    hits.FilterByTopHits(hits,max_num_hits)

# Load pdb file
structure = PDBFile(pdbfilename)
# Load neighbor map
neighborMap = NeighborMap(structure)
# For each hit
for hit in hits:
    # Locate nearest atoms within radius
    neighborAtoms = neighborMap.query(hit)
    # Collapse atoms into residues - distance is minimum of atoms
    neighborResidues = MyResidues(hit,neighborAtoms)
    # Sort residues by ascending distance
    neighborResidues.SortAscending()
    # If max residues:
    if max_num_residues:
        # Take top residues 
        neighborResidues.FilterByTopResidues(max_num_residues)

