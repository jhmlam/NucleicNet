#! /usr/bin/env python

import sys,os
import time
from Bio import Fasta

DEFAULT_DICT_FILE = '/project1/structure/mliang/pdb/derived_data/pdb_seqres.idx'
DEFAULT_OUTFH = sys.stdout

dict_file = DEFAULT_DICT_FILE
outfh = DEFAULT_OUTFH

start_time = time.time()
fdict = Fasta.Dictionary(dict_file)
elapse_time = time.time() - start_time
print >>sys.stderr, "Time to load dictionary:", elapse_time

start_time = time.time()
chainmap = {}
for key in fdict.keys():
    chainmap.setdefault(key[:4],[]).append(key)
elapse_time = time.time() - start_time
print >>sys.stderr, "Time to build chain map:", elapse_time

start_time = time.time()
args = sys.argv[1:]
if not args:
    args = sys.stdin

for field in args:
    fields = field.strip().split()
    for arg in fields:
        if arg in chainmap:
            for chain in chainmap[arg]:
                outfh.write(fdict[chain])
        else:
            try:
                outfh.write(fdict[arg])
            except KeyError:
                pass
elapse_time = time.time() - start_time
print >>sys.stderr, "Time to lookup entries:", elapse_time
