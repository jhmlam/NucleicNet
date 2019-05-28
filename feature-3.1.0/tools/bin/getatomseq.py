#! /usr/bin/env python

import sys,os
from pdbutils import goodpdbfilename
from simplepdb import PDBFile

args = sys.argv[1:]
if args: idlist = args
else: idlist = [x.strip() for x in sys.stdin]

def formatter(line,width):
    sections = []
    for i in range(0,len(line),width):
        sections.append(line[i:i+width])
    return '\n'.join(sections)

for pdbid in idlist:
    pdbfilename = goodpdbfilename(pdbid)
    if not pdbfilename: continue
    structure = PDBFile(pdbfilename)
    if not structure: continue
    for chain in structure.chains():
        print '> %s:%s' % (pdbid,chain.chainId)
        print formatter(chain.sequence(),60)
