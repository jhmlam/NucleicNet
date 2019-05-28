#! /usr/bin/env python

# Get nonsite chains for a list of chains
# Eliminate all pdbids that are 'homologous' to any of the chains

import sys,os
import getopt
import astralfilter
from logutils import *

DEFAULT_VERBOSITY = 0
DEFAULT_IDENTITY = 40

def eusage():
    print "Usage: %s [OPTIONS] SITEFILENAME NONSITEFILENAME" % os.path.basename(sys.argv[0])
    print """
Options:
    -v  Increase verbosity
    -n IDENTITY
        Set identity level
"""
    sys.exit(0)

verbosity = DEFAULT_VERBOSITY
identity = DEFAULT_IDENTITY
args = sys.argv[1:]
try:
    optlist,args = getopt.getopt(args,'vn:')
except getopt.GetoptError:
    eusage()
for opt,arg in optlist:
    if opt in ['-v']:
        verbosity += 1
        setdebuglevel(verbosity)
    if opt in ['-n']:
        identity = int(arg)
if len(args) != 2:
    eusage()
sitefilename = args[0]
nonsitefilename = args[1]

if not os.path.exists(sitefilename):
    fatal_error('File does not exist:',sitefilename)
if not os.path.exists(nonsitefilename):
    fatal_error('File does not exist:',nonsitefilename)

homologFile = astralfilter.AstralHomologFile(identity)

# Create map of not-allowed canonical domain ids
# For each site entry
illegalIds = {}
for line in file(sitefilename):
    entry = line.strip()
    # Find canonical
    domainIds = homologFile.getbytype(entry[:4],'pdb')
    debug(1,'Number canonical for %s:' % entry,len(domainIds))
    # Mark canonical as not allowed
    for id in domainIds:
        illegalIds[id] = 1
debug(1,'Number illegal',len(illegalIds))

# For each nonsite entry
for line in file(nonsitefilename):
    entry = line.strip()
    # Find canonical
    domainIds = homologFile.getbytype(entry[:4],'pdb')
    # If canonical not allowed, continue
    illegal = 0
    for id in domainIds:
        if illegalIds.has_key(id):
            illegal = 1
            break
    if illegal:
        continue
    # print
    print entry
