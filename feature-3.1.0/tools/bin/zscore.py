#! /usr/bin/env python
# $Id: zscore.py,v 1.1 2004/11/17 09:33:35 mliang Exp $
# Converts values to zscore

import sys,os
import getopt
import fileinput
from Numeric import *
from MLab import *

def eusage():
    print "Usage: %s [OPTIONS] [FILENAME...]" % os.path.basename(sys.argv[0])
    print """
Options:
    -m MEAN
    -s STDEV
    -f STATFILE
        get mean and stdev from stat file, format:
        count mean stdev sum sumsqr
"""
    sys.exit()

mu = None
sigma = None
args = sys.argv[1:]
try:
    optlist,args = getopt.getopt(args,"m:s:f:")
except getopt.GetoptError:
    eusage()
for opt,arg in optlist:
    if opt in ['-m']:
        mu = float(arg)
    elif opt in ['-s']:
        sigma = float(arg)
    elif opt in ['-f']:
        mu,sigma = map(float,file(arg).next().split()[1:3])
infh = fileinput.input(args)
outfh = sys.stdout

if mu is None or sigma is None:
    values = [float(line) for line in infh if line.strip()]
    if mu is None:
        mu = mean(values)
    if sigma is None:
        sigma = std(values)
    for value in values:
        zscore = (value-mu)/sigma
        print >>outfh,zscore
else:
    for line in infh:
        if not line.strip():
            continue
        value = float(line)
        zscore = (value-mu)/sigma
        print >>outfh,zscore
