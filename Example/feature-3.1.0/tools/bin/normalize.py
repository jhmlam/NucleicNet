#! /usr/bin/env python
# $Id: normalize.py,v 1.1 2004/11/17 09:33:01 mliang Exp $
# normalizes values to 1.0

import sys,os
import getopt
import fileinput
from Numeric import *
from MLab import *

def eusage():
    print "Usage: %s [OPTIONS] [FILENAME...]" % os.path.basename(sys.argv[0])
    sys.exit()

def myfmt(obj):
    if type(obj) == float:
        return '%.4g' % obj
    return str(obj)

sigma = None
args = sys.argv[1:]
try:
    optlist,args = getopt.getopt(args,"h:")
except getopt.GetoptError:
    eusage()
for opt,arg in optlist:
    if opt in ['-h']:
        eusage()
infh = fileinput.input(args)
outfh = sys.stdout

labels = []
values = []
for line in infh:
    if not line.split() or line[0] == '#':
        continue
    fields = line.split()
    labels.append(fields[0])
    values.append(map(float,fields[1:]))
sumvalues = sum(array(values))

for idx in range(len(labels)):
    label = labels[idx]
    value = array(values[idx])/sumvalues
    print >>outfh,'%s\t%s' % (label,'\t'.join(map(myfmt,value)))
