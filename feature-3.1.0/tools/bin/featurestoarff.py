#! /usr/bin/env python

# Converts .feature format to weka format
import sys,os
import getopt
from utils import *

DEFAULT_VERBOSITY = 0
DEFAULT_CLASS_NAME = '?'
DEFAULT_PROPERTY_FILENAME = 'propnames.txt'
DEFAULT_COLUMN_OFFSET = 1

def eusage():
    print "Usage: %s [OPTIONS] FILENAME" % os.path.basename(sys.argv[0])
    print """
Options:
    -v  Increase verbosity
    -c CLASSNAME [Default: %(DEFAULT_CLASS_NAME)s]
        Specify CLASSNAME of data
    -p PROPERTYNAME LIST [Default: %(DEFAULT_PROPERTY_FILENAME)s]
        Specify file to read in property names
    -o COLUMNOFFSET [Default: %(DEFAULT_COLUMN_OFFSET)s]
        Specify starting column for data
""" % globals()
    sys.exit(0)

verbosity = DEFAULT_VERBOSITY
className = DEFAULT_CLASS_NAME
propFilename = DEFAULT_PROPERTY_FILENAME
columnOffset = DEFAULT_COLUMN_OFFSET

args = sys.argv[1:]
try:
    optlist,args = getopt.getopt(args,"vp:c:o:")
except getopt.GetoptError:
    eusage()
for opt,arg in optlist:
    if opt in ['-v']:
        verbosity += 1
        setdebuglevel(verbosity)
    elif opt in ['-p']:
        propFilename = arg
    elif opt in ['-c']:
        className = arg
    elif opt in ['-o']:
        columnOffset = int(arg)
if len(args) != 1:
    eusage()
filename = args[0]

if className != '?':
    className = "'%s'" % className

propnames = []
for line in file(propFilename):
    propnames.append(line.strip().lower())
numProperties = len(propnames)

input = file(filename)

print "%% Input File: %s" % filename
print "@RELATION features"
for propname in propnames:
    print "@ATTRIBUTE '%s' NUMERIC" % propname
print "@ATTRIBUTE 'classname' {SITE,NONSITE}"
print "@DATA"
for line in input:
    line = chomp(line)
    fields = line.split()
    print ",".join(fields[columnOffset:columnOffset+numProperties]+[className])
