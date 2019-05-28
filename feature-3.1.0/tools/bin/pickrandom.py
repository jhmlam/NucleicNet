#! /usr/bin/env python

# Randomly select lines from file

import sys,os
import getopt
import random
from logutils import *

DEFAULT_SEED = None
DEFAULT_VERBOSITY = 0
DEFAULT_LABEL_LINES = False
DEFAULT_SPLIT_LINES = False
DEFAULT_OUTFH = sys.stdout

def eusage():
    print "Usage: %s [OPTION] FILENAME" % os.path.basename(sys.argv[0])
    print """
User must specify either -p VALUE or -e VALUE

Options:
    -p Probability of selecting line
        Use VALUE as probability of selection
    -e Expected number of lines
        Use VALUE as expected number of lines
    -l  Label lines
    -R RANDOM_SEED
        Set RANDOM_SEED as Seed
"""
    sys.exit(0)

def GetNumLines(filename):
    numlines = 0
    infh = file(filename)
    for line in infh:
        numlines += 1
    infh.close()
    return numlines

def main():
    verbosity = DEFAULT_VERBOSITY
    setdebuglevel(verbosity)
    mode = None
    value = None
    seed = DEFAULT_SEED
    labelLines = DEFAULT_LABEL_LINES
    splitLines = DEFAULT_SPLIT_LINES
    outfh = DEFAULT_OUTFH

    args = sys.argv[1:]
    try:
        optlist,args = getopt.getopt(args,"p:e:R:vls")
    except getopt.GetoptError:
        eusage()
    for opt,arg in optlist:
        if opt in ('-p',):
            mode = "probability_mode"
            value = float(arg)
        elif opt in ('-e',):
            mode = "expectation_mode"
            value = float(arg)
        elif opt in ('-l',):
            labelLines = True
        elif opt in ('-s',):
            splitLines = True
        elif opt in ('-R',):
            seed = int(arg)
        elif opt in ('-v',):
            verbosity += 1
            setdebuglevel(verbosity)
    if not mode or not value:
        eusage()
    if len(args) != 1:
        eusage()
    filename = args.pop(0)

    if not os.path.exists(filename):
        fatal_error('Could not open file',filename)

    if mode == 'probability_mode':
        probability = value
    elif mode == 'expectation_mode':
        numlinesinfile = GetNumLines(filename)
        probability = float(value) / numlinesinfile

    if probability < 0.0 or probability > 1.0:
        fatal_error('Probability out of range',probability)

    pickoutfh = [sys.stdout]*2
    if splitLines:
        pickoutfh[0] = file('%s.0' % filename,'w')
        pickoutfh[1] = file('%s.1' % filename,'w')

    random.seed(seed)
    debug(0,'Filename:',filename)
    debug(0,'Probability:',probability)
    debug(0,'Random Seed:',seed)
    debug(0,'Label Lines:',labelLines)
    debug(0,'Split Lines:',splitLines)

    infh = file(filename)
    for line in infh:
        picked = random.random() < probability
        if splitLines:
            pickoutfh[picked].write(line)
        elif labelLines:
            outfh.write("%s\t" % picked)
            outfh.write(line)
        else:
            if picked:
                outfh.write(line)
    infh.close()

if __name__ == '__main__':
    main()
