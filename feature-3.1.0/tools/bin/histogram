#! /usr/bin/env python
# $Id: histogram.py,v 1.10 2005/05/10 21:48:46 mliang Exp $
# Copyright (c) 2004 by Mike Liang. All rights reserved.

# Bins data for generating a histogram

import sys,os
import getopt
import fileinput
from Numeric import *

_eps = 1e-5
_DEFAULT_CENTERS = 10
_DEFAULT_EDGES = None
_DEFAULT_NORMALIZE = None
_DEFAULT_IGNORE_ZEROS = False
_DEFAULT_PRINT_EDGE = None
_DEFAULT_CUMULATIVE = False

_error_fh = sys.stderr
def fatal_error(*args):
    print >>_error_fh, "Fatal Error:"," ".join(map(str,args))
    sys.exit(2)

def collapse(lol):
    result = reduce(lambda a,b:a+b,lol)
    return result

def BinData(data,edges):
    data = asarray(data)
    edges = asarray(edges)
    counts = zeros(size(edges)+1)
    for val in data:
        for idx,edge in enumerate(edges):
            if val < edge:
                counts[idx] += 1
                break
        else:
            counts[-1] += 1
    return counts
    
class Histogram:
    def __init__(self,*args,**kwargs):
        self.BuildHistogram(*args,**kwargs)

    def BuildHistogram(self,data,centers=None,miny=None,maxy=None,edges=None):
        if centers is None and edges is None:
            raise ValueError('Must specify centers or edges')
        data = asarray(data,'d')
        if centers is not None:
            centers = asarray(centers,'d')
        if edges is not None:
            edges = asarray(edges,'d')

        # len(centers) == numbins
        # len(edges) == numbins+1
        # len(histC) == numbins+2 (underflow,overflow)
        if miny is None:
            miny = min(data)
        if maxy is None:
            maxy = max(data)

        if edges is not None:
            numbins = len(edges)-1
            centers = (edges[:-1]+edges[1:])/2.0
        elif size(centers) == 1:
            # generate centers and edges
            numbins = int(centers[0])
            # fudge boundaries if no change
            if maxy == miny:
                miny -= floor(numbins/2.0) - 0.5
                maxy += ceil(numbins/2.0) - 0.5
            # calculate binsize
            binsize = (maxy-miny)/numbins
            # generate bin edges (fudge upper range to generate outer edge)
            edges = arange(miny,maxy+binsize/2.0,binsize,'d')
            # replace outer edge with maximum data value
            edges[-1] = maxy
            # shift edges to get centers
            centers = edges[:-1]+binsize/2.0
        else:
            # convert centers to edges
            numbins = size(centers)
            # |edges| == |centers|+1
            edges = zeros(size(centers)+1,'d')
            # internal edges are midpoints of centers
            edges[1:-1] = (centers[:-1]+centers[1:])/2.0
            # left edge is c[0]-(c[1]-c[0])/2.0 or mindata
            #edges[0] = max((3*centers[0]-centers[1])/2.0,miny)
            edges[0] = (3*centers[0]-centers[1])/2.0
            # right edge is c[-1]+(c[-1]-c[-2])/2.0 or maxdata
            #edges[-1] = min((3*centers[-1]-centers[-2])/2.0,maxy)
            edges[-1] = (3*centers[-1]-centers[-2])/2.0

        # fudge edges to be ( ]  if edge val, is big, increase fudgefactor
        fudge = abs(edges)*_eps
        putmask(fudge,fudge<_eps,_eps)
        edges = edges + fudge
        # generate counts from edges |counts| == |edges|+1
        counts = BinData(data,edges)
        # collapse underflow,overflow
        counts[1] += counts[0]
        counts[-2] += counts[-1]
        counts = counts[1:-1]

        self.centers = centers
        self.edges = edges
        self.counts = counts

    def GetCounts(self):
        return self.counts

    def GetNormalizedCounts(self,normFactor=1.0):
        return self.counts*normFactor/sum(self.counts)

    def GetItems(self,label=None):
        if label is none:
            label = 'lower'
        return zip(self.counts,self.GetLabels(label))

    def GetLabels(self,label=None):
        if label is None:
            label = 'lower'
        if label == 'lower':
            labels = self.GetLowerEdge()
        elif label == 'upper':
            labels = self.GetUpperEdge()
        else:
            labels = self.GetCenters()
        return labels

    def GetEdges(self):
        return self.edges

    def GetCenters(self):
        return list(self.centers)

    def GetLowerEdge(self):
        return self.edges[:-1]
        #return [None] + list(self.GetInternalEdges())

    def GetUpperEdge(self):
        return self.edges[1:]
        #return list(self.GetInternalEdges()) + [None]

    def GetInternalEdges(self):
        return self.edges[1:-1]

def histogram(*args,**kwargs):
    hist = Histogram(*args,**kwargs)
    return hist.counts,hist.centers
    
def FormatFloat(val):
    if val is None:
        return '-'
    return '%.4g' % val

def _eusage():
    print "Usage: %s [OPTIONS] [FILENAME]" % os.path.basename(sys.argv[0])
    print """
Options:
    -n NUMBINS
        Use NUMBINS
    -c CENTERS
        Use CENTERS (whitespace delimited)
    -r MIN[[:MAX]:STEP]
        Use range params to generate centers
    -x [MIN][:MAX]
        Use range params as data minimum and maximum
    -N NORMFACTOR
        Normalizes counts to NORMFACTOR
    -z  Suppresses display of 0 count bins
    -e Print lower edge
    -E upper|lower
        Print upper or lower edge
"""
    sys.exit(1)

def _main():
    args = sys.argv[1:]
    normalize = _DEFAULT_NORMALIZE
    ignoreZeros = _DEFAULT_IGNORE_ZEROS
    centers = _DEFAULT_CENTERS
    edges = _DEFAULT_EDGES
    printEdge = _DEFAULT_PRINT_EDGE
    cumulative = _DEFAULT_CUMULATIVE
    kwargs = {'centers':centers, 'edges':edges}
    try:
        optlist,args = getopt.getopt(args,"n:c:r:R:N:zx:eE:C")
    except getopt.GetoptError,e:
        _eusage()
    for opt,arg in optlist:
        if opt in ['-n']:
            centers = int(arg)
        elif opt in ['-c']:
            if type(centers) != list:
                centers = []
            centers.extend(map(float,arg.split()))
        elif opt in ['-r']:
            fields = map(float,arg.split(':'))
            # fudge to generate upper bound
            if len(fields) > 1:
                fields[1] += _eps
            if len(fields) > 3 or len(fields) < 2:
                _eusage()
            centers = arange(*fields)
            kwargs['centers']=centers
        elif opt in ['-R']:
            fields = map(float,arg.split(':'))
            # fudge to generate upper bound
            if len(fields) > 1:
                fields[1] += _eps
            if len(fields) > 3 or len(fields) < 2:
                _eusage()
            edges = arange(*fields)
            kwargs['edges']=edges
        elif opt in ['-x']:
            fields = arg.split(':')
            if len(fields) < 1 or len(fields) > 2:
                _eusage()
            if fields[0] != '':
                kwargs['miny']=float(fields[0])
            if len(fields) == 2 and fields[1] != '':
                kwargs['maxy']=float(fields[1])
        elif opt in ['-N']:
            normalize = float(arg)
        elif opt in ['-z']:
            ignoreZeros = True
        elif opt in ['-e']:
            printEdge = 'lower'
        elif opt in ['-E']:
            printEdge = arg
        elif opt in ['-C']:
            cumulative = True

    infh = fileinput.input(args)

    data = collapse([map(float,line.split()) for line in infh])
    if data and (centers or edges):
        hist = Histogram(data,**kwargs)
        if normalize is not None:
            counts = hist.GetNormalizedCounts(normalize)
        else:
            counts = hist.GetCounts()
        labels = hist.GetLabels(printEdge)
        value = 0
        for count,label in zip(counts,labels):
            if count or not ignoreZeros:
                if not cumulative:
                    value = count
                else:
                    value += count
                print '\t'.join(map(FormatFloat,[value,label]))

if __name__ == '__main__':
    _main()
