#!/bin/env py
#
#
 
"""Hack script to generate an amber_params file with H charges
   added to each H atom's heavy atom.  H atoms are assumed to belong 
   to the heavy atom it follows in the input file."""
__author__ = "Randall J. Radmer"
__version__ = "1.0"
  
 
import sys, os
import getopt
import fileinput
#

def doIt(args_proper, printH, xValue):
    try:
        inFilename=args_proper[0]
        outFilename=args_proper[1]
    except IndexError:
        usageError()

    fIn=open(inFilename)
    inData={}
    outData={}
    setLabels=[]
    for line in fIn:
        commentIndex=line.find('#')
        if(commentIndex>=0): line=line[:commentIndex]
        items=line.split()
        if line.find('set:')==0:
            setLabels.append(items[-1])
            inData[setLabels[-1]]=[]
            outData[setLabels[-1]]=[]
        elif len(items)>=6:
            inData[setLabels[-1]].append(items[:6])
            outData[setLabels[-1]].append(items[:6])
    fIn.close()

    fOut=open(outFilename, 'w')
    for setLabel in setLabels:
        hTotal=0
        for itemIndex in range(len(inData[setLabel]), 0, -1):
            (resType, atomName, typeLabel, atomType,
             chargeLabel, chargeStringIn) = inData[setLabel][itemIndex-1]
            if '0123456789H'.find(atomName[0])>=0:
                hTotal+=float(chargeStringIn)
                outData[setLabel][itemIndex-1][-1]='0.0000'
            else:
                charge=float(chargeStringIn)+hTotal
                outData[setLabel][itemIndex-1][-1]='%.4f' % charge
                hTotal=0
        fOut.write('set: %s' % (setLabel))
        lastResType=''
        totalCharge=0
        for itemIndex in range(len(inData[setLabel])):
            (resType, atomName, typeLabel, atomType,
             chargeLabel, chargeStringIn) = inData[setLabel][itemIndex]
            if lastResType and lastResType!=resType:
                chargeStringOut='%.4f' % totalCharge
                if chargeStringOut=='-1.0000' or \
                   chargeStringOut=='0.0000'or \
                   chargeStringOut=='1.0000':
                    print 'Total Charge for %s %s: %s' % (setLabel, lastResType, chargeStringOut)
                else:
                    print '***Total Charge for %s %s: %s' % (setLabel, lastResType, chargeStringOut)
                totalCharge=0
            if printH or '0123456789H'.find(atomName[0])<0:
                s='%s %s %s %s %s %s' % tuple(outData[setLabel][itemIndex])
                fOut.write('     %s\n' % s)
            #print s
            totalCharge+=float(chargeStringIn)
            lastResType=resType
        if chargeStringOut=='-1.0000' or \
           chargeStringOut=='0.0000'or \
           chargeStringOut=='1.0000':
            print 'Total Charge for %s %s: %s' % (setLabel, lastResType, chargeStringOut)
        else:
            print '***Total Charge for %s %s: %s' % (setLabel, lastResType, chargeStringOut)

    fOut.close()

    return True

def parseCommandLine():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hHx:')
    HValue = 0
    xValue = 1.0 
    for option, parameter in opts:
        if option=='-h': usageError()
        if option=='-H': HValue = 1
        if option=='-x': xValue = float(parameter)
    return (args_proper, HValue, xValue)

def main():
    args_proper, printH, xValue = parseCommandLine()

    doIt(args_proper, printH, xValue)

    return

def usageError():
    print 'usage: %s inParamsfileName outParamsfileName' \
         % os.path.basename(sys.argv[0])
    sys.exit(1)

if __name__=='__main__':
    main()

