#! /usr/bin/env python

# Filters feature .point files by a chainid file

import sys,os
import re
from utils import chomp

def eusage():
    print "Usage: %s POINTFILE CHAINFILE" % os.path.basename(sys.argv[0])
    sys.exit(0)

class ChainFileEntry:
    def __init__(self,line=None):
        if line:
            self.parse(line)

    def parse(self,line):
        self.pdbid = line[:4].lower()
        self.chainid = line[4:5].lower()
        if self.chainid == '_':
            self.chainid = ''

class ChainFile:
    def __init__(self,filename=None):
        self.entries = []
        self.entriesByPdbid = {}
        if filename:
            self.read(filename)

    def read(self,filename):
        infh = file(filename)
        for line in infh:
            entry = ChainFileEntry(line)
            self.entries.append(entry)
            self.entriesByPdbid.setdefault(entry.pdbid,[]).append(entry)

    def isValid(self,pointEntry):
        for entry in self.entriesByPdbid.get(pointEntry.pdbid,[]):
            if entry.chainid == pointEntry.chainid or entry.chainid == '.':
                return True
        return False

COMMENT_RE = re.compile(r"(?P<resname>\w{3})(?P<resnum>-?\d+)(?P<icode>\w)?:(?P<chainid>\w)?@(?P<atomname>\w+)('(?P<altloc>\w))?")
class PointFileEntry:
    def __init__(self,line=None):
        if line:
            self.parse(line)

    def parse(self,line):
        line = chomp(line)
        self.line = line
        fields = line.split('\t')
        self.pdbid = fields[0].lower()
        self.chainid = ''
        self.comment = fields[-1]
        match = COMMENT_RE.search(self.comment)
        if match:
            self.chainid = match.group('chainid')
            if self.chainid:
                self.chainid = self.chainid.lower()

    def __str__(self):
        return self.line

class PointFile:
    def __init__(self,filename=None):
        self.entries = []
        if filename:
            self.read(filename)

    def read(self,filename):
        infh = file(filename)
        for line in infh:
            entry = PointFileEntry(line)
            self.entries.append(entry)

    def filterby(self,chainfile):
        for entry in self.entries:
            if chainfile.isValid(entry):
                yield entry


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        eusage()
    pointfilename = args[0]
    chainfilename = args[1]


    chainfile = ChainFile(chainfilename)
    pointfile = PointFile(pointfilename)
    for entry in pointfile.filterby(chainfile):
        print entry

if __name__ == '__main__':
    main()
