#! /usr/bin/env python
# $Id: atomselector.py,v 1.5 2004/09/21 22:37:47 mliang Exp $

# Selects atom coordinates from pdb file
# Output:
#   PDBID X Y Z # ALA123A:C@OG'1
import sys,os
import getopt
import re
from cStringIO import StringIO
from simplepdb import PDBAtom
from pdbutils import goodpdbfilename
from utils import openfile
from logutils import *

True = 1
False = 0
DEFAULT_VERBOSITY = 0
DEFAULT_GLOBAL_RESNAME = None
DEFAULT_GLOBAL_CHAINID = None
DEFAULT_GLOBAL_NAME = None
DEFAULT_GLOBAL_ALTLOC = ",A,1"

def eusage():
    print "Usage: %s [OPTIONS] ENTRIES..." % os.path.basename(sys.argv[0])
    print """
Options:
    -r RESNAME
        Sets global resname
    -c CHAINID
        Sets global chainid
    -a NAME
        Sets global atomname
    -f FILENAME
        Reads Entries from FILENAME

    entry = pdbid|resName|chainId|name|altLoc
"""
    sys.exit(0)

class Entry:
    attrs = ['pdbid','resName','chainId','name','altLoc']
    def __init__(self,field=None,defaults={}):
        self.pdbid = None
        self.constraints = {}
        if field:
            self.parse(field,defaults)

    def parse(self,field,defaults={}):
        def splitfield(field):
            if not field:
                return []
            return [s.strip() for s in field.lower().split(',')]

        fields = [s.strip() for s in field.split('|')]
        debug(2,"Fields:",fields)
        self.pdbid = fields[0]
        for idx in range(1,len(fields)):
            attr = self.attrs[idx]
            values = splitfield(fields[idx])
            if values:
                self.constraints[attr] = values
                debug(2,"Value[%s]:" % attr,values)
        for attr in self.attrs:
            if not self.constraints.has_key(attr):
                values = splitfield(defaults.get(attr))
                if values:
                    self.constraints[attr] = values
                    debug(2,"DefaultValue[%s]:" % attr,values)

    def accept(self,atom):
        for attr,values in self.constraints.items():
            aval = getattr(atom,attr).lower()
            if aval not in values:
                return False
        return True

    def __str__(self):
        outfh = StringIO()
        print >>outfh, "pdbid:",self.pdbid
        for attr in self.attrs[1:]:
            print >>outfh, "%s: %s" % (attr, self.constraints.get(attr,None))
        return outfh.getvalue()

def OutputAtom(pdbid,atom):
    x,y,z = atom.getCoord()
    idstring = atom.fullIdString()
    print '%(pdbid)s\t%(x)s\t%(y)s\t%(z)s\t#\t%(idstring)s' % locals()

args = sys.argv[1:]
user_entries = []
verbosity = DEFAULT_VERBOSITY
globalResname = DEFAULT_GLOBAL_RESNAME
globalChainid = DEFAULT_GLOBAL_CHAINID
globalName = DEFAULT_GLOBAL_NAME
globalAltloc = DEFAULT_GLOBAL_ALTLOC
try:
    optlist,args = getopt.getopt(args,"vr:c:a:f:l:")
except getopt.GetoptError:
    eusage()
for opt,arg in optlist:
    if opt in ['-v']:
        verbosity += 1
        setdebuglevel(verbosity)
    elif opt in ['-r']:
        globalResname = arg
    elif opt in ['-c']:
        globalChainid = arg
    elif opt in ['-a']:
        globalName = arg
    elif opt in ['-l']:
        globalAltloc = arg
    elif opt in ['-f']:
        user_entries.extend([l.strip() for l in file(arg)])
user_entries.extend(args)
if not user_entries:
    user_entries.extend([l.strip() for l in sys.stdin])

defaults = {
    'resName': globalResname,
    'chainId': globalChainid,
    'name': globalName,
    'altLoc': globalAltloc
}

entryMap = {}
for line in user_entries:
    entry = Entry(line,defaults)
    debug(1,entry)
    entryMap.setdefault(entry.pdbid,[]).append(entry)
pdbids = entryMap.keys()
pdbids.sort()

for pdbid in pdbids:
    debug(1,pdbid)
    pdbfilename = goodpdbfilename(pdbid)
    if not pdbfilename:
        warning('Could not find PDB file for',pdbid)
        continue
    debug(2,pdbfilename)
    entries = entryMap[pdbid]

    infh = openfile(pdbfilename)
    for line in infh:
        rectype = line[:6].strip()
        if rectype in ['ENDMDL']:
            break
        if rectype not in ['ATOM','HETATM']:
            continue
        atom = PDBAtom(line)

        for entry in entries:
            if entry.accept(atom):
                OutputAtom(pdbid,atom)

