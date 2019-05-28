
# Converts protein_residues and protein_amber files into .cc files

def convert_residues():
    import protein_residues

    for entrytype in ('normal', 'n_terminal', 'c_terminal'):
        print 'set:', entrytype
        entry = eval('protein_residues.%s' % entrytype)
        entry_items = entry.items()
        entry_items.sort()
        for (resname, resattrs) in entry_items:
            print ' '*4, 'residue:', resname
            print ' '*8, 'aliases:'
            aliases_items = resattrs['aliases'].items()
            aliases_items.sort()
            for (key, val) in aliases_items:
                print ' '*12, key, val
        print

ignoreList = (
    )

def convert_forcefield():
    import protein_amber
    typeSet = {}

    for entrytype in ('normal', 'n_terminal', 'c_terminal'):
        print 'set:', entrytype
        entry = eval('protein_amber.%s' % entrytype)
        entry_items = entry.items()
        entry_items.sort()
        for (key, val) in entry_items:
            type, charge = val['type'], val['charge']
            # Ignore Hydrogens and 'CT' and 'C'
            if type[0] == 'H':
                continue
            # Convert aromatics carbons to one type 'Ca'
            if type in ('CA', 'CB', 'CC', 'CN', 'CR', 'CW', 'C*', 'CK', 'CM', 'CQ', 'CV'):
                type = 'Ca'
            if type in ('NA', 'NB', 'NC', 'N*'):
                type = 'Na'
            print " "*4, key[0], key[1], 'type:', type, 'charge:', charge
            typeSet[type] = typeSet.get(type,0) + 1;
    return typeSet

ts = convert_forcefield()
#items = ts.items()
#items.sort()
#for (t,v) in items:
#    print t,v

