# =================================================================================
#    NucleicNet
#    Copyright (C) 2019-2022  Jordy Homing Lam, JHML. All rights reserved.
#    
#    Acknowledgement. 
#    JHML would like to thank Mingyi Xue and Joshua Chou for their patience and efforts 
#    in the debugging process and Prof. Xin Gao and Prof. Xuhui Huang for their 
#    continuous support.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    * Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
#    * Cite our work at Lam, J.H., Li, Y., Zhu, L. et al. A deep learning framework to predict binding preference of RNA constituents on protein surface. Nat Commun 10, 4941 (2019). https://doi.org/10.1038/s41467-019-12920-0
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================

import gc
import collections
from biopandas.pdb import PandasPdb
from biopandas.pdb import engines as PandasPdbEngines
from warnings import warn
import pandas as pd
import copy
import subprocess
from scipy.spatial import cKDTree
import itertools


class Sanitiser:
    
    def __init__(self, 
                DIR_InputPdbFile = "",
                DIR_SaveCleansed = "",
                DIR_SaveDataframe  = "",
                DIR_SaveUnzipped = "",
                ):
        """ 
        NOTE This script specifically handles pdb files with possibly 
        multiple states all written into a single pdb file.
        """
        self.DIR_SaveCleansed = DIR_SaveCleansed
        self.DIR_SaveDataframe = DIR_SaveDataframe
        self.DIR_InputPdbFile = DIR_InputPdbFile
        self.DIR_SaveUnzipped = DIR_SaveUnzipped
        self.pdbid = self.DIR_InputPdbFile.split("/")[-1].split(".")[0]

        # =======================================
        # Preliminary Inspections
        # =======================================
        ppdb_orig = PandasPdb()
        ppdb_orig.read_pdb(self.DIR_InputPdbFile)

        CHAINID_ALL = [i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuv1234567890']
        ppdb_orig_uniquechainid = ppdb_orig.df['ATOM']["chain_id"].unique().tolist()
        

        BioassemblyRequired = False
        try:
            others_df = ppdb_orig.df['OTHERS']

            # NOTE 1. ABORT Handles edge case where there are only CA
            if others_df['entry'].str.contains("CA ATOMS ONLY").sum() != 0:
                print("%s only has CA" %(self.pdbid))
                return

            # NOTE 2. ABORT Check if contain unknown resdues
            if ppdb_orig.df['ATOM']["residue_name"].str.contains("UNK").sum() != 0:
                print("%s contain UNK" %(self.pdbid))
                return   

            # NOTE 3. Handles symmetry group
            remark_df = others_df.loc[others_df['record_name'] == "REMARK"]
            if remark_df['entry'].str.contains("350   BIOMT1   2").sum() != 0: # NOTE Check if there are symmetry group operations other than identity
                print("%s requires bio assembly to be downloaded as .pdb1 and processed as .pdb2." %(self.pdbid))
                # NOTE 4tst is a reserved pdbid by PDB as a non-existent test bed.
                subprocess.call(
                    'rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp_data/biounit/PDB/divided/%s/%s.pdb1.gz %s/%s.pdb1.gz' 
                    %(self.pdbid[1:3],self.pdbid,self.DIR_SaveUnzipped,self.pdbid), 
                    shell = True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
                subprocess.call(
                    'gunzip %s/%s.pdb1.gz'%(self.DIR_SaveUnzipped,self.pdbid), 
                    shell = True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)

                BioassemblyRequired = False
                #self.DIR_InputPdbFile = self.DIR_SaveUnzipped + '/%s.pdb1'%(self.pdbid)
                # TODO a more careful assembly required

                with open(self.DIR_SaveUnzipped + '/%s.pdb1'%(self.pdbid),'r') as pdbfile_init:
                    self.pdbtxt = pdbfile_init.read()
                    self.pdbtxt_splitted = self.pdbtxt.split("ENDMDL")# NOTE This works even if there is only one state


                # ===========================================
                # Group the nucleotide protein copies
                # ============================================
                # TODO Get all "unit chain" of nucleotide.
                ppdb = PandasPdb()
                ppdb.pdb_text = self.pdbtxt_splitted[0]
                ppdb._df = ppdb._construct_df(pdb_lines=self.pdbtxt_splitted[0].splitlines(True))
                tempdf = ppdb._df['ATOM']
                UnitChainid_Nuc = sorted(set(tempdf.loc[tempdf['residue_name'].isin(['DA','DC','DG','DT','DU',
                                                        'A','C','G','T','U',])]['chain_id'].tolist()))

                grandpdbdf = []
                for pdbid_i in range(len(self.pdbtxt_splitted)):

                    ppdb = PandasPdb()
                    ppdb.pdb_text = self.pdbtxt_splitted[pdbid_i]
                    ppdb._df = ppdb._construct_df(pdb_lines=self.pdbtxt_splitted[pdbid_i].splitlines(True))

                    # NOTE THis handles lines after ENDMDL
                    if ppdb.df['ATOM'].shape[0] == 0:
                        continue   
                    tempdf = ppdb._df['ATOM']
                    tempdf.loc[:,'Nucleotide'] = tempdf['chain_id'].map(lambda x: x in UnitChainid_Nuc)
                    tempdf.loc[:,'TextID'] = pdbid_i
                    grandpdbdf.append(tempdf)
                    del ppdb
                    gc.collect()
                grandpdbdf = pd.concat(grandpdbdf, ignore_index= True)

                # TODO Determine how close are nucleotide copies. We will iterate (likely not covering) on the nearest in each text

                nucdf = grandpdbdf.loc[grandpdbdf["Nucleotide"] == True]
                nuctree = cKDTree(nucdf[["x_coord", "y_coord", "z_coord"]].values)

                prodf = grandpdbdf.loc[grandpdbdf["Nucleotide"] == False]
                protree = cKDTree(prodf[["x_coord", "y_coord", "z_coord"]].values)

                
                retrieval_pro_dict = {} # NOTE This maps nuc_textid to a list of (textid, chain_id) tuples corresponding to proteins closest to the key nuc_textid
                for pdbid_i in range(len(self.pdbtxt_splitted)):
                    tempnucdf = nucdf.loc[nucdf['TextID'] == pdbid_i]
                    temnucxyz = tempnucdf[["x_coord", "y_coord", "z_coord"]].values
                    # NOTE Locate the protein(s) nearby 10 angstrom for eahc of these nucleotide copies
                    proresult = protree.query_ball_point( temnucxyz, 10.0, p=2., eps=0, workers=1, return_sorted=None, return_length=False)
                    proresult = sorted(set(list(itertools.chain(*proresult))))

                    retrieval_pro_dict[pdbid_i] = sorted(set(zip(prodf.iloc[proresult]['TextID'].values.tolist(),
                                prodf.iloc[proresult]['chain_id'].values.tolist())))



                # TODO Indeed a better solution is a minimal spanning tree by taking distance as 1 - (intersect/ something?)
                #      Then we only need a bfs on this tree 
                # NOTE Find the next text nuc with largest overlaps in their nuc interacting protein tuples
                retrieval_nuc_dict = {}
                for pdbid_i in range(len(self.pdbtxt_splitted)):

                    set_i = set(retrieval_pro_dict[pdbid_i])

                    temp_overlap_dict = {}
                    for pdbid_j in range(len(self.pdbtxt_splitted)):
                        if pdbid_j == pdbid_i:
                            continue
                        set_j = set(retrieval_pro_dict[pdbid_j])


                        temp_overlap_dict[pdbid_j] = len(set_i & set_j)

                    nuc_candidates = sorted(temp_overlap_dict, key=temp_overlap_dict.get)[::-1]
                    #nuc_candidates = [i for i in nuc_candidates if i not in encountered_nuc]
                    retrieval_nuc_dict[pdbid_i] = nuc_candidates #nuc_candidates[0] #max(temp_overlap_dict, key = lambda k : temp_overlap_dict.get(k))




                # NOTE if assembled, then we can assume there are no multistate. w will work on the split text
                CHAINID_REMAIN = sorted(CHAINID_ALL)
                textchain_encountered = [] 
                temp_pdbtxt_splitted = []
                TEXTID_REMAIN = set(list(range(len(self.pdbtxt_splitted))))
                next_textid = 0
                while (len(CHAINID_REMAIN) > 0) & (len(TEXTID_REMAIN) > 0):
                    
                    selfdf = grandpdbdf.loc[grandpdbdf['TextID'] == next_textid]
                    pro_retrieved = retrieval_pro_dict[next_textid]


                    # renaming all self chain
                    for x, y in selfdf.groupby('chain_id', as_index=False):

                        tempdf = pd.DataFrame(y)
                        t = next_textid
                        c = tempdf['chain_id'].values[0]
                        #print(t,c, 'before')
                        if (t,c) in textchain_encountered:
                            continue
                        
                        #print(len(CHAINID_REMAIN)  ,len(TEXTID_REMAIN))
                        try:
                            tempdf.loc[:,'chain_id'] = CHAINID_REMAIN.pop(0)
                        except IndexError:
                            continue
                        #print(t,c, 'after')
                        tempdf = tempdf.drop(columns=['TextID', 'Nucleotide'])



                        temp_pdbtxt_splitted.append( self.UTIL_PdbDfDictToStr({'ATOM':tempdf}, RemoveLinesStartingWith = ["MODEL", "TER"]))
                        textchain_encountered.append((t,c))
                    #print("retrieve nearby")
                    # we will favor the protein on the argument that there are much more nonsite than site
                    for t, c in pro_retrieved:

                        if (t,c) in textchain_encountered:
                            continue
                        #print(t,c, 'before')
                        tempdf = grandpdbdf.loc[(grandpdbdf['TextID'] == t) & (grandpdbdf['chain_id'] == c)]
                        #print(len(CHAINID_REMAIN)  ,len(TEXTID_REMAIN))
                        try:
                            tempdf.loc[:,'chain_id'] = CHAINID_REMAIN.pop(0)
                        except IndexError:
                            continue
                        #print(t,c, 'after')
                        tempdf = tempdf.drop(columns=['TextID', 'Nucleotide'])
                        temp_pdbtxt_splitted.append( self.UTIL_PdbDfDictToStr({'ATOM':tempdf}, RemoveLinesStartingWith = ["MODEL", "TER"]))
                        textchain_encountered.append((t,c))

                    # Update 
                    TEXTID_REMAIN.remove(next_textid)
                    next_candidate = retrieval_nuc_dict[next_textid]
                    for i in next_candidate:
                        if i not in TEXTID_REMAIN:
                            continue
                        else:
                            next_textid = i
                            break

                
                # NOTE PandasPdb do not remove duplicates. 
                # We will manoever this property, but in the future we should rewrite PandasPdb TODO

                temp_pdbtxt_splitted = '\n'.join(temp_pdbtxt_splitted)
                #print(('nan' in temp_pdbtxt_splitted))
                ppdb = PandasPdb()
                ppdb.pdb_text = temp_pdbtxt_splitted
                ppdb._df = ppdb._construct_df(pdb_lines=temp_pdbtxt_splitted.splitlines(True))

                tempdf = ppdb.df['ATOM']
                # Prefilter Hydrogen to allow more atom to be written
                tempdf = tempdf.loc[tempdf['element_symbol'] != 'H']
                tempdf.loc[:,"atom_number"] = (tempdf.index + 1)

                
                
                # NOTE Pdb format can only hold 99999 atoms at max
                tempdf = tempdf.loc[tempdf["atom_number"] <= 99999]


                ppdb.df['ATOM'] = tempdf

                # NOTE 
                self.UTIL_PdbDfDictToStr(ppdb.df, RemoveLinesStartingWith = [], 
                                        DIR_Save = "%s/%s.pdb2"% (self.DIR_SaveUnzipped,self.pdbid),
                                        records = ["ATOM"])
                self.DIR_InputPdbFile = self.DIR_InputPdbFile + "2"

                del ppdb, tempdf
                gc.collect()

        except KeyError:
            print("Warning. %s supplied does not seem to orignate from Pdb." %(self.pdbid))
        
        # ===============================
        # Split the multistates
        # ===============================

        with open(self.DIR_InputPdbFile,'r') as pdbfile_init:
            self.pdbtxt = pdbfile_init.read()
            self.pdbtxt_splitted = self.pdbtxt.split("ENDMDL")


    def Sanitise(self, 
                SaveCleansed = True, SaveDf = True,
                Select_HeavyAtoms = True,
                Select_Resname = [
                    "A","T","C","G","U",
                    "DA","DT","DC","DG","DU",
                    "ALA","CYS","ASP","GLU","PHE","GLY", 
                    "HIS","ILE","LYS","LEU","MET","ASN", 
                    "PRO","GLN","ARG","SER","THR","VAL", 
                    "TRP","TYR"
                    ],
                ):
        # NOTE THis handles exits when UNK and CA are encountered
        try:
            self.pdbtxt_splitted
        except AttributeError:
            #print("ABORTED. There is no text left in %s. Check what is going wrong?" %(self.pdbid))
            return

        aux_id = 0
        for pdbid_i in range(len(self.pdbtxt_splitted)):

            # NOTE THis handles lines after ENDMDL
            if "ATOM" not in self.pdbtxt_splitted[pdbid_i]:
                continue
            #print(('nan' in self.pdbtxt_splitted[pdbid_i]))

            ppdb = PandasPdb()
            ppdb.pdb_text = self.pdbtxt_splitted[pdbid_i]
            ppdb._df = ppdb._construct_df(pdb_lines=self.pdbtxt_splitted[pdbid_i].splitlines(True))
            
            # NOTE THis handles lines after ENDMDL
            if ppdb.df['ATOM'].shape[0] == 0:
                continue          



            #print(ppdb.df['ATOM'])#[['x','y','z']])



            PART1_Filtering = True
            if PART1_Filtering:

                # NOTE Remove Hydrogens
                if Select_HeavyAtoms:
                    tempdf = ppdb.df['ATOM']
                    ppdb.df['ATOM'] = tempdf.loc[tempdf['element_symbol'] != 'H']

                # NOTE Select only certain residues
                if len(Select_Resname) > 0 :
                    tempdf = ppdb.df['ATOM']
                    ppdb.df['ATOM'] = tempdf.loc[tempdf["residue_name"].isin(Select_Resname)]


            PART2_HandlingAltLoc =True
            if PART2_HandlingAltLoc:
                # NOTE https://proteopedia.org/wiki/index.php/Alternate_locations
                tempdf = ppdb.df['ATOM']
                chainres_nonalt = sorted(set(list(map(tuple, tempdf.loc[(tempdf['alt_loc'] == '')][['chain_id',  'residue_number']].values.tolist()))))
                chainres_alt = sorted(set(list(map(tuple,tempdf.loc[(tempdf['alt_loc'] != '')][['chain_id',  'residue_number']].values.tolist()))))
                #print("halo1")
                #print(chainres_alt)
                if len(chainres_alt) > 0:
                    """
                    # NOTE BUG OBOSLETE. These are extreme edge cases...
                    # NOTE Check if chainres_alt is in non alt. 
                    #      if yes, skip for simplicity. This is incorrect?
                    chainres_alt_remaining = []
                    for i in chainres_alt:
                        if i in chainres_nonalt:
                            continue
                        chainres_alt_remaining.append(i)
                    chainres_alt = chainres_alt_remaining
                    
                    # NOTE in some very old pdb entries e.g. 1zir they actually have a preferred loc stored.
                    if len(chainres_alt) == 0:
                        tempdf = ppdb.df['ATOM']
                        tempdf = tempdf.loc[tempdf['alt_loc'].isin([""])]
                        ppdbtemp = copy.deepcopy(ppdb)

                        ppdbtemp.df['ATOM'] = tempdf

                        if SaveDf:
                            ppdbtemp.df['ATOM'].to_pickle('%s/%s%08d.pkl'%(self.DIR_SaveCleansed, self.pdbid, aux_id), 
                                                        compression='infer', protocol=4, storage_options=None)
                        if SaveCleansed:
                            #ppdbtemp.to_pdb("%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id), 
                            #                    records=['ATOM'], gz=False, append_newline=True)
                            self.UTIL_PdbDfDictToStr(ppdbtemp.df, RemoveLinesStartingWith = [], 
                                            DIR_Save = "%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id),
                                            records = ["ATOM"])


                        del ppdbtemp
                        gc.collect()
                        aux_id +=1
                        continue
                    """



                    # NOTE what remains are those that have no duplicate in the non-alt. 
                    chainresalt_alt = sorted(set(list(map(tuple, tempdf.loc[(tempdf['alt_loc'] != '')][['chain_id',  'residue_number', 'alt_loc']].values.tolist()))))
                    # NOTE Check the assumption that all are paired up. 
                    check_altconsistency = collections.defaultdict(list)
                    for i in chainresalt_alt:
                        check_altconsistency[(i[0],i[1])].append(i[2])
                    #print(set(list(map(tuple,check_altconsistency.values()))))
                    #print(set(list(map(tuple,check_altconsistency.values()))))
                    altconsistent = (len(sorted(set(list(map(tuple,check_altconsistency.values()))))) == 1)
                    
                    #for i in sorted(set(list(map(tuple,check_altconsistency.values())))):
                    all_altindex = sorted(set(list(map(tuple,check_altconsistency.values()))))
                    all_altindex_ = [set(i) for i in all_altindex]
                    altcommon = set.intersection(*all_altindex_)
                    #print(set.intersection(*[set(i) for i in [("A","B"), ("A", "B"), ("A","B","C")]]))
                    #print(altcommon)
                    if len(altcommon) == 0:
                        altcounter = collections.Counter(all_altindex)
                        altcommon = altcounter.most_common(1)[0][0]
                        print("%s cannot find a common set of alternative location. Consider removal? We will proceed with the largest common set" %(self.pdbid), altcounter)
                        #print(set(altcommon))
                        #altcommon = [list(i) for i in altcommon]
                        altcommon = set(altcommon)
                        
                    
                    for altchoice in altcommon:
                        #print(altchoice)
                        #print("halo")
                        tempdf = ppdb.df['ATOM']
                        tempdf = tempdf.loc[tempdf['alt_loc'].isin(["", altchoice])]

                        ppdbtemp = copy.deepcopy(ppdb)

                        ppdbtemp.df['ATOM'] = tempdf

                        if SaveDf:
                            ppdbtemp.df['ATOM'].to_pickle('%s/%s%08d.pkl'%(self.DIR_SaveCleansed, self.pdbid, aux_id), 
                                                        compression='infer', protocol=4, storage_options=None)
                        if SaveCleansed:
                            #ppdbtemp.to_pdb("%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id), 
                            #                    records=['ATOM'], gz=False, append_newline=True)
                            self.UTIL_PdbDfDictToStr(ppdbtemp.df, RemoveLinesStartingWith = [], 
                                            DIR_Save = "%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id),
                                            records = ["ATOM"])
                        del ppdbtemp
                        gc.collect()
                        aux_id +=1

                    continue
                    

            PART2_SaveSeparate = True
            if not  ppdb.df['ATOM'].empty:
                # NOTE we will only save the atom record for simplicity.
                #"""
                if SaveDf:
                    ppdb.df['ATOM'].to_pickle('%s/%s%08d.pkl'%(self.DIR_SaveCleansed, self.pdbid, aux_id), 
                                                           compression='infer', protocol=4, storage_options=None)
                if SaveCleansed:
                    #ppdb.to_pdb("%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id), 
                    #                                records=['ATOM'], gz=False, append_newline=True)
                    self.UTIL_PdbDfDictToStr(ppdb.df, RemoveLinesStartingWith = [], 
                                            DIR_Save = "%s/%s%08d.pdb"% (self.DIR_SaveCleansed,self.pdbid,aux_id),
                                            records = ["ATOM"])
                #"""
                pass

            aux_id +=1
            del ppdb
            gc.collect()
        return


    def Minimise(self):
        return
    def Regularise(self):
        return

    # =========================
    # Misc. Some light utils
    # =========================
    def UTIL_PdbDfDictToStr(self, df, RemoveLinesStartingWith = [], DIR_Save = "", records = []):

        """
        https://github.com/rasbt/biopandas/blob/main/biopandas/pdb/pandas_pdb.py
        """
        if len(records) == 0:
            records = df.keys()

        dfs = {r: df[r].copy() for r in records if not df[r].empty}

        for r in dfs.keys():
            for col in PandasPdbEngines.pdb_records[r]:
                dfs[r][col['id']] = dfs[r][col['id']].apply(col['strf'])
                dfs[r]['OUT'] = pd.Series('', index=dfs[r].index)

            for c in dfs[r].columns:
                # fix issue where coordinates with four or more digits would
                # cause issues because the columns become too wide
                if c in {'x_coord', 'y_coord', 'z_coord'}:
                    for idx in range(dfs[r][c].values.shape[0]): 
                        if len(dfs[r][c].values[idx]) > 8: # NOTE 8 digits including decimal point
                            dfs[r][c].values[idx] = \
                                str(dfs[r][c].values[idx]).strip()
                if c in {'line_idx', 'OUT'}:
                    pass
                elif r in {'ATOM', 'HETATM'} and c not in PandasPdbEngines.pdb_df_columns:
                    warn('Column %s is not an expected column and'
                         ' will be skipped.' % c)
                else:
                    dfs[r]['OUT'] = dfs[r]['OUT'] + dfs[r][c]


        df = pd.concat(dfs, sort=False)
        df.sort_values(by='line_idx', inplace=True)

        s = df['OUT'].tolist()
        for idx in range(len(s)):
            if len(s[idx]) < 80:
                s[idx] = '%s%s' % (s[idx], ' ' * (80 - len(s[idx])))
        if len(RemoveLinesStartingWith) > 0:
            s = [i for i in s if not str(i).startswith(tuple(RemoveLinesStartingWith))]

        to_write = '\n'.join(s)
        del df, dfs
        gc.collect()

        if len(DIR_Save) > 0:
            with open(DIR_Save, 'w') as f:

                for idx in range(len(s)):
                    if len(s[idx]) < 80:
                        s[idx] = '%s%s' % (s[idx], ' ' * (80 - len(s[idx])))

                to_write = '\n'.join(s)
                f.write(to_write)
                f.write('\n')



        return to_write


# =================================================================================
#    NucleicNet
#    Copyright (C) 2019-2022  Jordy Homing Lam, JHML. All rights reserved.
#    
#    Acknowledgement. 
#    JHML would like to thank Mingyi Xue and Joshua Chou for their patience and efforts 
#    in the debugging process and Prof. Xin Gao and Prof. Xuhui Huang for their 
#    continuous support.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    * Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation and/or 
#    other materials provided with the distribution.
#    * Cite our work at Lam, J.H., Li, Y., Zhu, L. et al. A deep learning framework to predict binding preference of RNA constituents on protein surface. Nat Commun 10, 4941 (2019). https://doi.org/10.1038/s41467-019-12920-0
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==================================================================================