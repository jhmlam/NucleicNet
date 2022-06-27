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

import subprocess
import os
#from argparse import ArgumentParser
import sys
import re
import pandas as pd
#from io import StringIO
import urllib.request
import glob
import gzip
import multiprocessing
from functools import partial
import tqdm
from biopandas.pdb import PandasPdb

from scipy import spatial
import itertools


from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandReadPdbFtp import *
from NucleicNet.DatasetBuilding.commandCoordinate import Sanitiser

def OOC_SanitiseCoordinateData(f, CleansedPdbFolder = "", UnzippedPdbFolder = ""):

    CoordinateSanitiserC = Sanitiser(DIR_InputPdbFile = f,
                    DIR_SaveCleansed = CleansedPdbFolder,
                    DIR_SaveDataframe  = CleansedPdbFolder, # NOTE This contains the nucleicacid bound coordinate dataframe
                    DIR_SaveUnzipped= UnzippedPdbFolder)
    CoordinateSanitiserC.Sanitise(SaveCleansed = True, SaveDf = True,
                Select_HeavyAtoms = True,
                Select_Resname = [
                    "A","T","C","G","U",
                    "DA","DT","DC","DG","DU",
                    "ALA","CYS","ASP","GLU","PHE","GLY", 
                    "HIS","ILE","LYS","LEU","MET","ASN", 
                    "PRO","GLN","ARG","SER","THR","VAL", 
                    "TRP","TYR"
                    ],)
    return
            

def OOC_MakeApoCoordinate(f, ApoPdbFolder = ""):
    pdbid = f.split("/")[-1].split(".")[0]
    if os.path.exists("%s/%s.pdb" %(ApoPdbFolder, pdbid)):
        return

    ppdb = PandasPdb()
    CurrentPdbStructure = ppdb.read_pdb(f)
    # Keep Protein 
    CurrentPdbStructure.df['ATOM'] = \
        CurrentPdbStructure.df['ATOM'][
            CurrentPdbStructure.df['ATOM']["residue_name"].isin(
                ["ALA","CYS","ASP","GLU","PHE","GLY", 
                    "HIS","ILE","LYS","LEU","MET","ASN", 
                    "PRO","GLN","ARG","SER","THR","VAL", 
                    "TRP","TYR"
                ]
                )
                ]

    # Save the Copied Pdb
    if not CurrentPdbStructure.df['ATOM'].empty:
        CurrentPdbStructure.to_pdb(
            path="%s/%s.pdb" %(ApoPdbFolder, pdbid), 
            records=['ATOM'])

    del ppdb, CurrentPdbStructure
    gc.collect()
    return

def OOC_MakeDssp(f, DsspFolder = "", DsspExecutable = ""):
    subprocess.call("%s -i %s -o %s/%s.dssp" %(
        DsspExecutable,
        f,
        DsspFolder,
        f.split("/")[-1].split(".")[0]
        ), shell = True)
    return


def OOC_FindProteinChainBoundWithNuc(kk):

      from biopandas.pdb import PandasPdb
      pdbid = kk[0]
      unzipped = kk[1]

      #print(pdbid)
      ppdb = PandasPdb()
      CurrentPdbStructure = ppdb.read_pdb("%s/%s.pdb" %(unzipped, str(pdbid)))
      all_df = CurrentPdbStructure.df['ATOM']
      # DNA Chain
      DNAChain = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(["DA","DT","DC","DG","DU"])]["chain_id"].tolist()
      #DNAChain = ','.join(sorted(set(DNAChain)))
      DNAChain = sorted(set(DNAChain))
      if not DNAChain:
         DNAChain= []

      # RNA Chain
      RNAChain = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U"])]["chain_id"].tolist()
      RNAChain = sorted(set(RNAChain))
      #RNAChain = ','.join(sorted(set(RNAChain)))
      if not RNAChain:
         RNAChain= []

      NAChain = RNAChain+DNAChain

      # NOTE We look for any chain within 5 angstrom of NAChain (exclude self). These are the bound chain
      all_tree = spatial.cKDTree(all_df[['x_coord', 'y_coord', 'z_coord']].values)
      NA_xyz = all_df.loc[all_df['chain_id'].isin(NAChain)][['x_coord', 'y_coord', 'z_coord']].values

      lol = all_tree.query_ball_point(NA_xyz, 5.0, p=2., eps=0, workers=1, return_sorted=None, return_length=False)
      nearNA_index = sorted(set(itertools.chain(*lol)))



      return (pdbid, sorted(set(all_df.iloc[nearNA_index]['chain_id'].unique()) - set(NAChain) ))
      #print(nearNA_index)


class SyncPdbData():
    def __init__(self, DerivedDataFolder = "Database-PDB/DerivedData", 
                       n_MultiprocessingWorkers = 16, 
                       UnzippedPdbFolder = "Database-PDB/unzipped",
                       CleansedPdbFolder = "Database-PDB/cleansed",
                       ApoPdbFolder = "Database-PDB/apo",
                       DsspFolder = "Database-PDB/dssp",
                       DsspExecutable = "../util/dssp",
                       exclusion = ["pdb_chain_go.lst"]):
        self.exclusion = exclusion
        self.DerivedDataFolder = DerivedDataFolder
        self.n_MultiprocessingWorkers = n_MultiprocessingWorkers
        self.UnzippedPdbFolder = UnzippedPdbFolder
        self.ApoPdbFolder = ApoPdbFolder
        self.DsspFolder = DsspFolder
        self.DsspExecutable = DsspExecutable
        self.CleansedPdbFolder = CleansedPdbFolder

        MkdirList([self.UnzippedPdbFolder, self.DerivedDataFolder, 
                   self.ApoPdbFolder,self.CleansedPdbFolder, self.DsspFolder])


    def UpdateDerivedData(self, PART1_Downloading = False, 
                            ExcludeShortPeptideLength = 30, ExcludeShortNucleotideLength = 4,
                            ExcludeChemicalComponentList = ["1MG","5MC","H2U","5MU","PSU"],
                            ExcludePdbidList = [],
                            RedownloadMmseqClust = False):

        #PART1_Downloading = True
        if PART1_Downloading:
            
            # Download NPIDB data NOTE Dependency removed as the server no longer work
            #print("Downloading NPIDB data")
            #urllib.request.urlretrieve('http://npidb.belozersky.msu.ru/data/pdb_new/lists/ListOfComplexes.txt', 
            #                        '%s/NPIDB_ListOfComplexes.txt'%(self.DerivedDataFolder))

            # Download all derived data of the Pdb.

            # NOTE pdb_entry_type.txt: Experiment type, Nucleic or Protein. TODO resolut.idx: Resolution
            print("Downloading Derived Data from PDB")
            subprocess.call('rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp/derived_data/pdb_* %s' %(self.DerivedDataFolder), shell = True)
            subprocess.call('rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp/derived_data/index/*  %s' %(self.DerivedDataFolder), shell = True)

            # NOTE cc-to-pdb.tdd: Map of Ligand to Pdb Entries 
            print("Downloading Ligand Expo Data")
            urllib.request.urlretrieve('http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd', 
                                    '%s/cc-to-pdb.tdd'%(self.DerivedDataFolder))

            # NOTE Components-smiles-stereo-oe.smi/Components-smiles-oe.smi: Smiles of ligands
            urllib.request.urlretrieve('http://ligand-expo.rcsb.org/dictionaries/Components-smiles-oe.smi', 
                                    '%s/Components-smiles-oe.smi'%(self.DerivedDataFolder))
            urllib.request.urlretrieve('http://ligand-expo.rcsb.org/dictionaries/Components-smiles-stereo-oe.smi', 
                                        '%s/Components-smiles-stereo-oe.smi'%(self.DerivedDataFolder))
            urllib.request.urlretrieve('http://ligand-expo.rcsb.org/dictionaries/cc-counts-extra.tdd', 
                                        '%s/cc-counts-extra.tdd'%(self.DerivedDataFolder))

            




            # =======================================
            # Sequence Homology Clusters
            # ==========================================
            # NOTE BlastClust ID for individual chain in pdb entries. 
            # NOTE OBSOLETE https://groups.google.com/a/rcsb.org/g/api/c/ALLI4pouK_w?pli=1
            #print("Downloading BlastClust ID")
            #for bcnum in [30,40,50,70,90,95,100]:
            #    urllib.request.urlretrieve("https://cdn.rcsb.org/resources/sequence/clusters/bc-%s.out" %(bcnum),
            #                                "%s/bc-%s.out" %(self.DerivedDataFolder, bcnum))



            if RedownloadMmseqClust:
                print("WARNING. Download and translation of Mmseq2 cluster can take up to 800 minutes to complete.")
                # NOTE Mmseq2
                for bcnum in [30,40,50,70,90,95,100]:
                    urllib.request.urlretrieve("https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-%s.txt" %(bcnum),
                                                "%s/mmseqraw-%s.out" %(self.DerivedDataFolder, bcnum))

                User_maxtrial = 500
                # NOTE This is global precompiled entityid chainid dict
                if os.path.exists("%s/EntityidChainidDict.pkl" %(self.DerivedDataFolder)):
                    with open("%s/EntityidChainidDict.pkl" %(self.DerivedDataFolder), 'rb') as f:
                        CulmulatedEntityChainDict = pickle.load(f)
                else:
                    CulmulatedEntityChainDict = defaultdict(list)

                
                for Mmseqnum in [30,40,50,70,90,95,100]:
                    MmseqRawDict, MmseqRawDictPdbid = ReadMmseqClust("%s/mmseqraw-%s.out" %(self.DerivedDataFolder, Mmseqnum))
                    MmseqDict = defaultdict(list)
                    FailingClusters = list(MmseqRawDict.keys())
                    for trial in range(User_maxtrial):
                        MmseqDict, FailingClusters, CulmulatedEntityChainDict = RecurringEntityIDTranslation(MmseqRawDict, 
                                                            MmseqDict = MmseqDict, 
                                                            FailingClusters = FailingClusters,
                                                            CulmulatedEntityChainDict = CulmulatedEntityChainDict)


                    with open("%s/mmseq-%s.out" %(self.DerivedDataFolder, Mmseqnum), 'w') as f:
                        for clusterid in sorted(MmseqDict.keys()):
                            f.write('%s\n' %(' '.join(MmseqDict[clusterid])))


                #import pickle
                #with open("%s/EntityidChainidDict.pkl" %(self.DerivedDataFolder), 'wb') as f:
                #    pickle.dump(CulmulatedEntityChainDict,f, protocol=4)
                if len(FailingClusters) > 0:
                    print("WARNING. There are %s failing cluster we failed to map with entity id or chain id" %(len(FailingClusters) ))

            print("Downloading uniprot pdb chain mapping")
            # NOTE pdb_chain_uniprot.lst : pdb chain to uniprot mapping 
            FtpDirectoryDownloading("ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/text/",
                                    self.DerivedDataFolder , exclusion = self.exclusion)
            
            # TODO protmod.tsv.gz: Pdb with modified residues
            #FtpDirectoryDownloading("ftp://resources.rcsb.org/protmod/", self.DerivedDataFolder)



        PART2_MakeDataframe = True
        if PART2_MakeDataframe:
            
            print("Making derived data")
            DateTitle = ReadDateTitle(self.DerivedDataFolder)
            #Title = ReadTitle(self.DerivedDataFolder)
            PubmedID = ReadPubmedid(self.DerivedDataFolder)
            NPIDB = ReadNPIDB_ListOfComplexes(self.DerivedDataFolder)
            #solvent = ReadSolvent(self.DerivedDataFolder)
            #DfLigandToPdbids, DictPdbidToLigands, DfPdbidToLigNum =  ReadLigandPerPdb(self.DerivedDataFolder,solvent)
            DfComplete = ReadPdbEntryType(self.DerivedDataFolder)
            DfResolu = ReadResolu(self.DerivedDataFolder)
            print("Reading SeqResHeader")
            DfSeqResHeader, DfStatChainlength = ReadSeqResHeader(self.DerivedDataFolder)
            Df_CC2PdbTdd, Dict_Pdb2CCTdd = ReadCC2PdbTdd(self.DerivedDataFolder)
            


            print("Merging")
            DfGrand = pd.merge(DfComplete, DfResolu, how='outer', on = 'Pdbid')
            DfGrand.loc[:,'ChemicalComponent'] = DfGrand['Pdbid'].map(Dict_Pdb2CCTdd)
            DfGrand = pd.merge(DfGrand, DfStatChainlength, how='outer', on = 'Pdbid')
            # NOTE fused polymer cc is not counted as LigNum
            #DfGrand = pd.merge(DfGrand, DfPdbidToLigNum, how='outer', on = 'Pdbid')
            DfGrand = pd.merge(DfGrand, NPIDB, how='outer', on = 'Pdbid')
            DfGrand = pd.merge(DfGrand, PubmedID, how='outer', on = 'Pdbid')
            DfGrand = pd.merge(DfGrand, DateTitle, how='outer', on = 'Pdbid')

            print("Making BC")
            BC = ReadBCInternalSymmetry(self.DerivedDataFolder)
            DfGrand = pd.merge(DfGrand, BC, how='outer', on = 'Pdbid')

            print("Making Mmseq")
            MMSEQ = ReadMmseqInternalSymmetry(self.DerivedDataFolder)
            DfGrand = pd.merge(DfGrand, MMSEQ, how='outer', on = 'Pdbid')







            DfGrand = DfGrand.reset_index(drop = True)
            DfGrand = DfGrand.drop_duplicates(subset=["Pdbid"])
            DfGrand = DfGrand.reset_index(drop = True)



            if ExcludeShortPeptideLength > 0:
                print("Filtering Short Peptide")
                protein_keeplen = DfGrand.loc[DfGrand["MeanChainLength_Peptide"] >= ExcludeShortPeptideLength -1]["Pdbid"].tolist()
                DfGrand = DfGrand.loc[DfGrand["Pdbid"].isin(protein_keeplen)]
                DfGrand = DfGrand.reset_index(drop = True)

            if ExcludeShortNucleotideLength > 0:
                print("Filtering Short Nucleotide")
                na_keeplen = DfGrand.loc[DfGrand["MeanChainLength_Peptide"] >= ExcludeShortNucleotideLength -1]["Pdbid"].tolist()
                DfGrand = DfGrand.loc[DfGrand["Pdbid"].isin(na_keeplen)]
                DfGrand = DfGrand.reset_index(drop = True)

            if len(ExcludeChemicalComponentList) > 0:
                print("Filtering Chemical Component")
                Pdbid_containing_CC = []
                for chem_i in ExcludeChemicalComponentList:
                    try:
                        Pdbid_containing_CC.extend(sorted(Df_CC2PdbTdd.loc[Df_CC2PdbTdd["ChemicalComponent"] == chem_i]["Pdbids"].str.split(" ").values.tolist()[0]))
                    except IndexError:
                        print("%s not found" %(chem_i))


                #print(Pdbid_containing_CC)
                Pdbid_containing_CC = sorted(set(Pdbid_containing_CC))
                Pdbid_containing_CC = [i for i in Pdbid_containing_CC if len(i) == 4]
                DfGrand = DfGrand.loc[~(DfGrand["Pdbid"].isin(Pdbid_containing_CC))]
                DfGrand = DfGrand.reset_index(drop = True)

            if len(ExcludePdbidList) > 0:
                print("Pdbid Exclusion")
                print(ExcludePdbidList)
                ExcludePdbidList = [str(i).lower() for i in ExcludePdbidList]
                DfGrand = DfGrand.loc[~(DfGrand["Pdbid"].isin(ExcludePdbidList))]
                DfGrand = DfGrand.reset_index(drop = True)                

        DfGrand = DfGrand.reset_index(drop = True)
        DfGrand = DfGrand.drop_duplicates(subset=["Pdbid"])
        DfGrand = DfGrand.reset_index(drop = True)

        DfGrand.to_pickle('%s/DataframeGrand.pkl'%(self.DerivedDataFolder), compression='infer', protocol=4, storage_options=None)


    def UpdateCoordinateData(self, PdbidList = []):
        if len(PdbidList) == 0:
            print("ABORTED. Supply a list of PdbidList")
            return
        for i in tqdm.tqdm(PdbidList):
            if not os.path.exists("%s/%s.pdb" %(self.UnzippedPdbFolder, i)):
                try:
                    subprocess.call('rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp_data/structures/divided/pdb/%s/pdb%s.ent.gz %s/%s.pdb.gz' %(i[1:3],i,self.UnzippedPdbFolder,i), shell = True)
                    subprocess.call('gunzip %s/%s.pdb.gz'%(self.UnzippedPdbFolder,i), shell = True) 
                    #shutil.move("%s/%s.pdb" %(UnzippedPdbFolder,i), "%s/%s.pdb"%(UnzippedPdbFolder,i))
                except FileNotFoundError:
                    print("Missing File %s. Maybe obsolete or having more than 62 chains or 999999 atoms." %(i))

    # TODO Separate this from making Apo.
    def MakeSanitiseCoordinate(self, DIR_folder = None):
        if DIR_folder is None:
            readfolder = self.UnzippedPdbFolder
        else:
            readfolder = DIR_folder

        filelist_ = glob.glob("%s/*.pdb" %(readfolder))
        filelist_alreadysanitised = glob.glob("%s/*00000000.pdb" %(self.CleansedPdbFolder))
        filelist_alreadysanitised = [i.split("/")[-1][:4] for i in filelist_alreadysanitised]
        filelist = []
        for i in filelist_:
            if i.split("/")[-1][:4] in filelist_alreadysanitised:
                continue
            # NOTE check on specific pdbid cases 5w1c 6gv4 4fte 4g0r
            #if i.split("/")[-1][:4] == "6gv4":
            #    pass
            #else:
            #    continue

            filelist.append(i)

        if len(filelist) == 0:
            return

        OOC_SanitiseCoordinateData_partial = partial(OOC_SanitiseCoordinateData, 
                                CleansedPdbFolder = self.CleansedPdbFolder,
                                UnzippedPdbFolder = self.UnzippedPdbFolder)

        pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
        pool.map(OOC_SanitiseCoordinateData_partial, filelist)




    def MakeApoCoordinate(self, DIR_folder = None):

        if DIR_folder is None:
            readfolder = self.CleansedPdbFolder
        else:
            readfolder = DIR_folder

        filelist = glob.glob("%s/*.pdb" %(readfolder))
        
        if len(filelist) == 0:
            return

        OOC_MakeApoCoordinate_partial = partial(OOC_MakeApoCoordinate, 
                                ApoPdbFolder = self.ApoPdbFolder)

        pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
        pool.map(OOC_MakeApoCoordinate_partial, filelist)


        return


    def MakeDssp(self, DIR_folder = None):
        if DIR_folder is None:
            readfolder = self.ApoPdbFolder
        else:
            readfolder = DIR_folder
        filelist = glob.glob("%s/*.pdb" %(readfolder))

        if len(filelist) == 0:
            return      

        OOC_MakeDssp_partial = partial(OOC_MakeDssp, 
                                            DsspFolder = self.DsspFolder, DsspExecutable = self.DsspExecutable)

        pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
        pool.map(OOC_MakeDssp_partial, filelist)

    # ====================
    # MISC
    # ====================
    def FindProteinChainBoundWithNuc(self):
            # NOTE Rerun the script after you downloaded some pdb files
            # TODO Possibly supplement with other information that can be coordinate derived.
            """
            This script finds the protein chains that has a bound NA.
            """
            Pdbids = sorted([
                i.split("/")[-1].split(".")[0] for i in glob.glob("%s/*.pdb"%(self.CleansedPdbFolder))
                ])
            if len(Pdbids) > 0:
                print("Finding Which Protein Chain are Nuc Bound")
                unzip = [self.CleansedPdbFolder for i in range(len(Pdbids))] #TODO why not partial here?
                iteration = list(zip(Pdbids, unzip))
                pool=multiprocessing.Pool(processes=self.n_MultiprocessingWorkers, maxtasksperchild=10000)
                results_ = pool.map(OOC_FindProteinChainBoundWithNuc, iteration)
                results_ = dict(results_)

                # NOTE below is a patch to sanitisation protocol
                results = collections.defaultdict(list)
                for k, v in results_.items():
                    results[k[:4]].extend(v)
                for k, v in results.items():
                    results[k] = sorted(set(v))               

                Df_grand = pd.read_pickle("%s/DataframeGrand.pkl"%(self.DerivedDataFolder))
                Df_grand.loc[:,'NucBoundChains'] = Df_grand['Pdbid'].map(results)

                Df_grand.to_pickle('%s/DataframeGrand_CoordSupplement.pkl'%(self.DerivedDataFolder), compression='infer', protocol=4, storage_options=None)

                return results
            else:
                print("ABORTED. No Coord Data.")
                return None

    def OBSOLETE_DerivedDataCoordinateData(self, DfGrand):
        # NOTE Rerun the script after you downloaded some pdb files
        PART3_UpdateWithCoordinateData = True
        Pdbids = sorted([i.split("/")[-1].split(".")[0] for i in glob.glob("%s/*.pdb"%(self.UnzippedPdbFolder))])
        if len(Pdbids) > 0:
            print("Deriving Info from Pdb files")
            unzip = [self.UnzippedPdbFolder for i in range(len(Pdbids))]
            iteration = list(zip(Pdbids, unzip))
            pool=multiprocessing.Pool(processes=self.n_MultiprocessingWorkers, maxtasksperchild=10000)

            results = pool.map(ReadInfoPdb, iteration)
            pdbid, DNAChain, RNAChain, Hetatm, MdlTyp, Modres = zip(*results)

            DfPdbFile = pd.DataFrame({"Pdbid":list(pdbid), "DNA Chains":list(DNAChain), 
                                    "RNA Chains":list(RNAChain), "Hetatm":list(Hetatm), 
                                    "Model Status":list(MdlTyp),"Modified Residues":list(Modres) })

            DfGrand = pd.merge(DfGrand, DfPdbFile, how='outer', on = 'Pdbid')    

        DfGrand.to_pickle('%s/DataframeGrand_Supplemented.pkl'%(self.DerivedDataFolder), compression='infer', protocol=4, storage_options=None)


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