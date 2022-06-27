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
from biopandas.pdb import PandasPdb
from scipy import spatial
from scipy.spatial.distance import euclidean
import numpy as np
import tempfile
import operator
import itertools
import pickle
import gc
import tqdm
from functools import partial
import multiprocessing

from NucleicNet.DatasetBuilding.util import *


def Landmark_type_class_dict():
    # Redundant Nucleotides
    # NOTE I indicate non-redundant landmark sets with *_
    naclass = ["D","P","R","A","G","C","U","T","nucsite_"]
    naclass = sorted(naclass)
    # TODO Reserve 3 letter code rather than 1 let because it overlap w/ amino acid letters....
    #      E.g. reference the 3-let on pdb chem comp???
    dd = {  "nucsite":[i for i in naclass],
            "fpocket":["F", "fpocket_"],
            "fpocket":["F", "fpocket_"],
            "nonsite":["X", "nonsite_"]}   
    return dd



def OOC_Fpocket(pdbfiledir, mode = "default", 
                DIR_OutputLandmarkFolder = "../Database-PDB/landmark", 
                UpdateExist = False,SavingLandmark = True, IndicateNonRedun = True,
                EditDistanceThreshold = 0.1,
                PrintingVerbose = False):
    # NOTE That since fpocket can produce elongated pocket the residue number though represent pocket id 
    #      we cannot take their centroid
    pdbid = pdbfiledir.split("/")[-1].split(".")[0]
    # NOTE To avoid the verbosity in naming I only allow default
    mode = "default"
    if UpdateExist:
        pass
    else:
        #if os.path.exists(DIR_OutputLandmarkFolder + '/%s.fpocket%s.landmark' %(pdbid, mode)):
        if os.path.exists(DIR_OutputLandmarkFolder + '/%s.fpocket.landmark' %(pdbid)):
            return

    ppdb = PandasPdb()
    CurrentPdbStructure = ppdb.read_pdb(pdbfiledir)

    # ====================================================
    # TODO This section is repeated maybe move to util?
    # ======================================================
    PART0_PreliminaryCheckCA = True
    if PART0_PreliminaryCheckCA:
        if CurrentPdbStructure.df['OTHERS']['entry'].str.contains("CA ATOMS ONLY").sum() != 0:
            print("%s only has CA" %(pdbid))
            return

    PART1_Sanitize = True
    if PART1_Sanitize:

        df = CurrentPdbStructure.df['ATOM']
        # TODO This means only reading the first model when there are multiple models in the pdb
        df = df.drop_duplicates(subset=['atom_number'], keep='first')

        # Remove Hydrogens
        df = df[df['element_symbol'] != 'H']

        protein_df = df.loc[df['residue_name'].isin(
                        ["ALA","CYS","ASP","GLU","PHE","GLY", 
                        "HIS","ILE","LYS","LEU","MET","ASN", 
                        "PRO","GLN","ARG","SER","THR","VAL", 
                        "TRP","TYR"
                        ])]


    try:
        DIR_fpocket_env = sorted([i for i in sys.path if (('anaconda3' in i) & ('site-packages' in i) & ('envs' in i)) ])[0].split('lib')[0] + "bin/fpocket"
    except:
        DIR_fpocket_env = 'fpocket'

    with tempfile.TemporaryDirectory() as tmpdirname:
        #print("%s %s \n"%(tmpdirname, pdbid), "\n", )
        fpocketmode_dict = {
            "default": "%s -f %s/%s.pdb; sleep 0.2 ;mv %s/%s_out/ %s/%s_surface/; sleep 0.2" 
                        %(DIR_fpocket_env, tmpdirname, pdbid, tmpdirname, pdbid, tmpdirname, pdbid) ,
            "shallow": "%s -f %s/%s.pdb -m 3 -M 6 -i 2; sleep 0.2 ;mv %s/%s_out/ %s/%s_surface/; sleep 0.2" 
                        %(DIR_fpocket_env, tmpdirname, pdbid, tmpdirname, pdbid, tmpdirname, pdbid) ,
            }        
        # NOTE Copy tpo memory folder
        subprocess.call("cp %s %s/%s.pdb"%(pdbfiledir, tmpdirname, pdbid), shell = True)
        subprocess.call(fpocketmode_dict[mode],shell = True)

        try:
            ppdb = PandasPdb()
            CurrentPdbStructureSurface = ppdb.read_pdb(
                    "%s/%s_surface/%s_out.pdb" %(tmpdirname, pdbid, pdbid)
                    )
            landmark_df = CurrentPdbStructureSurface.df['HETATM'][
                ['atom_number',  'atom_name', 'residue_name', 
                    'chain_id', 'residue_number', 
                    'x_coord', 'y_coord', 'z_coord']]
            landmark_df = landmark_df.assign(residue_name = "F", occupancy = 1.0)# , fpocket_ = 1)
            landmark_df = landmark_df.assign(centroid_id = list(range(landmark_df.shape[0])))

            # Erase temp file in memory
            subprocess.call("rm -r %s/%s_surface/" %(tmpdirname, pdbid) ,shell = True)
            del ppdb, CurrentPdbStructureSurface
            gc.collect()
        except FileNotFoundError:
            print("Fpocket generates no output for %s" %(pdbid))
            return

        if IndicateNonRedun:



            # NOTE Because nothing has to be chosen!
            if len(sorted(set(protein_df['chain_id'].tolist()))) == 1:
                landmark_df = landmark_df.assign(fpocket_ = 1)
                pass
            else:
                protein_tree = spatial.cKDTree(protein_df[['x_coord', 'y_coord', 'z_coord']].values)

                # ========================================
                # TODO Move this part as a util
                # ========================================
                # NOTE Provide a ppdb df return chain groups 
                #       param: df, EditDistanceThreshold
                # Calculate percentage difference in string. 
                # group chain when difference < 0.1
                protein_chainresdf = [pd.DataFrame(y) for x, y in protein_df.groupby(['chain_id'], as_index=False)]
                chainseq = {}
                for chain_df in protein_chainresdf:
                    chain_i = chain_df['chain_id'].tolist()[0]
                    chain_df = chain_df.drop_duplicates(subset=['residue_number'], keep='first')
                    chainseq[chain_i] = chain_df['residue_name'].tolist()
                #print(chainseq)

                # TODO Move edit distance threshold as param
                grouped_chain = defaultdict(list)
                copygroup = 0
                checked_chains = []
                for chain_i,chain_A in chainseq.items():

                    if chain_i in checked_chains:
                        continue

                    grouped_chain[copygroup].append(chain_i)

                    for chain_j, chain_B in chainseq.items():

                        if chain_i == chain_j:
                            continue

                        if chain_j in checked_chains:
                            continue

                        seqdist = SimpleEditDistance(chain_A, chain_B)
                        seqdist = seqdist / min([len(chain_A), len(chain_B)])
                        if seqdist < EditDistanceThreshold:
                            grouped_chain[copygroup].append(chain_j)
                            checked_chains.append(chain_j)
                    copygroup += 1

                if PrintingVerbose:
                    print("%s has the following chain groups and sequences"% (pdbid))
                    print(grouped_chain)
                    print(chainseq)
                


                # NOTE Now pick the chain with smallest b factor
                ChainChosen = []
                for copygroup, copygroup_chainlist in grouped_chain.items():
                    ChainChoosingCriteria = {}

                    if len(copygroup_chainlist) == 1:
                        ChainChosen.append(copygroup_chainlist[0])
                        continue

                    for chain_i in copygroup_chainlist:
                        ChainChoosingCriteria[chain_i] = protein_df.loc[protein_df['chain_id'] == chain_i]['b_factor'].mean()

                    ChainChosen.append(min(ChainChoosingCriteria.items(), key=operator.itemgetter(1))[0])

                # ============================================================
                # NOTE Assign fpocket landmark to closest protein chain
                # ============================================================
                _ , protein_nearest = protein_tree.query(landmark_df[['x_coord', 'y_coord', 'z_coord']].values, 
                                        k = [1], 
                                        eps=0, p=2, 
                                        distance_upper_bound=np.inf, workers=1)
                allatomchainid = protein_df['chain_id'].tolist()
                landmark_df = landmark_df.assign(chain_id =  [allatomchainid[protein_nearest[i][0]] for i in range(len(protein_nearest))])
                landmark_df = landmark_df.assign(fpocket_ = landmark_df['chain_id'].isin(ChainChosen).astype(int).tolist())

        else:
            landmark_df = landmark_df.assign(fpocket_ =1)




        if SavingLandmark:
            #with open(DIR_OutputLandmarkFolder + '/%s.fpocket%s.landmark' %(pdbid, mode), 'wb') as fn:
            with open(DIR_OutputLandmarkFolder + '/%s.fpocket.landmark' %(pdbid), 'wb') as fn:
                pickle.dump(landmark_df,fn, protocol=4)
    return

def OOC_Nucleotide(pdbfiledir, 
                IndicateNonRedun = True,
                DIR_OutputLandmarkFolder = "../Database-PDB/landmark", 
                ProteinContactDistanceThreshold = 5.0,
                AtomicContactDistanceThreshold = 3.8,
                EditDistanceThreshold = 0.1,
                UpdateExist = False,SavingLandmark = False, PrintingVerbose = False):

    # NOTE In this latest version 
    #      will call DAB DAR DAP vs AB AR AP to handle NA-hybrid systems 
    # TODO Move to util
    BackboneGroup = {   "P": ["P", "OP1", "OP2", "O5\'", "O3\'"], 
                        "R": ["C1\'","C2\'","C3\'","C4\'","C5\'", "O3\'", "O4\'"]#,
                        #"D": ["C1\'","C2\'","C3\'","C4\'","C5\'", "O3\'", "O4\'"] 
                    }  
    BaseGroup     = {   "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2",       "N3", "C4"],
                        "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],    
                        "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
                        "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                        "DA": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2",       "N3", "C4"],
                        "DG": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],    
                        "DT": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"],
                        "DC": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                    }

    pdbid = pdbfiledir.split("/")[-1].split(".")[0]

    if UpdateExist:
        pass
    else:
        if (os.path.exists(DIR_OutputLandmarkFolder + '/%s.nucsite.landmark' %(pdbid))) & SavingLandmark:
            #print('/%s.nucsite%s.landmark' %(pdbid, redunstr))
            #print("%s landmark nucleotide exists" %(pdbid))
            return
    ppdb = PandasPdb()
    CurrentPdbStructure = ppdb.read_pdb(pdbfiledir)

    PART0_PreliminaryCheckCA = True
    if PART0_PreliminaryCheckCA:
        if CurrentPdbStructure.df['OTHERS']['entry'].str.contains("CA ATOMS ONLY").sum() != 0:
            print("%s only has CA" %(pdbid))
            return

    PART1_Sanitize = True
    if PART1_Sanitize:

        df = CurrentPdbStructure.df['ATOM']
        # TODO This means only reading the first model when there are multiple models in the pdb
        df = df.drop_duplicates(subset=['atom_number'], keep='first')

        # Remove Hydrogens
        df = df[df['element_symbol'] != 'H']

        protein_df = df.loc[df['residue_name'].isin(
                        ["ALA","CYS","ASP","GLU","PHE","GLY", 
                         "HIS","ILE","LYS","LEU","MET","ASN", 
                         "PRO","GLN","ARG","SER","THR","VAL", 
                         "TRP","TYR"
                        ])]

    PART2_GettingNaChainid = True
    if PART2_GettingNaChainid:
        # DNA Chain
        DNAChain = df[df["residue_name"].isin(["DA","DT","DC","DG","DU"])]["chain_id"].tolist()
        DNAChain = sorted(set(DNAChain))
        # RNA Chain
        RNAChain = df[df["residue_name"].isin(["A","T","C","G","U"])]["chain_id"].tolist()
        RNAChain = sorted(set(RNAChain))
        # TODO I rejected all pdbid w/ Modres but here is  the critical part that can allow masking 
        Modres = []
        for chain_i in RNAChain:
            Modres.extend(df.loc[  (df["chain_id"] == chain_i) & 
                            ~(df["residue_name"].isin(["A","T","C","G","U"]))
                            ].index.tolist())
        for chain_i in DNAChain:
            Modres.extend(df.loc[  (df["chain_id"] == chain_i) & 
                            ~(df["residue_name"].isin(["DA","DT","DC","DG","DU"]))
                            ].index.tolist())


    PART3_GrepNaComponents = True
    if PART3_GrepNaComponents:
        # Grep components
        nadf = df.loc[df["chain_id"].isin(RNAChain + DNAChain)]
        #print(nadf)
        # NOTE we can safely assume that each nucleotide takes a unique chainid_resid
        na_chainresdf = [pd.DataFrame(y) for x, y in nadf.groupby(['chain_id','residue_number'], as_index=False)]
        centroid_id = 0
        landmark_df = []
        for local_nadf in na_chainresdf:

            local_nadf = local_nadf[
                ['atom_number',  'atom_name', 'residue_name', 
                'chain_id', 'residue_number', 
                'x_coord', 'y_coord', 'z_coord']]

            # NA name
            na_name = local_nadf["residue_name"].tolist()[0]

            if na_name not in ["A","G","C","U","DA","DT","DC","DG"]:
                na_name = "M" # NOTE Modified base
                continue 

            # Grep base
            basedf = local_nadf.loc[local_nadf['atom_name'].isin(BaseGroup[na_name])]
            # NOTE you may also want to check the completeness of the nucleotide by basedf.shape[0] == len(BaseGroup[na_name])
            if basedf.shape[0] > 0:
                basedf = basedf.assign(centroid_id = centroid_id, 
                                        occupancy = 1.0, 
                                        residue_name = "%s"%(na_name[-1]), # NOTE The last character. This covers DNA base name overlap. 
                                        residue_name_ = "%s" %(na_name))
                landmark_df.append(basedf)
                centroid_id +=1

            # Grep Backbone
            for backbone_c, backbone_alist in BackboneGroup.items():
                bonedf = local_nadf.loc[local_nadf['atom_name'].isin(backbone_alist)]
                if (backbone_c == "R") & (na_name[0] == "D"):
                    backbone_c = "D"
                if bonedf.shape[0] > 0:
                    bonedf = bonedf.assign(centroid_id = centroid_id, 
                                            occupancy = 1.0, 
                                            residue_name = "%s"%(backbone_c),
                                            residue_name_ = "%s" %(na_name))
                    landmark_df.append(bonedf)
                    centroid_id +=1  
        try:          
            landmark_df = pd.concat(landmark_df, ignore_index=True)
        except:
            print("%s has no landmark. Made entirely of Modified Base? Check. ABORTED." %(pdbid))
            return
        #print(landmark_df)



    IndicateAtomicContactNumber = True
    if IndicateAtomicContactNumber:
        # NOTE Indicate components which make no touch to protein.
        #      What these touch-no-one component means is that the centroid is within the halo of 5 angstrom but the component interacts very weakly with 
        #      atoms on the protien likely screened from any interaction by nearby strong components. We will not include these in training dataset for site nor non-site labels.
        #      there can be something present but not likely interact with the protein. I hope this more stringent criterion will tame the false positive.

        unique_centroid_df = landmark_df.groupby(by=["chain_id","residue_number", "residue_name"]).mean().reset_index()
        contactnumberdict = {}
        for row_i, row in unique_centroid_df.iterrows():
            nacomptempdf = landmark_df.loc[(landmark_df["chain_id"] == row["chain_id"]) & (landmark_df["residue_number"] == row["residue_number"]) & (landmark_df["residue_name"] == row["residue_name"])]
            nacomp_tree = spatial.cKDTree(
                        nacomptempdf[['x_coord', 'y_coord', 'z_coord']].values
            )
            atoms = nacomp_tree.query_ball_point(protein_df[['x_coord', 'y_coord', 'z_coord']].values, 
                                            AtomicContactDistanceThreshold) # TODO This threshold maybe raised.

            contactnumberdict[(row["chain_id"], row["residue_number"], row["residue_name"])] = len(np.array(sorted(set(itertools.chain(*atoms.tolist())))))
        landmark_df.loc[:, "temptuple"] = list(zip(landmark_df["chain_id"].tolist(), landmark_df["residue_number"].tolist(),landmark_df["residue_name"].tolist()))
        landmark_df.loc[:, "ProteinAtomContactNumber"] = landmark_df["temptuple"].map(contactnumberdict)
        landmark_df = landmark_df.drop(columns=["temptuple"])
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(landmark_df[["chain_id","residue_number","ProteinAtomContactNumber"  ]])


    if IndicateNonRedun:
        # NOTE Because nothing has to be chosen!
        if len(sorted(set(landmark_df['chain_id'].tolist()))) == 1:
            landmark_df = landmark_df.assign(nucsite_ = 1)
            pass
        else:
            protein_tree = spatial.cKDTree(protein_df[['x_coord', 'y_coord', 'z_coord']].values)

            # Calculate percentage difference in string. group chain when difference < 0.1
            na_chainresdf = [pd.DataFrame(y) for x, y in nadf.groupby(['chain_id'], as_index=False)]
            chainseq = {}
            for chain_df in na_chainresdf:
                chain_i = chain_df['chain_id'].tolist()[0]
                chain_df = chain_df.drop_duplicates(subset=['residue_number'], keep='first')
                chainseq[chain_i] = chain_df['residue_name'].tolist()
            #print(chainseq)

            
            grouped_chain = defaultdict(list)
            copygroup = 0
            checked_chains = []
            for chain_i,chain_A in chainseq.items():

                if chain_i in checked_chains:
                    continue

                grouped_chain[copygroup].append(chain_i)

                for chain_j, chain_B in chainseq.items():

                    if chain_i == chain_j:
                        continue

                    if chain_j in checked_chains:
                        continue

                    seqdist = SimpleEditDistance(chain_A, chain_B)
                    seqdist = seqdist / min([len(chain_A), len(chain_B)])
                    #print(seqdist)
                    if seqdist < EditDistanceThreshold:
                        grouped_chain[copygroup].append(chain_j)
                        #checked_chains.append(chain_i)
                        checked_chains.append(chain_j)
                copygroup += 1

            if PrintingVerbose:
                print("%s has the following nucleotide chain groups and sequences"% (pdbid))
                print(grouped_chain)
                print(chainseq)
            
            # Pick the one that make most contact w/ protein. NOTE This favours interfacial resudyes
            # TODO Move contact threshold as param
            ChainChosen = []
            for copygroup, copygroup_chainlist in grouped_chain.items():
                ChainChoosingCriteria = {}

                if len(copygroup_chainlist) == 1:
                    ChainChosen.append(copygroup_chainlist[0])
                    continue

                for chain_i in copygroup_chainlist:
                    NAtree = spatial.cKDTree(
                        landmark_df.loc[landmark_df['chain_id'] == chain_i][['x_coord', 'y_coord', 'z_coord']].values
                        )
                    atoms = NAtree.query_ball_point(protein_df[['x_coord', 'y_coord', 'z_coord']].values, 
                                                    ProteinContactDistanceThreshold) # TODO This threshold maybe raised.
                    ChainChoosingCriteria[chain_i] = len(np.array(sorted(set(itertools.chain(*atoms.tolist())))))
                ChainChosen.append(max(ChainChoosingCriteria.items(), key=operator.itemgetter(1))[0])
            #landmark_df = landmark_df.loc[landmark_df['chain_id'].isin(ChainChosen)]
            landmark_df = landmark_df.assign(nucsite_ = landmark_df['chain_id'].isin(ChainChosen).astype(int).tolist())
    else:
        landmark_df = landmark_df.assign(nucsite_ =1)

    landmark_df = landmark_df.drop(columns=['residue_name_'])

    # NOTE This is the centroid. 
    if PrintingVerbose:
        print("%s has %s centroids. 3 centroid per nucleotide if no missing component. Make Sense?" %(pdbid,
        landmark_df.groupby(by=["chain_id","residue_number", "residue_name"]).mean().reset_index().shape[0],
        ))
    #print(landmark_df.groupby(by=["chain_id","residue_number", "residue_name"]).mean().reset_index())

    # NOTE The landmarks are atoms here we will take centroid later on.
    if SavingLandmark:
        with open(DIR_OutputLandmarkFolder + '/%s.nucsite.landmark' %(pdbid), 'wb') as fn:
            pickle.dump(landmark_df,fn, protocol=4) 

    return


# NOTE Landmark are not necessrily close to the halo, we will do this when we typify
class Landmark:
    def __init__(self, 
                DIR_InputPdbFolder = "../Database-PDB/apo", 
                DIR_OutputLandmarkFolder = "../Database-PDB/landmark",
                DIR_GrandPdbDf = "../Database-PDB/DerivedData/DataframeGrand.pkl",

                n_MultiprocessingWorkers = 10):
        self.DIR_InputPdbFolder = DIR_InputPdbFolder
        self.DIR_OutputLandmarkFolder = DIR_OutputLandmarkFolder
        self.DIR_GrandPdbDf = DIR_GrandPdbDf
        self.n_MultiprocessingWorkers = n_MultiprocessingWorkers


        MkdirList([DIR_OutputLandmarkFolder])

    def Fpocket(self, mode = "default", UpdateExist = False, SavingLandmark = True, 
                    IndicateNonRedun = True,EditDistanceThreshold = 0.1,
                    PrintingVerbose = False):

        pdbfilelist = sorted(glob.glob(self.DIR_InputPdbFolder + "/*.pdb"))
        Fpocket_partial = partial(OOC_Fpocket, 
                            mode = mode, UpdateExist = UpdateExist, SavingLandmark = SavingLandmark,
                            DIR_OutputLandmarkFolder = self.DIR_OutputLandmarkFolder, 
                            IndicateNonRedun = IndicateNonRedun, 
                            EditDistanceThreshold = EditDistanceThreshold,
                            PrintingVerbose = PrintingVerbose)
        pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
        pool.map(Fpocket_partial, pdbfilelist)

        return

    def Nucleotide(self, IndicateNonRedun = True,
                DIR_NaBoundPdbFolder = "../Database-PDB/cleansed",
                ProteinContactDistanceThreshold = 5.0,
                AtomicContactDistanceThreshold = 3.8,
                EditDistanceThreshold = 0.1,
                UpdateExist = False,SavingLandmark = True, PrintingVerbose = False):


        pdbfilelist = sorted(glob.glob(DIR_NaBoundPdbFolder + "/*.pdb"))
        #pdbfilelist = sorted(glob.glob(DIR_NaBoundPdbFolder + "/4f3t00000000.pdb"))

        OOC_Nucleotide_partial = partial(OOC_Nucleotide, 
                IndicateNonRedun = IndicateNonRedun,
                DIR_OutputLandmarkFolder = self.DIR_OutputLandmarkFolder,
                ProteinContactDistanceThreshold = ProteinContactDistanceThreshold,
                AtomicContactDistanceThreshold = AtomicContactDistanceThreshold,
                EditDistanceThreshold = EditDistanceThreshold,
                UpdateExist = UpdateExist,SavingLandmark = SavingLandmark, PrintingVerbose = PrintingVerbose)
        pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
        pool.map(OOC_Nucleotide_partial, pdbfilelist)

        return

    def Peptide(self):
        return



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