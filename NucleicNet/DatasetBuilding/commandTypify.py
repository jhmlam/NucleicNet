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

import tempfile
import itertools
import pickle
import gc
import tqdm
import multiprocessing
import operator
from functools import partial


from biopandas.pdb import PandasPdb

from scipy import spatial, sparse
import numpy as np


from NucleicNet.DatasetBuilding.util import *
from NucleicNet.DatasetBuilding.commandLandmark import Landmark_type_class_dict


# =============================================
# TODO Move to commandSantise, generalise
# =============================================
def Sanitise_ReadProteinPdb(pdbfiledir, SelectedResnameList = [], ExcludedResnameList = []):

    ppdb = PandasPdb()
    CurrentPdbStructure = ppdb.read_pdb(pdbfiledir)
    pdbid = pdbfiledir.split("/")[-1].split(".")[0]
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
        #print(df.shape)

    if (len(SelectedResnameList) > 0) & (len(ExcludedResnameList) > 0):
        df = df.loc[(df['residue_name'].isin(
                            SelectedResnameList)) & ~(df['residue_name'].isin(
                            ExcludedResnameList))]
    else:

        if len(SelectedResnameList) > 0:
            df = df.loc[df['residue_name'].isin(
                            SelectedResnameList)]
            #print(df.shape)
        if len(ExcludedResnameList) > 0:
            df = df.loc[~df['residue_name'].isin(
                            ExcludedResnameList)]

    return df

def Sanitise_GetChainGroups(protein_df, EditDistanceThreshold = 0.1, PrintingVerbose = False):
    """
    This function returns chains with high sequence similarity
    Input a ppdb df return chain groups
    """
    # NOTE Because nothing has to be chosen!
    if len(sorted(set(protein_df['chain_id'].tolist()))) == 1:
        return {0:sorted(set(protein_df['chain_id'].tolist()))}
    else:

        # Calculate percentage difference in string. 
        protein_chainresdf = [pd.DataFrame(y) for x, y in protein_df.groupby(['chain_id'], as_index=False)]
        chainseq = {}
        for chain_df in protein_chainresdf:
            chain_i = chain_df['chain_id'].tolist()[0]
            chain_df = chain_df.drop_duplicates(subset=['residue_number'], keep='first')
            chainseq[chain_i] = chain_df['residue_name'].tolist()

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

        return grouped_chain

        # NOTE Below is an example to choose chain using bfactor
        ChainChosen = []
        for copygroup, copygroup_chainlist in grouped_chain.items():
            ChainChoosingCriteria = {}

            if len(copygroup_chainlist) == 1:
                ChainChosen.append(copygroup_chainlist[0])
                continue

            for chain_i in copygroup_chainlist:
                #NAtree = spatial.cKDTree(
                #    landmark_df.loc[landmark_df['chain_id'] == chain_i][['x_coord', 'y_coord', 'z_coord']].values
                #    )
                #atoms = NAtree.query_ball_point(protein_df[['x_coord', 'y_coord', 'z_coord']].values, ProteinContactDistanceThreshold)
                #ChainChoosingCriteria[chain_i] = len(np.array(sorted(set(itertools.chain(*atoms.tolist())))))
                ChainChoosingCriteria[chain_i] = protein_df.loc[protein_df['chain_id'] == chain_i]['b_factor'].sum()

            ChainChosen.append(min(ChainChoosingCriteria.items(), key=operator.itemgetter(1))[0])

# =========================
# Out of class functions
# =========================


def OOC_ByLandmark(SparseClassMatrix, pdbid, landmark_avail,tc_mapping, 
                    DIR_LandmarkFolder, TakeCentroid, DistanceCutoff, halo_coord):
    #print(pdbid)
    Supplementary_NotNonSite = []
    for t in landmark_avail:
        with open(DIR_LandmarkFolder+"/%s.%s.landmark" %(pdbid,t),'rb') as fn:
            landmark_df = pickle.load(fn)

        if TakeCentroid:
            landmark_df = landmark_df.groupby(by=[
                "chain_id","residue_number", "residue_name"
                ]).mean().reset_index()

        landmark_classlist = landmark_df["residue_name"].tolist()
        landmark_classlist_mapped = [tc_mapping[i] for i in landmark_classlist]


        landmark_coord = landmark_df[['x_coord', 'y_coord','z_coord']].values
        landmark_occupancy = landmark_df["occupancy"].values


        # NOTE This assign halo to nearest landmark essentially the voronoi cell
        # NOTE The nonredundant has a suffix "_"
        landmark_nonredundant = landmark_df["%s_" %(t)].values


        landmark_proteinatomcontactnumber = None
        if t == 'nucsite':
            landmark_proteinatomcontactnumber = landmark_df["ProteinAtomContactNumber"].values
            # TODO 1 or 0 in given number
            landmark_proteinatomcontactnumber = (landmark_proteinatomcontactnumber > 0.0).astype(int)



        # ================================
        # Partition into voronoi cells
        # ================================


        landmark_tree = spatial.cKDTree(landmark_coord)
        landmark_nearest_dist, landmark_nearest = landmark_tree.query(halo_coord, k = [1], 
                                    eps=0, p=2, 
                                    distance_upper_bound=np.inf, workers=1)

        landmark_nearest_dist = landmark_nearest_dist.T[0]
        landmark_nearest = landmark_nearest.T[0]

        # TODO if ConcentrateOnCentroid:
        # else:


        index_within = np.where(
                                landmark_nearest_dist < DistanceCutoff
                            )[0].tolist()
        index_within = sorted(set(index_within))




        # =============================
        # Update Typi matrix
        # ==============================
        for l in index_within:
                # NOTE The index of the cetroid is given as landmark_nearest[l]
                class_index=landmark_classlist_mapped[landmark_nearest[l]]
                temp_occupancy = landmark_occupancy[landmark_nearest[l]]
                SparseClassMatrix[l,class_index] += temp_occupancy


        # ===================================
        # Supplementary filtering
        # ===================================
        haha = 0
        if landmark_proteinatomcontactnumber is not None:
            for l in index_within:
                class_index=landmark_classlist_mapped[landmark_nearest[l]]
                temp_contact = landmark_proteinatomcontactnumber[landmark_nearest[l]]
                SparseClassMatrix[l,class_index] *= temp_contact
                # NOTE index that are assigned to a base that make no contact is not nonsite
                if temp_contact == 0:
                    Supplementary_NotNonSite.append(l)
                else:
                    haha +=1




        # ======================================
        # Nonredund
        # ========================================
        # Indicate nonredundant 
        # Why add a second time? No The nonredundant has a suffix "_", and landmark_nonredundant has 0,1 values, 
        # so nonredun will have a value 2 redund will have value 1
        if landmark_proteinatomcontactnumber is not None:
            for l in index_within:
                SparseClassMatrix[l,tc_mapping["%s_" %(t)]] += landmark_nonredundant[landmark_nearest[l]] * landmark_proteinatomcontactnumber[landmark_nearest[l]]
        else:
            for l in index_within:
                SparseClassMatrix[l,tc_mapping["%s_" %(t)]] += landmark_nonredundant[landmark_nearest[l]]







    Supplementary_NotNonSite = sorted(set(Supplementary_NotNonSite))
    #print(len(Supplementary_NotNonSite), SparseClassMatrix.shape , haha, pdbid)


    return SparseClassMatrix , Supplementary_NotNonSite

def OOC_ByNonsite(SparseClassMatrix, pdbid,tc_mapping, halo_tree,
                DIR_BoundPdbFolder, EditDistanceThreshold, PrintingVerbose,
                NotNonsiteResnameList,NotNonsiteDistanceCutoff, halo_coord,
                Supplementary_NotNonSite):
    # Find halo within NotNonSiteResname 
    notnonsite_df = Sanitise_ReadProteinPdb(DIR_BoundPdbFolder+"/%s.pdb" %(pdbid),
                                        SelectedResnameList = NotNonsiteResnameList,
                                        ExcludedResnameList = [])

    notnonsite_coord = notnonsite_df[['x_coord', 'y_coord', 'z_coord']].values
    notnonsite_halo_index = halo_tree.query_ball_point(notnonsite_coord, NotNonsiteDistanceCutoff)
    notnonsite_halo_index = sorted(set(itertools.chain(*notnonsite_halo_index.tolist())))
    # TODO Supplement a set of not non site corresponding to the trimming process?
    notnonsite_halo_index.extend(Supplementary_NotNonSite)
    notnonsite_halo_index = sorted(set(notnonsite_halo_index))

    for i in notnonsite_halo_index:
            SparseClassMatrix[i,tc_mapping['notnonsite']] += 1.0


    protein_df = Sanitise_ReadProteinPdb(DIR_BoundPdbFolder+"/%s.pdb" %(pdbid), 
                                        SelectedResnameList = ["ALA","CYS","ASP","GLU","PHE","GLY", 
                                                            "HIS","ILE","LYS","LEU","MET","ASN", 
                                                            "PRO","GLN","ARG","SER","THR","VAL", 
                                                                "TRP","TYR"],
                                        ExcludedResnameList = [])
    protein_coord = protein_df[['x_coord', 'y_coord', 'z_coord']].values
    #print(set(protein_df.chain_id.tolist()))
    protein_tree = spatial.cKDTree(protein_coord)

    grouped_chain = Sanitise_GetChainGroups(protein_df, 
                        EditDistanceThreshold = EditDistanceThreshold, 
                        PrintingVerbose = PrintingVerbose)

    ChainChosen = []
    for copygroup, copygroup_chainlist in grouped_chain.items():
        ChainChoosingCriteria = {}

        if len(copygroup_chainlist) == 1:
            ChainChosen.append(copygroup_chainlist[0])
            continue
        for chain_i in copygroup_chainlist:
            ChainChoosingCriteria[chain_i] = protein_df.loc[protein_df['chain_id'] == chain_i]['b_factor'].mean()

        ChainChosen.append(min(ChainChoosingCriteria.items(), key=operator.itemgetter(1))[0])
    #print(ChainChosen)
    #print(grouped_chain)
    # Assign halo to protein chain 
    _ , protein_nearest = protein_tree.query(halo_coord, 
                            k = [1], 
                            eps=0, p=2, 
                            distance_upper_bound=np.inf, workers=1)
    protein_nearest = protein_nearest.T[0]
    allatomchainid = protein_df['chain_id'].tolist()
    halo_chain = [allatomchainid[protein_nearest[i]] for i in range(halo_coord.shape[0])]


    for i in range(halo_coord.shape[0]):
        if (halo_chain[i] in ChainChosen) & (i not in notnonsite_halo_index):
            SparseClassMatrix[i,tc_mapping['nonsite_']] += 1.0
    return SparseClassMatrix

def OOC_RetrieveHaloCoords(halo_tuple, reshape = True):

        halo_index, lattice_shape, lattice_minmax, halo_bound = halo_tuple
        minx, miny, minz, maxx, maxy, maxz = lattice_minmax

        x_numintervals = lattice_shape[0] 
        y_numintervals = lattice_shape[1] 
        z_numintervals = lattice_shape[2] 

        lattice_coord = np.mgrid[           minx:maxx:(x_numintervals)*1j, 
                                            miny:maxy:(y_numintervals)*1j, 
                                            minz:maxz:(z_numintervals)*1j
                                            ]

        halo_coord = lattice_coord[:, halo_index.T[0], halo_index.T[1], halo_index.T[2]]
        halo_coord = np.squeeze(halo_coord).T
        del lattice_coord
        gc.collect()
        return halo_coord



def OOC_TypifyPipeline(pdbid, UpdateExist = True, SavingTypi = True, PrintingVerbose = False,
                        DIR_TypiFolder = "",
                        DIR_HaloFolder = "",
                        DIR_LandmarkFolder = "",
                        DIR_BoundPdbFolder = "",
                        Recipe = ['landmark','nonsite'],
                        TakeCentroid = True, 
                        DistanceCutoff = 5.0, NotNonsiteDistanceCutoff = 5.0, EditDistanceThreshold = 0.1,
                        NotNonsiteResnameList = ["DA","DT","DC","DG","DU",
                                                    "A","T","C","G","U"],
                        tc_mapping = {},landmark_avail = [],
                        ):

    # Check file exists
    if UpdateExist:
        pass
    else:
        if (os.path.exists(DIR_TypiFolder + '/%s.typi.npz' %(pdbid))) & SavingTypi:
            return


    # ckdtree of halo
    try:
        with open(DIR_HaloFolder+"/%s.halotup"%(pdbid),'rb') as fn:
            halo_tuple = pickle.load(fn)
        halo_coord = OOC_RetrieveHaloCoords(halo_tuple)
        halo_tree = spatial.cKDTree(halo_coord)
    except FileNotFoundError:
        print("%s has no halo. Why? Check. ABORTED." %(pdbid))
        return


    # NOTE This has shape (n_halo, n_possible_class)
    SparseClassMatrix = sparse.csr_matrix((halo_coord.shape[0], 
                                len(tc_mapping.keys()))).tolil()
    # =====================
    # By Landmark
    # =====================
    if "landmark" in Recipe:
        SparseClassMatrix, Supplementary_NotNonSite = OOC_ByLandmark(SparseClassMatrix, pdbid, landmark_avail,tc_mapping, 
                DIR_LandmarkFolder, TakeCentroid, DistanceCutoff, halo_coord)
    
    # =======================
    # By Nonsite
    # =======================
    if "nonsite" in Recipe:
        SparseClassMatrix = OOC_ByNonsite(SparseClassMatrix, pdbid,tc_mapping, halo_tree,
            DIR_BoundPdbFolder, EditDistanceThreshold, PrintingVerbose,
            NotNonsiteResnameList,NotNonsiteDistanceCutoff, halo_coord, Supplementary_NotNonSite)

    """
    # NOTE This prints the amount contained in each 
    # TODO we can do some statistics on these per pdb as well as globally for each column
    print(SparseClassMatrix[:,0:0+1].sum(), 
            SparseClassMatrix[:,2:9+1].sum(),  
            SparseClassMatrix[:,10].sum(),
            SparseClassMatrix[:,11].sum(),
            SparseClassMatrix[:,12].sum(),
            SparseClassMatrix.shape )
    """
    SparseClassMatrix = SparseClassMatrix.tocsr(copy=False)
    if SavingTypi:
        sparse.save_npz(DIR_TypiFolder + '/%s.typi.npz' %(pdbid),
                                            SparseClassMatrix,compressed = True)
        #with open(DIR_TypiFolder + '/%s.typi' %(pdbid), 'wb') as fn:
        #    pickle.dump(SparseClassMatrix,fn, protocol=4)

    return
# ====================
# Body
# ====================


class Typify:
    def __init__(self,  DIR_TypiFolder = "../Database-PDB/typi",
                        DIR_HaloFolder = "../Database-PDB/halo",
                        n_MultiprocessingWorkers = 16):

        self.DIR_TypiFolder = DIR_TypiFolder
        self.DIR_HaloFolder = DIR_HaloFolder
        self.tc_dict = Landmark_type_class_dict()
        self.n_MultiprocessingWorkers = n_MultiprocessingWorkers
        MkdirList([self.DIR_TypiFolder])

    def Pipeline(self, DIR_LandmarkFolder = "../Database-PDB/landmark", 
                         DIR_BoundPdbFolder = "../Database-PDB/cleansed",
                         NotNonsiteResnameList = ["DA","DT","DC","DG","DU",
                                                    "A","T","C","G","U",
                                                    # NOTE also accept other modified base and possibly ligand names if wanted
                                                    "1MG","5MC","1MA","U37","UFT",
                                                    "SUR","PYO","P5P","OMC","OMG","OMU","ONE",
                                                    "FMU","GTA","CFZ","CFL","AP7","ADN","AZM",
                                                    "A9Z","6MZ","N5M","5BU","2PR","2AD","23G","A23",
                                                    "APC","CGI","PGP","U5P",
                                                    "DI","I"], 
                         Recipe = ["landmark", "nonsite"],
                         DistanceCutoff = 5.0, NotNonsiteDistanceCutoff = 5.0,
                         EditDistanceThreshold = 0.1,
                         PrintingVerbose = False,
                         TakeCentroid = True, SavingTypi = True, UpdateExist = True):

        # NOTE This asserts a thin shell of halo that is bordering the site and non-site. 
        #      maybe useful during training model refinement/tokenisation.
        #  
        #NotNonsiteDistanceCutoff = DistanceCutoff

        # ======================
        # Check file availble
        # ======================
        PART0_CheckAvail = True
        if PART0_CheckAvail:
            landmark_avail = sorted(set(glob.glob(DIR_LandmarkFolder +"/*.landmark")))
            pdbid_avail = sorted(set([i.split("/")[-1].split(".")[0] for i in landmark_avail]))
            landmark_avail = sorted(set([i.split("/")[-1].split(".")[-2] for i in landmark_avail]))

            for i in landmark_avail:
                assert (i in list(self.tc_dict.keys())) , "The landmark type %s is not defined." %(i)

            # NOTE This remove pdbid where landmark type are missing. e.g. it contains no nucleotide
            pdbid_excluded = []
            for pdbid in pdbid_avail:
                for t in landmark_avail:
                    if not os.path.exists(DIR_LandmarkFolder +"/%s.%s.landmark" %(pdbid,t)):
                        pdbid_excluded.append(pdbid)



            
            pdbid_avail = sorted(set(pdbid_avail) - set(pdbid_excluded))
            
            print("The following pdbid were excluded. Check")
            print(pdbid_excluded)
            if UpdateExist:
                pass
            else:
                pdbid_excluded = []
                for i in pdbid_avail:
                    if (os.path.exists(self.DIR_TypiFolder + '/%s.typi.npz' %(i))):
                        pdbid_excluded.append(i)
                pdbid_avail = sorted(set(pdbid_avail) - set(pdbid_excluded))
                
            print("There are %s PDB entries to process" %(len(pdbid_avail)))

            #pdbid_avail = ['4f3t00000000']

        # ============================
        # Class index mapping
        # ============================
        PART1_DefineClassMapping = True
        if PART1_DefineClassMapping:
            all_possible_class = []
            for t in landmark_avail:
                    all_possible_class.extend(self.tc_dict[t])

            tc_mapping = {}
            for i in range(len(all_possible_class)):
                tc_mapping[all_possible_class[i]] = i
            tc_mapping['nonsite_'] = i+1
            tc_mapping['notnonsite'] = i+2

            # TODO Store tc_mapping
            print(tc_mapping)

            if SavingTypi:
                with open(self.DIR_TypiFolder + '/ClassIndex.pkl', 'wb') as fn:
                    pickle.dump(tc_mapping,fn, protocol=4)


        # ============================
        # Multiprocessing
        # ============================
        DIR_TypiFolder = self.DIR_TypiFolder
        DIR_HaloFolder = self.DIR_HaloFolder

        pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
        partial_TypifyPipeline = partial(OOC_TypifyPipeline, 
                        UpdateExist = UpdateExist, SavingTypi = SavingTypi, PrintingVerbose = PrintingVerbose,
                        DIR_TypiFolder = DIR_TypiFolder,
                        DIR_HaloFolder = DIR_HaloFolder,
                        DIR_LandmarkFolder = DIR_LandmarkFolder,
                        DIR_BoundPdbFolder = DIR_BoundPdbFolder,
                        Recipe = Recipe,
                        TakeCentroid = TakeCentroid, 
                        DistanceCutoff = DistanceCutoff, NotNonsiteDistanceCutoff = NotNonsiteDistanceCutoff, 
                        EditDistanceThreshold = EditDistanceThreshold,
                        NotNonsiteResnameList = NotNonsiteResnameList,
                        tc_mapping = tc_mapping,landmark_avail = landmark_avail,
                        )
        pool.map(partial_TypifyPipeline, pdbid_avail)

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