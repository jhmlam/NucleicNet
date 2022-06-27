import glob
import pickle
import numpy as np
import tqdm
import os
from scipy import sparse
import pandas as pd
import multiprocessing

def OOC_IntegrityFeature(pdbid):
        with np.load("../Database-PDB/feature/%s" %(pdbid)) as f:
                feat = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
        Integrity_Feature_ = np.sum(feat, axis = 0) # TODO This should be done columnwise, because the row almost always sum to 1 when there is no ambiguity

        return (pdbid, Integrity_Feature_)
        
class Checksum:
    def __init__(self, DIR_Database = "../Database-PDB/"):
        self.DIR_Database = DIR_Database
        with open("../Database-PDB/CHECKSUM_Datapoint.pkl", "rb") as fn:
            self.datasizedf = pickle.load( fn)
        with open("../Database-PDB/CHECKSUM_TypiIntegrity.pkl", "rb") as fn:
            self.Integrity_Typi = pickle.load( fn)
        with open("../Database-PDB/CHECKSUM_HaloIntegrity.pkl", "rb") as fn:
            self.Integrity_Halo = pickle.load( fn)
            #print(self.Integrity_Typi)
        with open("../Database-PDB/CHECKSUM_FeatureIntegrity.pkl", "rb") as fn:
            self.Integrity_Feature = pickle.load( fn)
        with open("../Database-PDB/CHECKSUM_CleansedIntegrity.pkl", "rb") as fn:
            self.Integrity_Cleansed = pickle.load( fn)
        with open("../Database-PDB/CHECKSUM_ApoIntegrity.pkl", "rb") as fn:
            self.Integrity_Apo = pickle.load( fn)

        with open("../Database-PDB/CHECKSUM_LandmarkFpocketIntegrity.pkl", "rb") as fn:
            self.Integrity_LandmarkFpocket = pickle.load( fn)
        with open("../Database-PDB/CHECKSUM_LandmarkNucsiteIntegrity.pkl", "rb") as fn:
            self.Integrity_LandmarkNucsite = pickle.load( fn)




    def CheckTypi(self):

        Subject = "../Database-PDB/typi/"
        PdbidList = sorted(glob.glob("../Database-PDB/typi/*.typi.npz"))
        PdbidList = [i.split("/")[-1].split(".")[0] for i in PdbidList]

        print("Collecting %s" %(Subject))
        datasizedf_test = {}
        typiintegrity_test = {}
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
            pdbid = PdbidList[pdbid_i]
            with np.load("../Database-PDB/typi/%s.typi.npz" %(pdbid)) as f:
                    typi = sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape= f['shape'])
            datasizedf_test[pdbid] = typi.shape[0]
            typiintegrity_test[pdbid] = np.sum(typi, axis = 0) 

        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.datasizedf.keys()) + list(datasizedf_test.keys()) ) )) :
            try:
                if self.datasizedf[pdbid] != datasizedf_test[pdbid]:
                    Err_DiffNumDatapoint[pdbid] = (datasizedf_test[pdbid],self.datasizedf[pdbid])
                    #print("WARNING. %s (%s) has a different number of datapoint than the benchmark set (%s). Was halo run correct?" %(pdbid, datasizedf_test[pdbid],self.datasizedf[pdbid] ))
                if not np.allclose(self.Integrity_Typi[pdbid], typiintegrity_test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (typiintegrity_test[pdbid],self.Integrity_Typi[pdbid])
            except KeyError:
                try:
                    self.datasizedf[pdbid]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    datasizedf_test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser

    def CheckHalo(self):

        Subject = "../Database-PDB/halo/"
        PdbidList = sorted(glob.glob("../Database-PDB/halo/*.halo*")) 
        PdbidList = sorted(set([i.split("/")[-1].split(".")[0] for i in PdbidList]))

        print("Collecting %s" %(Subject))
        Integrity_Halo_Test = {}# []
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
            pdbid = PdbidList[pdbid_i]

            ContentCheckXyz = os.path.getsize("../Database-PDB/halo/%s.haloxyz" %(pdbid)) 
            ContentCheckTup = os.path.getsize("../Database-PDB/halo/%s.halotup" %(pdbid))

            with open("../Database-PDB/halo/%s.halotup" %(pdbid), 'rb') as fn:
                halonum, e1,e2,e3 = pickle.load(fn) 
            Integrity_Halo_Test[pdbid] = np.array([halonum.shape[0], sum(e1), sum(e2), sum(e3), ContentCheckXyz, ContentCheckTup])

        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.datasizedf.keys()) + list(Integrity_Halo_Test.keys()) ) )) :
            try:
                if self.datasizedf[pdbid] != int(Integrity_Halo_Test[pdbid][0]):
                    Err_DiffNumDatapoint[pdbid] = (int(Integrity_Halo_Test[pdbid][0]),self.datasizedf[pdbid])
                    #print("WARNING. %s (%s) has a different number of datapoint than the benchmark set (%s). Was halo run correct?" %(pdbid, datasizedf_test[pdbid],self.datasizedf[pdbid] ))
                if not np.allclose(self.Integrity_Halo[pdbid], Integrity_Halo_Test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (Integrity_Halo_Test[pdbid],self.Integrity_Halo[pdbid])
            except KeyError:
                try:
                    self.datasizedf[pdbid]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    Integrity_Halo_Test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser

    def CheckFeature(self):

        Subject = "../Database-PDB/feature/"
        PdbidList = sorted(glob.glob("../Database-PDB/feature/*.npz"))
        PdbidList = [i.split("/")[-1] for i in PdbidList]

        print("Collecting %s" %(Subject))
        pool = multiprocessing.Pool(12)
        Integrity_Feature_Test = pool.map(OOC_IntegrityFeature, PdbidList)
        Integrity_Feature_Test = dict(Integrity_Feature_Test)

        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.Integrity_Feature.keys()) + list(Integrity_Feature_Test.keys()) ) )) :
            try:
                if not np.allclose(self.Integrity_Feature[pdbid], Integrity_Feature_Test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (Integrity_Feature_Test[pdbid],self.Integrity_Feature[pdbid])
            except KeyError:
                try:
                    pdbid_ = pdbid.split(".")[0] # NOTE Check if there are cases where there are typi but not feature
                    self.datasizedf[pdbid_]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    Integrity_Feature_Test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        #print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        #print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content or Number of Datapoint. Likely the latter.")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser

    def CheckCleansed(self):

        Subject = "../Database-PDB/cleansed/"

        PdbidList = sorted(glob.glob("../Database-PDB/cleansed/*.pdb")) 
        PdbidList = sorted(set([i.split("/")[-1].split(".")[0] for i in PdbidList]))
        Integrity_Cleansed_Test = {}
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
            pdbid = PdbidList[pdbid_i]
            Integrity_Cleansed_Test[pdbid] = os.path.getsize("../Database-PDB/cleansed/%s.pdb" %(pdbid)) 



        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.datasizedf.keys()) + list(Integrity_Cleansed_Test.keys()) ) )) :
            try:
                
                if not np.allclose(self.Integrity_Cleansed[pdbid], Integrity_Cleansed_Test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (Integrity_Cleansed_Test[pdbid],self.Integrity_Cleansed[pdbid])
            except KeyError:
                try:
                    self.Integrity_Cleansed[pdbid]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    Integrity_Cleansed_Test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser


    def CheckApo(self):

        Subject = "../Database-PDB/apo/"

        PdbidList = sorted(glob.glob("../Database-PDB/apo/*.pdb")) 
        PdbidList = sorted(set([i.split("/")[-1].split(".")[0] for i in PdbidList]))
        Integrity_Cleansed_Test = {}
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
            pdbid = PdbidList[pdbid_i]
            Integrity_Cleansed_Test[pdbid] = os.path.getsize("../Database-PDB/apo/%s.pdb" %(pdbid)) 



        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.datasizedf.keys()) + list(Integrity_Cleansed_Test.keys()) ) )) :
            try:
                
                if not np.allclose(self.Integrity_Apo[pdbid], Integrity_Cleansed_Test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (Integrity_Cleansed_Test[pdbid],self.Integrity_Apo[pdbid])
            except KeyError:
                try:
                    self.Integrity_Apo[pdbid]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    Integrity_Cleansed_Test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser


    def CheckLandmarkNucsite(self):

        Subject = "../Database-PDB/landmark/*nucsite.landmark"
        PdbidList = sorted(glob.glob("../Database-PDB/landmark/*.nucsite.landmark")) 
        PdbidList = sorted(set([i.split("/")[-1].split(".")[0] for i in PdbidList]))
        Integrity_LandmarkNucsite_Test = {}
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
            pdbid = PdbidList[pdbid_i]
            ContentCheckNucsite = os.path.getsize("../Database-PDB/landmark/%s.nucsite.landmark" %(pdbid)) 
            df = pd.read_pickle("../Database-PDB/landmark/%s.nucsite.landmark" %(pdbid))
            Integrity_LandmarkNucsite_Test[pdbid] = np.array([df['centroid_id'].sum() , df.shape[0], np.around(df['x_coord'].sum(),3), np.around(df['y_coord'].sum(),3), np.around(df['z_coord'].sum(),3)])


        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.Integrity_LandmarkNucsite.keys()) + list(Integrity_LandmarkNucsite_Test.keys()) ) )) :
            try:
                if not np.allclose(self.Integrity_LandmarkNucsite[pdbid], Integrity_LandmarkNucsite_Test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (Integrity_LandmarkNucsite_Test[pdbid],self.Integrity_LandmarkNucsite[pdbid])
            except KeyError:
                try:
                    self.Integrity_LandmarkNucsite[pdbid]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    Integrity_LandmarkNucsite_Test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser

    def CheckLandmarkFpocket(self):

        Subject = "../Database-PDB/landmark/*fpocket.landmark"
        PdbidList = sorted(glob.glob("../Database-PDB/landmark/*.fpocket.landmark")) 
        PdbidList = sorted(set([i.split("/")[-1].split(".")[0] for i in PdbidList]))
        Integrity_LandmarkFpocket_Test = {}
        for pdbid_i in tqdm.tqdm(range(len(PdbidList))):
            pdbid = PdbidList[pdbid_i]
            ContentCheckFpocket = os.path.getsize("../Database-PDB/landmark/%s.fpocket.landmark" %(pdbid)) 
            df = pd.read_pickle("../Database-PDB/landmark/%s.fpocket.landmark" %(pdbid))
            Integrity_LandmarkFpocket_Test[pdbid] = np.array([df['centroid_id'].sum() , df.shape[0], np.around(df['x_coord'].sum(),3), np.around(df['y_coord'].sum(),3), np.around(df['z_coord'].sum(),3)])

        print("Checking %s" %(Subject))
        # NOTE Check size and file missing or extra and content
        Err_DiffNumDatapoint = {}
        Err_DiffContent = {}
        Err_ExtraFileInUser = {}
        Err_MissingFileInUser ={}
        for pdbid in tqdm.tqdm(sorted(set(list(self.Integrity_LandmarkFpocket.keys()) + list(Integrity_LandmarkFpocket_Test.keys()) ) )) :
            try:
                if not np.allclose(self.Integrity_LandmarkFpocket[pdbid], Integrity_LandmarkFpocket_Test[pdbid]):
                    #print("WARNING. %s (%s) does not have the same class labels than the benchmark set (%s). Was typi run correct?" %(pdbid, typiintegrity_test[pdbid],self.Integrity_Typi[pdbid] ))
                    Err_DiffContent[pdbid] = (Integrity_LandmarkFpocket_Test[pdbid],self.Integrity_LandmarkFpocket[pdbid])
            except KeyError:
                try:
                    self.Integrity_LandmarkFpocket[pdbid]
                except KeyError:
                    #print("WARNING. %s is an extra typi file in your newly built Database" %(pdbid))
                    Err_ExtraFileInUser[pdbid] = False
                try:
                    Integrity_LandmarkFpocket_Test[pdbid]
                except KeyError:
                    #print("WARNING. %s is a missing typi file in your newly built Database" %(pdbid))
                    Err_MissingFileInUser[pdbid] = False
        # ====================
        # Summary
        # ====================
        print("\nWARNING. A list of Pdbid with Different Number of Datapoint")
        print(sorted(Err_DiffNumDatapoint.keys()))
        print("\nWARNING. A list of Pdbid with Different Content")
        print(sorted(Err_DiffContent.keys()))
        print("\nWARNING. A list of Pdbid Extra in %s" %(Subject))
        print(sorted(Err_ExtraFileInUser.keys()))
        print("\nWARNING. A list of Pdbid Missing in %s" %(Subject))
        print(sorted(Err_MissingFileInUser.keys()))

        return Err_DiffNumDatapoint, Err_DiffContent, Err_ExtraFileInUser, Err_MissingFileInUser

