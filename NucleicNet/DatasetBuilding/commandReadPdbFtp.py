from io import StringIO
import sys
import urllib.request
from collections import defaultdict
import collections
import gc
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
import networkx as nx
import tqdm
import requests
import json
import pandas as pd
from collections import defaultdict
import tqdm

sys.path.append('../')

from NucleicNet.DatasetBuilding.util import *

# =====================================
# General Ftp url folder download
# =====================================
def FtpDirectoryDownloading(urltobe,storage, exclusion = ["pdb_chain_go.lst"]):

    """
    if help:
        # NOTE exclusion is a list of large files of size around 3GB, but maybe useful as a classification problem. 
        #      In that case, the class should correspond to some GO annotation 
        #      (ideally mutually exclusive labels, e.g. most kinases are non-NA binding)
        #      Contrastive training against some non-NA binding protein may also be considered    
        
    """

    urlpath =urllib.request.urlopen(urltobe)
    content = StringIO(urlpath.read().decode('utf-8'))
    df = pd.read_csv(content, sep='\s+', header = None)
    for filename in df.iloc[:,-1].tolist():

        if filename in exclusion: 
            continue
        urllib.request.urlretrieve('%s/%s' %(urltobe,filename), '%s/%s'%(storage, filename))

# ===============================
# Read the raws
# ================================
# Read In the Full Pdb document of experiment type and protein/na
def ReadPdbEntryType(folder):
    DfComplete = pd.read_csv("%s/pdb_entry_type.txt" %(folder), header = None, sep = '\t', names = ["Pdbid", "ProNu", "Experiment"])
    return DfComplete

# Read In resolution
def ReadResolu(folder):
    DfResolu = pd.read_csv("%s/resolu.idx" %(folder), header = None, 
                            sep = '\t;\t', names = ["Pdbid","Resolution"], 
                            skiprows=6, engine = 'python')
    DfResolu.loc[:,"Pdbid"] = DfResolu['Pdbid'].str.lower()

    return DfResolu.dropna()

# Read In commonest solvent ions and modified residue
def ReadSolvent(folder):
    DfSolvent = pd.read_csv("%s/SolventIonWater.txt" %(folder), sep = '\t')
    return DfSolvent["id"].tolist()

# Read In Ligand per Pdb
def ReadLigandPerPdb(folder,solvent):
    dfligtopdb = pd.read_csv("%s/cc-to-pdb.tdd"%(folder), sep="\t", names = ["PdbCC", "Pdbid"])

    pdbdict=defaultdict(list)
    for i, row in dfligtopdb.iterrows():
        dfligtopdb.set_value(i,'Pdbid',row['Pdbid'].split(" "))
        for j in row['Pdbid']:
            pdbdict[j].append(row['PdbCC'])
    dfpdbtolig = pd.DataFrame({'Pdbid' :pd.Series(sorted(pdbdict.keys())) })
    dfpdbtolig.loc[:,'LigNum'] = pd.Series([]*len(dfpdbtolig))
    for i, row in  dfpdbtolig.iterrows():
        dfpdbtolig.set_value(i,'LigNum',len(set(pdbdict[row["Pdbid"]])-set(solvent)))
    return dfligtopdb, pdbdict, dfpdbtolig

# Read In NPIDB data TODO We should write our own by perusing pdb_seqres.txt
def ReadNPIDB_ListOfComplexes(folder):
    NPIDB = pd.read_csv('%s/NPIDB_ListOfComplexes.txt' %(folder), sep='\t\t\t', engine = 'python')

    for i in NPIDB.keys():
        NPIDB[str(i)] = NPIDB[str(i)].replace({'\"':''},regex=True)

    for i in [str("[resolution]"), str("[models]"), str("[organism_id]")]:
        NPIDB[str(i)] = NPIDB[str(i)].apply(pd.to_numeric, errors='ignore')

    NPIDB = NPIDB.rename(index=str, columns={"[pdb_id]":"Pdbid", 
                        "[organism_id]":"OrganismID", "[kind]": "NucleicAcid", 
                        "[dna_ch]":"DnaChain", "[rna_ch]":"RnaChain", "[prt_ch]":"ProteinChain", 
                        "[organism]":"OrganismName", "[classification]":"NpidbClassification"})

    NPIDB = NPIDB.drop(['[depDate]','[resolution]','[experimental_method]','[title]','[models]'], axis=1)
    return NPIDB

# Read In Title
def ReadTitle(folder):
    df = pd.read_csv('%s/compound.idx' %(folder), sep='\t', names = ["Pdbid","Title"], skiprows=4)
    df = df.apply(lambda x: x.astype(str).str.lower())
    return df
    
def ReadCC2PdbTdd(folder):
    Df_CC2PdbTdd = pd.read_csv('%s/cc-to-pdb.tdd' %(folder), sep='\t', names = ["ChemicalComponent","Pdbids"])
    Dict_Pdb2CCTdd = defaultdict(list)
    for row_i, row in Df_CC2PdbTdd.iterrows():
        pdbids_ = row["Pdbids"].split(" ")
        #print(pdbids_)
        if row["ChemicalComponent"] is np.nan:
            continue
        for i in pdbids_:
            if len(i) != 4:
                continue
            Dict_Pdb2CCTdd[i].append(row["ChemicalComponent"])

    return Df_CC2PdbTdd , Dict_Pdb2CCTdd

def ReadSeqResHeader(folder):
    # Skip even Line
    df = pd.read_csv(folder+"/pdb_seqres.txt", skiprows=lambda x: x%2 == 1, 
                     sep = 'mol:', header=None, names = ["RawPrefix", "RawSuffix"],
                     engine='python')
    new = df["RawPrefix"].str.split("_", n = 1, expand = True)
    df.loc[:,"Pdbid"] = new[0]
    df.loc[:,"Pdbid"] = df["Pdbid"].str.replace(">","")
    df.loc[:,"Chainid"] = new[1]
    #print(df)
    new = df["RawSuffix"].str.split(" ", n = 2, expand = True)
    df.loc[:,"ChainType"] = new[0]
    df.loc[:,"ChainLength"] = new[1]
    df.loc[:,"ChainLength"] = df["ChainLength"].str.replace("length:","").astype(int)
    df = df.drop(columns = ["RawSuffix", "RawPrefix"])

    StatChainlength = df.groupby(['Pdbid', 'ChainType']).agg(
        MaxChainLength=pd.NamedAgg(column='ChainLength', aggfunc=max),
        MinChainLength=pd.NamedAgg(column='ChainLength', aggfunc=min),
        SumChainLength=pd.NamedAgg(column='ChainLength', aggfunc=sum),
        MeanChainLength=pd.NamedAgg(column='ChainLength', aggfunc='mean'),
        VarChainLength=pd.NamedAgg(column='ChainLength', aggfunc='var'),
        CountChainLength=pd.NamedAgg(column='ChainLength', aggfunc='count'),
    )
    StatChainlength = StatChainlength.reset_index(level=['ChainType', 'Pdbid'])
    StatChainlength_na = StatChainlength.loc[StatChainlength['ChainType'] == 'na']
    StatChainlength_na = StatChainlength_na.drop(columns=["ChainType"])
    StatChainlength_na.columns = StatChainlength_na.columns.map(lambda x : x+'_Nucleotide' if x!='Pdbid' else x)
    StatChainlength_protein = StatChainlength.loc[StatChainlength['ChainType'] == 'protein']
    StatChainlength_protein = StatChainlength_protein.drop(columns=["ChainType"])
    StatChainlength_protein.columns = StatChainlength_protein.columns.map(lambda x : x+'_Peptide' if x!='Pdbid' else x)
    StatChainlength = pd.merge(StatChainlength_protein, StatChainlength_na, how='outer', on = 'Pdbid')
    StatChainlength = StatChainlength.fillna(0)
    StatChainlength.loc[:,"SumChainLength_All"] = StatChainlength["SumChainLength_Nucleotide"] + StatChainlength["SumChainLength_Peptide"]
    #print(StatChainlength)
    del StatChainlength_protein, StatChainlength_na
    gc.collect()
    # NOTE df is a per chain dataframe 
    return df, StatChainlength

# Read In Pubmed ID
def ReadPubmedid(folder):
    df = pd.read_csv('%s/pdb_pubmed.lst' %(folder), sep='\t', names = ["Pdbid","Ordinal","PubmedID"], skiprows=2)
    df = df.drop(['Ordinal'],axis=1)
    return df

# Read In Date and Title
def ReadDateTitle(folder):

    df = pd.read_csv('%s/entries.idx' %(folder), sep="\t", skiprows=2, header=None, names = ["Pdbid","Header","Date","Title","Source","Authors","Resolution","Experiment"])
    df = df[["Pdbid","Header","Date","Title","Source","Authors"]]

    # NOTE some date are wrongly notated in pdb! so an exception has to be done row by row or by coerce
    df.loc[:,"Date"] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors = 'coerce') #df['Date'].astype('datetime64[ns]')

    df.loc[:,"Year"] = df['Date'].dt.year
    df.loc[:,"Month"] = df['Date'].dt.month

    """
    for row_i, row in tqdm.tqdm(df.iterrows()):
        try:
            df.loc[row_i, "Year"] = int(row["Date"].year)
            df.loc[row_i, "Month"] = int(row["Date"].month)
        except:
            df.loc[row_i, "Year"] = np.NaN
            df.loc[row_i, "Month"] = np.NaN
    """

    df.loc[:,"Pdbid"] = df['Pdbid'].str.lower()
    df.loc[:,"Title"] = df['Title'].str.lower()
    df.loc[:,"Authors"] = df['Authors'].str.lower()
    df.loc[:,"Header"] = df['Header'].str.lower()
    df.loc[:,"Source"] = df['Source'].str.lower()
    return df



def ReadInfoPdb(kk):

      from biopandas.pdb import PandasPdb
      pdbid = kk[0]
      unzipped = kk[1]

      #print(pdbid)
      ppdb = PandasPdb()
      CurrentPdbStructure = ppdb.read_pdb("%s/%s.pdb" %(unzipped, str(pdbid)))

      # DNA Chain
      DNAChain = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(["DA","DT","DC","DG","DU"])]["chain_id"].tolist()
      DNAChain = ','.join(sorted(set(DNAChain)))
      if not DNAChain:
         DNAChain= None

      # RNA Chain
      RNAChain = CurrentPdbStructure.df['ATOM'][CurrentPdbStructure.df['ATOM']["residue_name"].isin(["A","T","C","G","U"])]["chain_id"].tolist()
      RNAChain = ','.join(sorted(set(RNAChain)))
      if not RNAChain:
         RNAChain= None

      # Ligand
      Hetatm = CurrentPdbStructure.df['HETATM']["residue_name"].tolist()
      Hetatm = ','.join(sorted(set(Hetatm)))
      if not Hetatm:
         Hetatm= None


      # Model simplified status
      if not CurrentPdbStructure.df['OTHERS'][CurrentPdbStructure.df['OTHERS']["record_name"] == "MDLTYP"].empty:
        MdlTyp = CurrentPdbStructure.df['OTHERS'][CurrentPdbStructure.df['OTHERS']["record_name"] == "MDLTYP"]["entry"].tolist()[0].split(" ONLY, ")[0].strip(" ")
      else:
        MdlTyp = None


      # Modified Residue
      if not CurrentPdbStructure.df['OTHERS'][CurrentPdbStructure.df['OTHERS']["record_name"] == "MODRES"].empty:
         Modres = CurrentPdbStructure.df['OTHERS'][CurrentPdbStructure.df['OTHERS']["record_name"] == "MODRES"]["entry"].tolist()
         Modresdf = pd.read_csv(StringIO('\n'.join(Modres)), delim_whitespace=True, names=["Pdbid","Modres","chain_id","residue_id","unmod","mod","res"]).drop(['mod','res'], axis=1)
         try:
            Modres = ",".join(sorted(set(Modresdf["Modres"].tolist())))
         except TypeError:
            print("%s has a strange mod res"%(pdbid))
            Modres = "UNK"
      else:

         Modres = None

      del ppdb, CurrentPdbStructure

      return pdbid, DNAChain, RNAChain, Hetatm, MdlTyp, Modres



# ============================
# Sequence Homology
# ============================

def ReadBlastClust(bcdir):
    with open("%s" %(bcdir),"r") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        BCDict  = {}
        for i in range(len(content)):
            BCDict[i] = content[i].split(" ")
        BCDictPdbid = {}
        for key, value in BCDict.items():
            PdbidSet = set()
            for j in value:
                PdbidSet.add(j.split("_")[0].lower())
            BCDictPdbid[key] = sorted(PdbidSet)

    return BCDict, BCDictPdbid

def ReadMmseqClust(Mmseqdir):
    # NOTE Resultant is entity id not chain id. The format is the same as old days blast clust
    with open("%s" %(Mmseqdir),"r") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        MmseqDict  = {}
        for i in range(len(content)):
            MmseqDict[i] = content[i].split(" ")
        MmseqDictPdbid = {}
        for key, value in MmseqDict.items():
            PdbidSet = set()
            for j in value:
                PdbidSet.add(j.split("_")[0].lower())
            MmseqDictPdbid[key] = sorted(PdbidSet)

    return MmseqDict, MmseqDictPdbid


# NOTE THis is usefulk for the new mmseq cluster on pdb
def RecurringEntityIDTranslation(MmseqRawDict, 
                                MmseqDict = defaultdict(list), FailingClusters = [],
                                CulmulatedEntityChainDict = defaultdict(list)):

  if len(FailingClusters) == 0:
      return MmseqDict, FailingClusters, CulmulatedEntityChainDict

  FailingClusters_ = []
  #jj = 3
  # NOTE Chekc on each cluster indicated by the raw file query on the entity ID in each cluster
  for clusterid, list_pdbid_entityid in tqdm.tqdm(MmseqRawDict.items()):
      # NOTE This mean it has been done for this cluster
      if clusterid not in FailingClusters:
        continue
      #if jj == 0:
      #  continue
      # NOTE When list_pdbid_entityid are in culmulated translation no need to query again 
      encountered_entityid = list(CulmulatedEntityChainDict.keys())
      # NOTE unknown
      list_pdbid_entityid_ = sorted(set(list_pdbid_entityid) - set(encountered_entityid))
      list_pdbid_entityid_knownchainid = sorted(set(list_pdbid_entityid) - set(list_pdbid_entityid_))
      list_pdbid_entityid  = list_pdbid_entityid_

      # NOTE if we need to consult internet!
      if len(list_pdbid_entityid) > 0:
        querytemplate = """{
              polymer_entities(entity_ids:%s
              ) {
                rcsb_id
                polymer_entity_instances {
                  rcsb_polymer_entity_instance_container_identifiers {
                    auth_asym_id
                  }
                }
              }
            }""" %('["' + '","'.join(list_pdbid_entityid) + '"]')



        url = 'https://data.rcsb.org/graphql?query='
        r = requests.post(url, json={'query': querytemplate})
        if r.status_code != 200: # NOTE Mechanism to retry
            print('Warning. Somethings wrong at cluster %s' %(clusterid))
            FailingClusters_.append(clusterid)
            continue

        json_data = json.loads(r.text)
        for polymer_entities in json_data['data']['polymer_entities']:
            pdbid = polymer_entities['rcsb_id'].split("_")[0]

            for ii in polymer_entities['polymer_entity_instances']: # NOTE Chains
                pdbid_chainid = "%s"%(pdbid)+ "_"+ ii['rcsb_polymer_entity_instance_container_identifiers']['auth_asym_id']
                MmseqDict[clusterid].append(pdbid_chainid)

                # TODO Build culmulated dictionary
                CulmulatedEntityChainDict[polymer_entities['rcsb_id']].append(pdbid_chainid)



      # TODO Direct update from culmulated dicitonary. pass when empty list
      for polymer_entityid in list_pdbid_entityid_knownchainid:
          MmseqDict[clusterid].extend(CulmulatedEntityChainDict[polymer_entityid])
      #print(len(CulmulatedEntityChainDict.keys()))
      #print(CulmulatedEntityChainDict)
      #jj -=1

  return MmseqDict, FailingClusters_, CulmulatedEntityChainDict











# =======================================
# Blast CLust Clan Graph related 
# ====================================




# Read In Check if any potential symmetric chain content in pdbid
def ReadBCInternalSymmetry(folder):
  BCDict = defaultdict(dict)
  collector={}
  for percent in ["100","95","90","70","50","40","30"]:
    temp = defaultdict(list)
    with open("%s/bc-%s.out"%(folder,percent),"r") as f:
      content = f.readlines()
      # i is the cluster id
      for i in range(len(content)):
          temptemp = defaultdict(list)
          for j in content[i].rstrip().split(" "):
              #temp[j.split("_")[0].lower()].append(j.split("_")[1])
              temptemp[j.split("_")[0].lower()].append(j.split("_")[1])
          for k in temptemp.keys():
              temp[k].append(list(temptemp[k]))
          del temptemp
    for key in temp.keys():
        bigstring = ""
        for l in temp[key]:
           midstring=""
           for m in l:
               midstring+=m
               midstring+=str(",")
           midstring = midstring[0:-1]
           bigstring+=midstring
           bigstring+=str("_")
        bigstring = bigstring[0:-1]
        temp[key] = bigstring

    t = temp.items()
    Pdbids = [u[0] for u in t]
    Chains = [u[1] for u in t]
    df = pd.DataFrame({'Pdbid': list(Pdbids), "InternalSymmetryBC-%s"%(percent):list(Chains)})
    #print (df)
    collector[percent] = df

  df = collector["95"]
  #df = pd.merge(collector["100"], collector["95"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["90"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["70"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["50"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["40"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["30"], how='outer', on = 'Pdbid')
  #print(df)
  return df

def ReadBCExternalSymmetry(folder):
  """
  # NOTE
  # This reads all available bc-{percent}.out files curated by pdb.
  # This includes all the pdbid (even for those w/o NA)
  # The output is BCDict is {BC_percent: {BC_clusterid: {pdbid: [chains included]}}}
  # To retrieve just pdbids within the same cluster. we can [sorted(clustermates[0].keys()) for clustermates in BCDict[100].values()]
  """
  BCDict = {}
  for percent in ["100","95","90","70","50","40","30"]:
    BCDict_local = defaultdict(list)
    with open("%s/bc-%s.out"%(folder,percent),"r") as f:
      content = f.readlines()
      # i is the cluster id for individual blast percent
      for i in range(len(content)):
          temptemp = defaultdict(list)
          for j in content[i].rstrip().split(" "):
              temptemp[j.split("_")[0].lower()].append(j.split("_")[1])
          BCDict_local[i].append(temptemp)

    BCDict[int(percent)] = BCDict_local


  print("Number of clusters in each BC percent")
  for percent in [100,95,90,70,50,40,30]:
    print('BC%s\t' %(percent), len([sorted(clustermates[0].keys()) for clustermates in BCDict[percent].values()]))
    
  return BCDict

def MakeBcClanGraph(BCDict, BlastClustPercent = 70, 
    DIR_DataframeGrand_CoordSupplement ="../Database-PDB/DerivedData/DataframeGrand_CoordSupplement.pkl",
    ):

    """
    # NOTE 
    # This script further cluster the pdbid by overlapping biological assembly.
    # 1. Naive overlap without considering the bound chain will not work well because it can containing agents 
    #    e.g. Fab, Gfp, etc common for purification in structural biology 
    # 2. multiple BC clusts are joined with a "pdbid edge". Connected component on this graph is called a clan.
    #    We will train on redundant sample and validate on non-redundant sample (when available) from each cluster 
    #    We will test on each clan
    #  
    """

    CalculateJaccard = True



    # =============================================
    # Make a dict of Nuc Bound Protein chain
    # =============================================
    PART0_FindNucBoundChains = True
    if PART0_FindNucBoundChains:
        Df_grand = pd.read_pickle(DIR_DataframeGrand_CoordSupplement)
        Df_grandpronuc = Df_grand.loc[(Df_grand["ProNu"] == "prot-nuc")]

        pdbavail = sorted(set(Df_grandpronuc['Pdbid'].tolist()))
        NucBound = Df_grandpronuc[['Pdbid', 'NucBoundChains']].values.tolist()
        NucBound_dict = {}
        for i in NucBound:
            #if type(i[1]) is float: # NOTE These are pdb not downloaded
            #    pass
            #else:
            if type(i[1]) is float:
                NucBound_dict[i[0]] = []
                #if len(i[1]) == 0: # NOTE These are without any Nuc Chain
                #    continue
                    #print(i[1])
            else:
                NucBound_dict[i[0]] = i[1]





    # =================================================
    # Mapping membership of PDBID to BlastClust  
    # =================================================
    # NOTE Any connected component on the graph produce here is a potential overlap in biological assembly i.e. is a clan

    PART1_MakeClanGraph = True
    if PART1_MakeClanGraph:
        print("Making Clan Graph")
        G = nx.Graph()

        # NOTE clusterid will be reused as the count of clusters
        clusterid = 0
        for clustermates in tqdm.tqdm(BCDict[BlastClustPercent].values()):

            clustermates_intouch = []
            for pdbid, chainlist in clustermates[0].items():

                # NOTE This removes those not in grand df
                if pdbid not in pdbavail:
                    continue

                # NOTE This is important as it will remove Fab and other purifying agent. 
                #      Also note that there ARE NA that are intended for Fab binding!
                if pdbid not in NucBound_dict.keys():
                    continue

                if len(set(NucBound_dict[pdbid]) & set(chainlist)) > 0:
                    clustermates_intouch.append(pdbid)

            clustermates_intouch = sorted(set(clustermates_intouch))
            G.add_weighted_edges_from([(clusterid,i,0.01) for i in clustermates_intouch], attr = "BC%s_BcPdbidMembership" %(BlastClustPercent))

            clusterid +=1
        
        # Set Pdbid Attributes 
        G_nodes = G.nodes()
        for atrib in Df_grandpronuc.columns:

            #print(atrib)
            NucContent = Df_grandpronuc[['Pdbid', atrib]].values.tolist()

            for i in NucContent:
                if i[0] not in G_nodes:
                    continue
                G.nodes[i[0]][atrib] = i[1]



    # NOTE We can terminate the program here if we just want to find a training-testing dataset
    if CalculateJaccard:
        pass
    else:
        return G
    
    # =========================================
    # Find out the Relation among bc
    # =========================================
    
    PART2_CalculateJaccard = True
    if PART2_CalculateJaccard:
        print("Calculating Jaccard similarity (Overlap of Pdbids) among Blast Clusters")
        content_in_bc = {}
        for i in range(clusterid):
            if G.has_node(i):
                ii = list(G.neighbors(i))
                if len(ii) == 0:
                    continue
                content_in_bc[i] = ii

        jaccard_edge = []    
        for i in tqdm.tqdm(sorted(content_in_bc.keys())):
            for j in sorted(content_in_bc.keys()):
                if i >= j:
                    continue
                try:
                    jjj_nom = len(set(content_in_bc[i]) & set(content_in_bc[j]))
                    if jjj_nom == 0:
                        continue
                    jjj_denom = len(set(content_in_bc[i]) | set(content_in_bc[j]))
                    jjj = jjj_nom / jjj_denom
                except ZeroDivisionError: # NOTE Many of these has no connection at all!
                    continue

                jaccard_edge.append((i,j,jjj))

        G.add_weighted_edges_from(jaccard_edge, attr = "BC%s_RelationAmongBc" %(BlastClustPercent))


        # =========================================
        # Find out the Relation among pdbid
        # =========================================
        print("Calculating Jaccard similarity (Overlap of Blast Clusters) among Pdbid")
        # Get pdbid in cluster
        pdbid_ingraph = [i for i in list(G.nodes) if type(i) is str]
        content_in_pdbid = {}
        for i in pdbid_ingraph:
            if G.has_node(i):
                content_in_pdbid[i] = list(G.neighbors(i))

        # Calculate jaccard between pdbid
        jaccard_edge = []
        sorted_pdbid = sorted(content_in_pdbid.keys())
        for i in tqdm.tqdm(range(len(sorted_pdbid))):
            for j in range(len(sorted_pdbid)):
                if i >= j:
                    continue
                try:
                    jjj_nom = len(set(content_in_pdbid[sorted_pdbid[i]]) & set(content_in_pdbid[sorted_pdbid[j]]))
                    if jjj_nom == 0:
                        continue
                    jjj_denom =  len(set(content_in_pdbid[sorted_pdbid[i]]) | set(content_in_pdbid[sorted_pdbid[j]]))
                    jjj =  jjj_nom / jjj_denom
                except ZeroDivisionError: # NOTE Many of these has no connection at all!
                    continue
                jaccard_edge.append((sorted_pdbid[i],sorted_pdbid[j],jjj))

        G.add_weighted_edges_from(jaccard_edge, attr = "BC%s_RelationAmongPdbid" %(BlastClustPercent))

    return G

def OOC_GetMaxCliqueSize(connected_component_index, connected_component_lol = [], test_G_pdbid = None):
        c_ = connected_component_lol[connected_component_index]
        clansize = len(c_)
        if clansize > 2:
            mcsize = nx.algorithms.clique.node_clique_number(test_G_pdbid, nodes = sorted(c_))
            maxcliquesize_mean = np.mean(sorted(mcsize.values())) 
            maxcliquesize_max = np.max(sorted(mcsize.values()))
            maxcliquesize_min = np.min(sorted(mcsize.values()))
            #print(clansize)
        else:
            mcsize = clansize
            maxcliquesize_mean = clansize
            maxcliquesize_max = clansize
            maxcliquesize_min = clansize
        
        percent_vertexcover_mean = maxcliquesize_mean / clansize
        percent_vertexcover_min = maxcliquesize_min / clansize
        percent_vertexcover_max = maxcliquesize_max / clansize
        return (connected_component_index, clansize, maxcliquesize_mean, maxcliquesize_max, maxcliquesize_min, 
                percent_vertexcover_mean, percent_vertexcover_max, percent_vertexcover_min, mcsize)

def GetMaxCliqueSize(DIR_DerivedData = "../Database-PDB/DerivedData/", bc_percent = 100):

        df_maxclique = []
        with open(DIR_DerivedData+ 'ClanGraph_%s.pkl' %(bc_percent), 'rb') as fn:
            test_G = pickle.load(fn)
        
        test_G_BC = test_G.subgraph([i for i in test_G.nodes if type(i) is int])
        test_G_pdbid = test_G.subgraph([i for i in test_G.nodes if type(i) is str])

        

        connected_component_lol = sorted(nx.connected_components(test_G_pdbid), key=len, reverse=True)
        OOC_GetMaxCliqueSize_partial = partial(OOC_GetMaxCliqueSize, connected_component_lol = connected_component_lol, test_G_pdbid = test_G_pdbid)

        cc_chunk = ChunkList(range(len(connected_component_lol)), 50)
        tempdf = []
        for chunk in tqdm.tqdm(cc_chunk):
            pool = multiprocessing.Pool(25)
            temptempdf = pool.map(OOC_GetMaxCliqueSize_partial, chunk)
            tempdf.extend(temptempdf)
            pool.close()
            KillInactiveKernels(cpu_threshold = 0.1)

        for cc_info in tempdf:
            clanid, clansize, maxcliquesize_mean, maxcliquesize_max, maxcliquesize_min, \
                percent_vertexcover_mean, percent_vertexcover_max, percent_vertexcover_min, mcsize = cc_info
            templine = [bc_percent, clanid, clansize,
                        maxcliquesize_mean, maxcliquesize_max, maxcliquesize_min, 
                        percent_vertexcover_mean, percent_vertexcover_max, percent_vertexcover_min, 
                        mcsize]
            c_ = connected_component_lol[clanid]
            NucCount = collections.Counter([test_G_pdbid.nodes[i]['NucleicAcid'] for i in sorted(c_)])        
            for i in ['dna','rna', 'hybrid']:
                try:
                    templine.append(NucCount[i])
                except KeyError:
                    templine.append(0)

            df_maxclique.append(templine)

        df_maxclique = pd.DataFrame(df_maxclique, columns=["BC Percent", "BC Clan ID", "Clan Size", 
                                                            "Mean Size of Max Clique", "Max Size of Max Clique", "Min Size of Max Clique",
                                                            "Mean Percent Vertex Cover", "Max Percent Vertex Cover", "Min Percent Vertex Cover", 
                                                            "Pdbid Size Max Clique Dict",
                                                            "Dna", "Rna", "Hybrid"])

        del test_G, test_G_pdbid, test_G_BC
        gc.collect()

        return df_maxclique




# =======================================
# MMseq2 CLust Clan Graph related 
# ====================================
# NOTE This is esentially the same as bclust above but some namings changed accordingly. 



# Read In Check if any potential symmetric chain content in pdbid
def ReadMmseqInternalSymmetry(folder):
  MmseqDict = defaultdict(dict)
  collector={}
  for percent in ["100","95","90","70","50","40","30"]:
    temp = defaultdict(list)
    with open("%s/mmseq-%s.out"%(folder,percent),"r") as f:
      content = f.readlines()
      # i is the cluster id
      for i in range(len(content)):
          temptemp = defaultdict(list)
          for j in content[i].rstrip().split(" "):
              #temp[j.split("_")[0].lower()].append(j.split("_")[1])
              temptemp[j.split("_")[0].lower()].append(j.split("_")[1])
          for k in temptemp.keys():
              temp[k].append(list(temptemp[k]))
          del temptemp
    for key in temp.keys():
        bigstring = ""
        for l in temp[key]:
           midstring=""
           for m in l:
               midstring+=m
               midstring+=str(",")
           midstring = midstring[0:-1]
           bigstring+=midstring
           bigstring+=str("_")
        bigstring = bigstring[0:-1]
        temp[key] = bigstring

    t = temp.items()
    Pdbids = [u[0] for u in t]
    Chains = [u[1] for u in t]
    df = pd.DataFrame({'Pdbid': list(Pdbids), "InternalSymmetryMmseq-%s"%(percent):list(Chains)})
    #print (df)
    collector[percent] = df

  df = collector["95"]
  #df = pd.merge(collector["100"], collector["95"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["90"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["70"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["50"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["40"], how='outer', on = 'Pdbid')
  #df = pd.merge(df, collector["30"], how='outer', on = 'Pdbid')
  #print(df)
  return df

def ReadMmseqExternalSymmetry(folder):
  """
  # NOTE
  # This reads all available Mmseq-{percent}.out files curated by pdb.
  # This includes all the pdbid (even for those w/o NA)
  # The output is MmseqDict is {Mmseq_percent: {Mmseq_clusterid: {pdbid: [chains included]}}}
  # To retrieve just pdbids within the same cluster. we can [sorted(clustermates[0].keys()) for clustermates in MmseqDict[100].values()]
  """
  MmseqDict = {}
  for percent in ["100","95","90","70","50","40","30"]:
    MmseqDict_local = defaultdict(list)
    with open("%s/mmseq-%s.out"%(folder,percent),"r") as f:
      content = f.readlines()
      # i is the cluster id for individual Mmseq percent
      for i in range(len(content)):
          temptemp = defaultdict(list)
          for j in content[i].rstrip().split(" "):
              temptemp[j.split("_")[0].lower()].append(j.split("_")[1])
          MmseqDict_local[i].append(temptemp)

    MmseqDict[int(percent)] = MmseqDict_local


  print("Number of clusters in each Mmseq percent")
  for percent in [100,95,90,70,50,40,30]:
    print('Mmseq%s\t' %(percent), len([sorted(clustermates[0].keys()) for clustermates in MmseqDict[percent].values()]))
    
  return MmseqDict

def MakeMmseqClanGraph(MmseqDict, MmseqClustPercent = 70, 
    DIR_DataframeGrand_CoordSupplement ="../Database-PDB/DerivedData/DataframeGrand_CoordSupplement.pkl",
    ):

    """
    # NOTE 
    # This script further cluster the pdbid by overlapping biological assembly.
    # 1. Naive overlap without considering the bound chain will not work well because it can containing agents 
    #    e.g. Fab, Gfp, etc common for purification in structural biology 
    # 2. multiple Mmseq clusts are joined with a "pdbid edge". Connected component on this graph is called a clan.
    #    We will train on redundant sample and validate on non-redundant sample (when available) from each cluster 
    #    We will test on each clan
    #  
    """

    CalculateJaccard = True



    # =============================================
    # Make a dict of Nuc Bound Protein chain
    # =============================================
    PART0_FindNucBoundChains = True
    if PART0_FindNucBoundChains:
        Df_grand = pd.read_pickle(DIR_DataframeGrand_CoordSupplement)
        Df_grandpronuc = Df_grand.loc[(Df_grand["ProNu"] == "prot-nuc")]

        pdbavail = sorted(set(Df_grandpronuc['Pdbid'].tolist()))
        NucBound = Df_grandpronuc[['Pdbid', 'NucBoundChains']].values.tolist()
        NucBound_dict = {}
        for i in NucBound:
            #if type(i[1]) is float: # NOTE These are pdb not downloaded
            #    pass
            #else:
            if type(i[1]) is float:
                NucBound_dict[i[0]] = []
                #if len(i[1]) == 0: # NOTE These are without any Nuc Chain
                #    continue
                    #print(i[1])
            else:
                NucBound_dict[i[0]] = i[1]





    # =================================================
    # Mapping membership of PDBID to MmseqClust  
    # =================================================
    # NOTE Any connected component on the graph produce here is a potential overlap in biological assembly i.e. is a clan

    PART1_MakeClanGraph = True
    if PART1_MakeClanGraph:
        print("Making Clan Graph")
        G = nx.Graph()

        # NOTE clusterid will be reused as the count of clusters
        clusterid = 0
        for clustermates in tqdm.tqdm(MmseqDict[MmseqClustPercent].values()):

            clustermates_intouch = []
            for pdbid, chainlist in clustermates[0].items():

                # NOTE This removes those not in grand df
                if pdbid not in pdbavail:
                    continue

                # NOTE This is important as it will remove Fab and other purifying agent. 
                #      Also note that there ARE NA that are intended for Fab binding!
                if pdbid not in NucBound_dict.keys():
                    continue

                if len(set(NucBound_dict[pdbid]) & set(chainlist)) > 0:
                    clustermates_intouch.append(pdbid)

            clustermates_intouch = sorted(set(clustermates_intouch))
            G.add_weighted_edges_from([(clusterid,i,0.01) for i in clustermates_intouch], attr = "Mmseq%s_MmseqPdbidMembership" %(MmseqClustPercent))

            clusterid +=1
        
        # Set Pdbid Attributes 
        G_nodes = G.nodes()
        for atrib in Df_grandpronuc.columns:

            #print(atrib)
            NucContent = Df_grandpronuc[['Pdbid', atrib]].values.tolist()

            for i in NucContent:
                if i[0] not in G_nodes:
                    continue
                G.nodes[i[0]][atrib] = i[1]



    # NOTE We can terminate the program here if we just want to find a training-testing dataset
    if CalculateJaccard:
        pass
    else:
        return G
    
    # =========================================
    # Find out the Relation among Mmseq
    # =========================================
    
    PART2_CalculateJaccard = True
    if PART2_CalculateJaccard:
        print("Calculating Jaccard similarity (Overlap of Pdbids) among Mmseq Clusters")
        content_in_Mmseq = {}
        for i in range(clusterid):
            if G.has_node(i):
                ii = list(G.neighbors(i))
                if len(ii) == 0:
                    continue
                content_in_Mmseq[i] = ii

        jaccard_edge = []    
        for i in tqdm.tqdm(sorted(content_in_Mmseq.keys())):
            for j in sorted(content_in_Mmseq.keys()):
                if i >= j:
                    continue
                try:
                    jjj_nom = len(set(content_in_Mmseq[i]) & set(content_in_Mmseq[j]))
                    if jjj_nom == 0:
                        continue
                    jjj_denom = len(set(content_in_Mmseq[i]) | set(content_in_Mmseq[j]))
                    jjj = jjj_nom / jjj_denom
                except ZeroDivisionError: # NOTE Many of these has no connection at all!
                    continue

                jaccard_edge.append((i,j,jjj))

        G.add_weighted_edges_from(jaccard_edge, attr = "Mmseq%s_RelationAmongMmseq" %(MmseqClustPercent))


        # =========================================
        # Find out the Relation among pdbid
        # =========================================
        print("Calculating Jaccard similarity (Overlap of Mmseq Clusters) among Pdbid")
        # Get pdbid in cluster
        pdbid_ingraph = [i for i in list(G.nodes) if type(i) is str]
        content_in_pdbid = {}
        for i in pdbid_ingraph:
            if G.has_node(i):
                content_in_pdbid[i] = list(G.neighbors(i))

        # Calculate jaccard between pdbid
        jaccard_edge = []
        sorted_pdbid = sorted(content_in_pdbid.keys())
        for i in tqdm.tqdm(range(len(sorted_pdbid))):
            for j in range(len(sorted_pdbid)):
                if i >= j:
                    continue
                try:
                    jjj_nom = len(set(content_in_pdbid[sorted_pdbid[i]]) & set(content_in_pdbid[sorted_pdbid[j]]))
                    if jjj_nom == 0:
                        continue
                    jjj_denom =  len(set(content_in_pdbid[sorted_pdbid[i]]) | set(content_in_pdbid[sorted_pdbid[j]]))
                    jjj =  jjj_nom / jjj_denom
                except ZeroDivisionError: # NOTE Many of these has no connection at all!
                    continue
                jaccard_edge.append((sorted_pdbid[i],sorted_pdbid[j],jjj))

        G.add_weighted_edges_from(jaccard_edge, attr = "Mmseq%s_RelationAmongPdbid" %(MmseqClustPercent))

    return G

def OOC_MmseqGetMaxCliqueSize(connected_component_index, connected_component_lol = [], test_G_pdbid = None):
        c_ = connected_component_lol[connected_component_index]
        clansize = len(c_)
        if clansize > 2:
            mcsize = nx.algorithms.clique.node_clique_number(test_G_pdbid, nodes = sorted(c_))
            maxcliquesize_mean = np.mean(sorted(mcsize.values())) 
            maxcliquesize_max = np.max(sorted(mcsize.values()))
            maxcliquesize_min = np.min(sorted(mcsize.values()))
            #print(clansize)
        else:
            mcsize = clansize
            maxcliquesize_mean = clansize
            maxcliquesize_max = clansize
            maxcliquesize_min = clansize
        
        percent_vertexcover_mean = maxcliquesize_mean / clansize
        percent_vertexcover_min = maxcliquesize_min / clansize
        percent_vertexcover_max = maxcliquesize_max / clansize
        return (connected_component_index, clansize, maxcliquesize_mean, maxcliquesize_max, maxcliquesize_min, 
                percent_vertexcover_mean, percent_vertexcover_max, percent_vertexcover_min, mcsize)

def MmseqGetMaxCliqueSize(DIR_DerivedData = "../Database-PDB/DerivedData/", Mmseq_percent = 100):

        df_maxclique = []
        with open(DIR_DerivedData+ 'MmseqClanGraph_%s.pkl' %(Mmseq_percent), 'rb') as fn:
            test_G = pickle.load(fn)
        
        test_G_Mmseq = test_G.subgraph([i for i in test_G.nodes if type(i) is int])
        test_G_pdbid = test_G.subgraph([i for i in test_G.nodes if type(i) is str])

        

        connected_component_lol = sorted(nx.connected_components(test_G_pdbid), key=len, reverse=True)
        OOC_MmseqGetMaxCliqueSize_partial = partial(OOC_MmseqGetMaxCliqueSize, connected_component_lol = connected_component_lol, test_G_pdbid = test_G_pdbid)

        cc_chunk = ChunkList(range(len(connected_component_lol)), 50)
        tempdf = []
        for chunk in tqdm.tqdm(cc_chunk):
            pool = multiprocessing.Pool(25)
            temptempdf = pool.map(OOC_MmseqGetMaxCliqueSize_partial, chunk)
            tempdf.extend(temptempdf)
            pool.close()
            KillInactiveKernels(cpu_threshold = 0.1)

        for cc_info in tempdf:
            clanid, clansize, maxcliquesize_mean, maxcliquesize_max, maxcliquesize_min, \
                percent_vertexcover_mean, percent_vertexcover_max, percent_vertexcover_min, mcsize = cc_info
            templine = [Mmseq_percent, clanid, clansize,
                        maxcliquesize_mean, maxcliquesize_max, maxcliquesize_min, 
                        percent_vertexcover_mean, percent_vertexcover_max, percent_vertexcover_min, 
                        mcsize]
            c_ = connected_component_lol[clanid]
            NucCount = collections.Counter([test_G_pdbid.nodes[i]['NucleicAcid'] for i in sorted(c_)])        
            for i in ['dna','rna', 'hybrid']:
                try:
                    templine.append(NucCount[i])
                except KeyError:
                    templine.append(0)

            df_maxclique.append(templine)

        df_maxclique = pd.DataFrame(df_maxclique, columns=["Mmseq Percent", "Mmseq Clan ID", "Clan Size", 
                                                            "Mean Size of Max Clique", "Max Size of Max Clique", "Min Size of Max Clique",
                                                            "Mean Percent Vertex Cover", "Max Percent Vertex Cover", "Min Percent Vertex Cover", 
                                                            "Pdbid Size Max Clique Dict",
                                                            "Dna", "Rna", "Hybrid"])

        del test_G, test_G_pdbid, test_G_Mmseq
        gc.collect()

        return df_maxclique

