
#=============================================================

# Downstream I: Produce Grid FEATURE for given protein model

#=============================================================


SCRIPT_HOME='Script'
Working_DIR='GridData'                 # Hold the protein atomic model to be analysed
FEATURE_DIR='feature-3.1.0/data/'      # Hold the FEATURE program
FEATURE_property='proteins.properties' # Hold the property names to be analysed by the FEATURE program

# 0. Dssp
python ${SCRIPT_HOME}/NucleicNet_MakeDssp.py \
	--DsspFolder ${Working_DIR} \
	--PdbFolders ${Working_DIR} 

# 1. Create Ptf
python ${SCRIPT_HOME}/NucleicNet_MakePtf.py \
	--Distance 5.0 \
	--BlindPdbFolder ${Working_DIR} --VoronoiCellTruncation 5.00 \
	--MidlineHalo 2.50,5.50 --AwayPocketNucleic 5.50


# 2. Make raw FEATURE.ff files
python ${SCRIPT_HOME}/NucleicNet_MakeFF.py \
	--ApoFolder ${Working_DIR} \
	--DsspFolder ${Working_DIR} \
	--TrainingFolder ${Working_DIR} \
	--FeatureFolder ${FEATURE_DIR}

# 3. Process into python pickle files
python ${SCRIPT_HOME}/NucleicNet_ProcessFF.py \
	--Shells 6 \
	--TrainingFolder ${Working_DIR} \
	--FeatureFolder ${FEATURE_DIR} \
	--PropertyNameFile ${FEATURE_property}

