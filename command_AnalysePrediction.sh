
#=============================================================

# Downstream III: Package Deep learning output from NucleicNet 

#=============================================================


SCRIPT_HOME='Script'                # Hold Python scripts
Prediction_DIR='DL_output'          # Hold Deep Learning Ouputs
Output_DIR='Out'                    # Hold Readable Result
KnownSite_DIR='Control'             # Hold RNA-Protein Complexes for Sequence Logo prediction
Working_DIR='GridData'              # Hold Protein to be analysed and files generated in previous steps
SeqTxt_DIR='ExperimentalSequencing' # Hold Experiemental argonaute sequencing data


# ================================
# 0. Package Deep Learning Results
# ================================
for i in GridData/*.pdb 
do
j=$(echo ${i} | sed 's/GridData\///g' | sed 's/.pdb//g')
python ${SCRIPT_HOME}/NucleicNet_PreprocessingDLresult.py --TargetPdbFolder ${Working_DIR} --PredictionFolder ${Prediction_DIR} --OutputFolder ${Output_DIR} --Pdbid ${j}
done
date

# ==================
# 1. General RBP    
# ==================

# a. Visualise binding sites as a pymol session
for i in GridData/*.pdb
do
j=$(echo ${i} | sed 's/GridData\///g' | sed 's/.pdb//g')
python ${SCRIPT_HOME}/NucleicNet_VisualisePymol.py --Pdbid ${j} --OutputFolder ${Output_DIR} --ApoFolder ${Working_DIR}
done
date

# b. Plot Sequence Logo for protein analysed where RNA binding sites are known beforehand
# The pdb file that also contains the RNA should be put in ${KnownSite_DIR}
for i in GridData/*.pdb
do
j=$(echo ${i} | sed 's/GridData\///g' | sed 's/.pdb//g')
python ${SCRIPT_HOME}/NucleicNet_SequenceLogo_RNACcolor.py --KnownSitePDBFolder ${KnownSite_DIR} --PredictionFolder ${Prediction_DIR} --OutputFolder ${Output_DIR} --Pdbid ${j}
done

# ==========================
# 2. Human Argonaute 2
# ==========================
# a. Comparison with RipSeq
python ${SCRIPT_HOME}/NucleicNet_HMM_Ago_RipSeq.py --SeqTxtFolder ${SeqTxt_DIR}

# b. Comparison with Knockdown benchmark
python ${SCRIPT_HOME}/NucleicNet_HMM_Ago_Knockdown.py --SeqTxtFolder ${SeqTxt_DIR} 

# c. Score miRNA sequences
# The following script allows users to score individual miRNA sequence using the trained HMM model
python ${SCRIPT_HOME}/NucleicNet_HMM_Ago_ScoreSequences.py --TestSequences AAAUCCACAGCUACUUAUGCC,UAAAGGACGGUCAAGUUCAUG



