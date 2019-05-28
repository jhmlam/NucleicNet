
#===================================================================

# Downstream II: Analyse on pickled FEATURE for individual proteins

#===================================================================

grid_file_folder='./GridData' # Hold Features from previous step

# Users should load their own cuda cudnn before running the program.
module unload cuda
#module load cuda/9.0.176-cudNN7.0 
module load cuda/8.0.61-cudNN5.1


# Remove any old result 
rm ./DL/dl_input/*

# Copy the pickled features into DL/dl_input
cp ${grid_file_folder}/*.pkl ./DL/dl_input

# Run the deep learning code
cd DL/src_productive
sh main.sh

