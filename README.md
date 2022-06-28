# NucleicNet

Protein-RNA interaction is ubiquitous in cells and serves as the main mechanism for post-transcriptional regulation. Base-dominant interaction and backbone-dominant interaction categorize the two main modes of the way RNA interacting with proteins. Despite the advances in experimental technologies and computational methods to capture protein-RNA interactions, estimating binding preference of RNA backbone constituents and different bases on any location of a given protein surface is beyond the capacity of existing techniques. Here we show that these attributes can be predicted from the local physicochemical characteristics of the protein structure surface, by leveraging the power of deep learning. 

## How to cite us?
* Lam, J.H., Li, Y., Zhu, L. et al. A deep learning framework to predict binding preference of RNA constituents on protein surface. Nat Commun 10, 4941 (2019). https://doi.org/10.1038/s41467-019-12920-0

## Acknowledgement
* 


# System and hardware requirements
Installation of our software is handled in `Notebook 00`. We only support Linux workstations. Below are specs where the software has been tested.

* System. Ubuntu 20.04.3 LTS
* Processor. AMD Opteron(tm) processor 6376 × 32
* Storage. More than 500 GB storage. Git clone of the repo takes 12GB, compressed. After all the decompressions in `Notebook 00`, it can take > 85 GB. See the breakdown in `Notebook 00`
* RAM. At least 32 GB.
* Nvidia GPU. On par with a GeForce GTX 1070



# Release Notes

## v1.1 June 2022

This repository presents NucleicNet as a more developer-friendly package closely reproduces our work in 2019. Since early 2022, we have been working on modularising NuleicNet and simplifying its installation and its dependecies. The package presented here handles recent changes in PDB's query API as well as the ever increasing sizes of structures due to cryo-EM produced in recent years. We have also migrated from Tensorflow 1 to Pytorch as TF1 is difficult to maintain and stay competitive. All the codes are now written with python 3. Walkthroughs for dataset-building, training and downstream applications are included in Notebooks. 

* `Notebook 00` is for installation of our software.
* `Notebook 01-03` is for production of the dataset taking care of PDB structures <= 2021.
* `Notebook 04` is optional for understanding the structure and basic statistics of Altman's feature vector.
* `Notebook 05` is for training of models and `Notebook 06` for 3-fold cross validation. 
* `Notebook 07` is for downstream applications (pymol visualisation and Sequence logo) as in service provided by our webserver.

The NucleicNet is distributed under GNU AGPLv3. This code is meant to serve as a (great!) tutorial, and a basis for researchers to explore protein-surface learning tasks. We hope you all enjoy. To reproduce the results for the paper exactly as published go to our [legacy webserver this way!](http://www.cbrc.kaust.edu.sa/NucleicNet/). A few essential improvements built upon our paper in 2019.

### Dataset
* We have updated our dataset to RNA-bound PDB entries released before 2022. Importantly, not all RNA-bound entries carry base preference; examples of non-specific binding are shape-dependent components with disproportionately large amount of backbone contact e.g. ribosome, trna, etc. Again, literature accompanying the structure is consulted to avoid including close-contact-but-no-preference cases. 
* Several PDB entries are obsolete or superceded. They are excluded from consideration.
* Sparse Storage of Altman features. We have shifted away from storing a dense matrix of Altman features to a CSR format to handle the ever-growing structure size on PDB. 

### Large Batch Training Strategy
Previously, we use a very small batch size (128) to train our Resnet. But since the data size has grown more than 2 times in 4 years, the training of models even slower than before. Accordingly, we updated our training strategy to keep up with the pace of our machine learning colleagues. (Meanwhile, we still stick to the baseline Resnet model as proposed in the paper!)

* We implemented a Pytorch dataloader that efficiently incorporates our discussion on avoiding data imbalance attributed to redundancy in sequence homologue. 
* [Ghost Batch Normalization](https://arxiv.org/pdf/1705.08741.pdf) is used to facillitate larger batch training. 
* We follow the guiding principle outlined in [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489) to handle large batch training and learning rate tests
* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186). To facillitate production of an ensemble of models and to test the learning rate in a large dataset, we used a cosine cyclical learning rate with a multistep decay rather than a simple multistep decay.
* [AdamW](https://arxiv.org/pdf/1711.05101.pdf), which decouples the weight decay from gradient update, is used in lieu of Adam.

### Visualisation
* Rather than visualising only a fixed percentile of multi-label voxels. We implemented a thresholding scheme based on bit fraction (fractional height on logo diagrams) at each location. This allows a more systematic visualisation of multi-labels.

## Upcoming v 1.2
- [ ] Mmseq2. [Blast Clust is retired from PDB since April 2022](https://groups.google.com/a/rcsb.org/g/api/c/ALLI4pouK_w). While we kept a version from January 2022, we will likely transit to the MMseq2 clusters.
- [ ] NPIDB. NPIDB is no longer available. We will remove this dependency completely in the future.
- [ ] Cif and Ensemble. A better handle for structures with size exceeding pdb format limit. [PDB finally distribute assembly cif in May 2022!](https://www.rcsb.org/news/feature/62559153c8eabd0c4864f208)
- [ ] Bayesian hierarchical model. For base classication considering pyrimidine, purine and null. For site non site classification, considering site non-site. 


# FAQ

0. Q. How to get started? What to look for?
*  A. Walkthroughs for dataset-building, training and downstream applications are included in `Notebooks` folder. 
* * `Notebook 00` is for installation of our software.
* * `Notebook 00-03` is for production of the dataset taking care of PDB structures <= 2021.
* * `Notebook 04` is optional for understanding the structure and basic statistics of Altman's feature vector.
* * `Notebook 05` is for training of models and `Notebook 06` for 3-fold cross validation. 
* * `Notebook 07` is for downstream applications (pymol visualisation and Sequence logo) as in service provided by our webserver.



1. Q. Dataset. We need PDBID, grid coordinates and the corresponding labels. Where can I find it?
*  A. You will find the coordinate under `Database-PDB/halo/{pdbid}{conformerid}.xyz` and the corresponding labels under `Database-PDB/typi/{pdbid}{conformerid}.typi`. The PDBIDs for cross-fold classification is stored as a pickled pandas dataframe at `Database-PDB/DerivedData/TaskClanFoldDf_TaskAUCG_BC90_Fold9.pkl` and `Database-PDB/DerivedData/TaskClanFoldDf_TaskSXPR_BC90_Fold9.pkl`. But, for a fresh new preparation, follow through `Notebook 00 - 03`. 

2. Q. Sequence redundancy handle. You mentioned a lot about avoidance of internal and external redundancy, but how to incorporate these philosophies as a training strategy.
*  A. We make it easier for developers. The folds indicated in `Database-PDB/DerivedData/TaskClanFoldDf_Task*_BC90_Fold9.pkl` are readily separated using blast-clust 90. (See relevant codes in Notebook 00 - 03 on how to do it fresh new.) In general, these redundancies refer to copies of the same protein existing among different PDB entries (external) and within the same PDB entry (internal). They can be handled by clustering homologous sequences (e.g. BlastClust or MMseq2) and grouping entries sharing clusters as clans. For handling external redundancy, cross fold validation is done on disjoint clans. For handling internal redundancy, weighted sampling is done to retrieve equal amount of sample from each clan in each batch during training. These strategies are incorporated into our dataloaders. Since [April 2022 BlastClust is retired from PDB](https://groups.google.com/a/rcsb.org/g/api/c/ALLI4pouK_w) and we are still figuring out how to do it with MMseq2.

3. Q. Sanitization of Coordinates. We understand that raw PDB coordinates are difficult to process and the structure does matters a lot for a structure-based software. Garbage-in-garbage-out. How should we prepare for an amenable 'okay' PDB coordinate input? 
*  A. We provide a very basic sanitization protocol when calling Notebook 7 as a downstream application, but for a very detailed protocol we used to build the training coordinate in dataset, please consult to `NucleicNet/DatasetBuilding/commandCoordinate.py`. We also provide some general guidelines on [our structured Wiki page](https://github.com/jhmlam/NucleicNet/wiki/Specification-on-PDB-input-files); we will write more about this. Examples of acceptable files are stored in the `LocalServerExample/*.pdb`.
