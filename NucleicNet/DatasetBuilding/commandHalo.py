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
import itertools
import pickle
import gc
import tqdm
from functools import partial
import multiprocessing

from NucleicNet.DatasetBuilding.util import *

def OOC_Lattice_TrimIntersect(lattice_coord, lattice_index, target_coord, intersect_radius = 5.0):
        # NOTE lattice_coord and _index are the current remainings
        # NOTE lattice_surface_index finds the intersecting row_index
        lattice_tree = spatial.cKDTree(lattice_coord)
        lattice_surface_index = lattice_tree.query_ball_point(target_coord, intersect_radius, 
                                                            p=2.0, eps=0, workers=1, return_sorted=None, 
                                                            return_length=False)
        lattice_surface_index = sorted(set(itertools.chain(*lattice_surface_index.tolist())))
        lattice_coord = lattice_coord[lattice_surface_index,:]
        lattice_index = lattice_index[lattice_surface_index,:]


        return lattice_coord, lattice_index , lattice_surface_index

def OOC_Halo_Voxelise(pdbfiledir,  DIR_OutputHaloTupleFolder = "",
                                   #n_HaloNearestNeighbor = 50,
                                   UpdateExist = False,
                                   WritingXyz = False, SavingHalo = False,
                                   DIR_JustOnePdb = None, 
                                   HaloLowerBound = 1.0, HaloUpperBound = 4.0, LatticeSpacing = 1.0):


    halo_bound = (HaloLowerBound, HaloUpperBound)
    pdbid = pdbfiledir.split("/")[-1].split(".")[0]
    if os.path.exists(DIR_OutputHaloTupleFolder + '/%s.halotup' %(pdbid)) & ~(UpdateExist):
        return
    # =====================
    # Define Protein Lattice
    # =====================
    
    PART1_DefineProteinLattice = True
    if PART1_DefineProteinLattice:
        try:
            # NOTE This will read the pdb into memory
            #
            #subprocess.call("cp %s %s/%s.pdb" %(pdbfiledir, tmpdirname, pdbid) ,shell = True)

            ppdb = PandasPdb()
            CurrentPdbStructure = ppdb.read_pdb("%s" %(pdbfiledir))
            protein_df = CurrentPdbStructure.df['ATOM']
            protein_coord =  protein_df[["x_coord", "y_coord", "z_coord"]].values
            protein_tree = spatial.cKDTree(protein_coord)

            # Define Dimension of Grid box. NOTE This is a voxelised box just enough to hold the protein
            maxx, minx = protein_df["x_coord"].max(), protein_df["x_coord"].min()
            maxy, miny = protein_df["y_coord"].max(), protein_df["y_coord"].min()
            maxz, minz = protein_df["z_coord"].max(), protein_df["z_coord"].min()
            
            maxx += HaloUpperBound 
            maxy += HaloUpperBound 
            maxz += HaloUpperBound 

            minx -= HaloUpperBound 
            miny -= HaloUpperBound 
            minz -= HaloUpperBound 

            # NOTE The notation in https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html
            x_numintervals = int(np.abs(maxx - minx)/LatticeSpacing)+1
            y_numintervals = int(np.abs(maxy - miny)/LatticeSpacing)+1
            z_numintervals = int(np.abs(maxz - minz)/LatticeSpacing)+1
            lattice_coord = np.mgrid[   minx:maxx:(x_numintervals)*1j, 
                                        miny:maxy:(y_numintervals)*1j, 
                                        minz:maxz:(z_numintervals)*1j
                                        ]
            
            # NOTE This index refer to the full dense tensor. 
            #      We will make a sparse matrix. The ingredients are the indices and voxel size of the lattice
            lattice_index = np.mgrid[   0:lattice_coord.shape[1], 
                                        0:lattice_coord.shape[2], 
                                        0:lattice_coord.shape[3]
                                        ]
            lattice_shape = (int(lattice_coord.shape[1]),
                            int(lattice_coord.shape[2]),
                            int(lattice_coord.shape[3]))


            #print(x_numintervals, y_numintervals, z_numintervals, minx, miny, minz, maxx, maxy, maxz)
            #print(lattice_coord[2,3,5,7])

            # NOTE Reshape as (n,3)
            lattice_coord = np.matrix(lattice_coord.reshape(3, -1).T)
            lattice_index = np.matrix(lattice_index.reshape(3, -1).T)
            lattice_minmax = (minx, miny, minz, maxx, maxy, maxz)
        except:
            # NOTE 4wsb is a known case.
            print("ABORTED. %s failed to form lattice. Check why? Float overflow?" %(pdbid))
            return

    # ==================================
    # Trimming of Lattice into Halo 
    # ==================================
    PART2_TrimmingIntoHalo = True
    if PART2_TrimmingIntoHalo:
        #lattice_coord, lattice_index, _ = OOC_Lattice_TrimIntersect(lattice_coord, lattice_index, surface_coord, 
        #                                                    intersect_radius = HaloUpperBound)
        _, _, inner_halo_index = OOC_Lattice_TrimIntersect(lattice_coord, lattice_index, protein_coord, 
                                                            intersect_radius = HaloLowerBound)
        _, _, outer_halo_index = OOC_Lattice_TrimIntersect(lattice_coord, lattice_index, protein_coord, 
                                                            intersect_radius = HaloUpperBound)
        midline_halo_index = sorted(set(outer_halo_index) - set(inner_halo_index))

        halo_coord = lattice_coord[midline_halo_index,:]
        halo_index = lattice_index[midline_halo_index,:]

    



    halo_tuple = (halo_index, lattice_shape, lattice_minmax, halo_bound)
    # ==========================
    # Saving
    # ==========================
    if WritingXyz:
        WriteXyz(halo_coord.tolist(),"H",DIR_OutputHaloTupleFolder + "/%s.haloxyz" %(pdbid))

    if SavingHalo:
        with open(DIR_OutputHaloTupleFolder + '/%s.halotup' %(pdbid), 'wb') as fn:
            pickle.dump(halo_tuple,fn, protocol=4)                                  
        

    if DIR_JustOnePdb is not None:
        return halo_tuple
    else:
        return

class Halo:
    def __init__(self, HaloLowerBound = 2.0, HaloUpperBound = 4.0, LatticeSpacing = 1.0 , 
                  DIR_OutputHaloTupleFolder = "../Database-PDB/halo",
                  DIR_InputPdbFolder = "../Database-PDB/apo", 
                  DIR_OutputTypifyFolder = "../Database-PDB/typi",
                  n_MultiprocessingWorkers = 10,
                  #n_HaloNearestNeighbor = 50,
                  CallMkdirList = True,
                 ):
        self.HaloLowerBound = HaloLowerBound
        self.HaloUpperBound = HaloUpperBound
        self.LatticeSpacing = LatticeSpacing
        self.DIR_OutputHaloTupleFolder = DIR_OutputHaloTupleFolder
        self.DIR_InputPdbFolder = DIR_InputPdbFolder
        self.DIR_OutputTypifyFolder = DIR_OutputTypifyFolder
        self.n_MultiprocessingWorkers = n_MultiprocessingWorkers
        #self.n_HaloNearestNeighbor = n_HaloNearestNeighbor
        if CallMkdirList:
            MkdirList([self.DIR_OutputHaloTupleFolder, self.DIR_OutputTypifyFolder])

    # NOTE This will not take long I do it sequentially
    def Voxelise(self,
                        PlottingHalo = False, WritingXyz = False,
                        DIR_JustOnePdb = None, SavingHalo = True, UpdateExist = False):

        # ===================
        # I/O
        # ===================

        HaloLowerBound = self.HaloLowerBound
        HaloUpperBound = self.HaloUpperBound
        LatticeSpacing = self.LatticeSpacing
        DIR_OutputHaloTupleFolder = self.DIR_OutputHaloTupleFolder
        DIR_InputPdbFolder = self.DIR_InputPdbFolder


        # NOTE This determine whether just one pdb is inputted. Good for debugging
        if DIR_JustOnePdb is None:
            pdbfilelist_ = sorted(glob.glob(DIR_InputPdbFolder + "/*.pdb"))
            PlottingHalo = False
            DIR_WriteXyz = ""
            #UpdateExist = True
            if not UpdateExist:
                pdbfilelist  =[]
                for i in pdbfilelist_:
                    pdbid = i.split("/")[-1].split('.')[0]
                    #print(pdbid)
                    if not (os.path.exists(DIR_OutputHaloTupleFolder + '/%s.halotup' %(pdbid))):
                        #print(DIR_OutputHaloTupleFolder + '/%s.halotup' %(pdbid))
                        pdbfilelist.append(i)
                print(len(pdbfilelist_), len(pdbfilelist), )
            else:
                pdbfilelist = pdbfilelist_
        else:
            pdbfilelist = [DIR_JustOnePdb]

        # ========================
        # Body
        # ========================
        if DIR_JustOnePdb is None:
            VoxelisePartial = partial(OOC_Halo_Voxelise, 
                                    DIR_OutputHaloTupleFolder = DIR_OutputHaloTupleFolder,
                                    UpdateExist = UpdateExist,
                                    WritingXyz = WritingXyz, SavingHalo = SavingHalo,
                                    DIR_JustOnePdb = DIR_JustOnePdb,
                                    HaloLowerBound = HaloLowerBound, 
                                    HaloUpperBound = HaloUpperBound, 
                                    #n_HaloNearestNeighbor = self.n_HaloNearestNeighbor,
                                    LatticeSpacing = LatticeSpacing)
            pool = multiprocessing.Pool(self.n_MultiprocessingWorkers)
            pool.map(VoxelisePartial, pdbfilelist)
        else:
            halo_tuple = OOC_Halo_Voxelise(pdbfilelist[0],
                                    DIR_OutputHaloTupleFolder = DIR_OutputHaloTupleFolder,
                                    UpdateExist = UpdateExist,
                                    WritingXyz = WritingXyz, SavingHalo = SavingHalo,
                                    DIR_JustOnePdb = DIR_JustOnePdb,
                                    HaloLowerBound = HaloLowerBound, 
                                    HaloUpperBound = HaloUpperBound, 
                                    #n_HaloNearestNeighbor = self.n_HaloNearestNeighbor,
                                    LatticeSpacing = LatticeSpacing
                                )
        
        
        # ===========================
        # Visualise
        # ===========================
        if DIR_JustOnePdb is not None:
            if PlottingHalo:
                pdbfiledir = DIR_JustOnePdb
                ppdb = PandasPdb()
                CurrentPdbStructure = ppdb.read_pdb("%s" %(pdbfiledir))
                protein_df = CurrentPdbStructure.df['ATOM']
                protein_coord =  protein_df[["x_coord", "y_coord", "z_coord"]].values

                self.VisualiseHaloValues(halo_tuple, 
                                Lattice_Bleeding = HaloUpperBound, 
                                protein_coord = protein_coord)

            # NOTE The three output allows sparse to dense representation
            return halo_tuple
        else:
            return None



    # =========================
    # Some useful self function
    # =========================
    
    # NOTE this will be useful when we need to visualise features; just put halo_values 
    def VisualiseHaloValues(self, halo_tuple, 
                    halo_values = None, protein_coord = None, Lattice_Bleeding = 5.0,
                    halo_values_is_binaryclassmatrix = False, 
                    value_palette = "Gray_r", protein_palette = "Viridis"):

        # NOTE https://plotly.com/python/visualizing-mri-volume-slices/
        import plotly.io as pio
        pio.renderers.default = "notebook_connected"

        halo_index, lattice_shape, lattice_minmax, halo_bound = halo_tuple


        if halo_values is None:
            halo_values = np.ones(halo_index.shape[0])

        if len(halo_values.shape) > 1:
            if halo_values_is_binaryclassmatrix:
                for i in range(halo_values.shape[-1]):
                    halo_values[:,i] = halo_values[:,i] * (i+1)
                cmin = 0.0
                cmax = halo_values.shape[-1]
                halo_values = halo_values.sum(axis = 1)
                print(halo_values.max(), halo_values.min())
                print("I would suggest rainbow or haline_r as value_palette")
                #palette = ""
            else:
                # TODO We should ask for a level set drawing then
                halo_values = np.ones(halo_index.shape[0])
                cmin=0.0
                cmax=1.0
                pass
        else:
                cmin=0.0
                cmax=1.0
        # NOTE This makes the dense volume tensor; we dont care about the relative voxel size as long as it is 1 Ang
        #      TODO Handle this 
        lattice_volume = np.zeros(( lattice_shape[0], 
                                    lattice_shape[1], 
                                    lattice_shape[2]), dtype=float)
        print(lattice_volume.shape)
        print(halo_values.shape)
        print(halo_values)
        #halo_values = np.flatten(halo_values)

        lattice_volume[halo_index.T[0], halo_index.T[1], halo_index.T[2]] += halo_values.T




        n_z_frames, n_x_frames, n_y_frames = lattice_volume.shape
        import plotly.graph_objects as go

        frames=[go.Frame(
                    data=go.Surface(
                            z = ((n_z_frames -1) /10 - k * 0.1) * np.ones((n_x_frames, n_y_frames)),
                            surfacecolor = np.flipud(lattice_volume[(n_z_frames -1) - k]),
                            cmin=cmin, cmax=cmax,
                            )
                    ,name=str(k)
                    ) 
                    for k in range(n_z_frames)
            ]
        fig = go.Figure(frames = frames)


        # Add data to be displayed before animation starts
        fig.add_trace(go.Surface(
                        z=(n_z_frames -1) /10 * np.ones((n_x_frames, n_y_frames)),
                        surfacecolor=np.flipud(lattice_volume[(n_z_frames -1) ]),
                        colorscale=value_palette,
                        opacity = 0.8,
                        cmin=cmin, cmax=cmax,
                        #colorbar=dict(thickness=20, ticklen=4)
                        )
                        )

        if protein_coord is not None:

            protein_coord = protein_coord.T
            # NOTE unfortunately plotly slider reserve z for scrolling. This requires us to permute the tensor
            #      These handles the volume slice correspondence
            protein_coord = np.flipud(protein_coord)
            protein_coord[2,:] = protein_coord[2,:]/10.0
            protein_coord = np.matmul( np.array([[0,-1, 0],
                                                [1, 0, 0],
                                                [0, 0, 1]]) , protein_coord )

            protein_coord = protein_coord - np.expand_dims(np.min(protein_coord, axis =1),1)
            protein_coord = protein_coord[[1,0,2],:]
            protein_coord[0,:] = protein_coord[0,:] + Lattice_Bleeding
            protein_coord[1,:] = protein_coord[1,:] + Lattice_Bleeding
            protein_coord[2,:] = protein_coord[2,:] + Lattice_Bleeding / 10.0

            color_z = (protein_coord[2] - protein_coord[2].min())
            color_z = color_z/color_z.max()
            
            fig.add_trace(go.Scatter3d(
                x = protein_coord[0], y = protein_coord[1], z = protein_coord[2]
                ,mode = 'markers', marker = dict(
                size = 5,
                color = color_z, # set color to an array/list of desired values
                colorscale = protein_palette,
                opacity = 0.05
                )
                                        )
                        )


        
        frame_args_begin = {
                    "frame": {"duration": 0},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": 0, "easing": "linear"},
                }

        frame_args_end = {
                    "frame": {"duration": 5},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": 5, "easing": "linear"},
                }

        sliders = [
                    {
                        #"pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], frame_args_begin],
                                "label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig.frames)
                        ],
                    }
                ]

        # Layout

        fig.update_layout(
                title='Slices of Voxel Values on Halo',
                width=600,
                height=600,
                scene=dict(
                            zaxis=dict(range=[-0.1, (n_z_frames/10)], autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            ),
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args_end],
                                "label": "&#9654;", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args_begin],
                                "label": "&#9724;", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
        )
        fig.show()


        
        return  


    def RetrieveHaloCoords( self, halo_tuple, reshape = True):

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

    def RetrieveHaloNearestNeighbor(self, halo_tuple, WithoutSelf = False, n_HaloNearestNeighbor = 50):

        # NOTE I decide not to store this because it drastically increase the storage.
        #      cKD tree is fast.
        halo_index = halo_tuple[0]
        halo_tree = spatial.cKDTree(halo_index)
        _ , halo_nearest = halo_tree.query(halo_index, k = n_HaloNearestNeighbor+1, 
                                        eps=0, p=2, 
                                        distance_upper_bound=np.inf, workers=1)
        if WithoutSelf:
            halo_nearest = halo_nearest[:,1:]
        return halo_nearest

        
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