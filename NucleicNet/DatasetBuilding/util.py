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
from argparse import ArgumentParser
import sys
import re
import pandas as pd
from io import StringIO
import urllib.request
import glob
import gzip
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd

import torch

# ====================
# I/O
# =====================

# Mkdir if not exist
def MkdirList(folderlist):
    
    for i in folderlist:
        if not os.path.exists('%s' %(i)):
            os.mkdir('%s' %(i))

def WriteXyz(listofarray,label,fn):
    XYZTrial = []
    """
    if listofarray:
        Points=sorted(listofarray)
        for point in Points:
            XYZTrial.append('%s       %.5f        %.5f        %.5f\n' %(label, point[0], point[1], point[2]))
        with open("%s" %(fn),'w+') as f:
            for point in XYZTrial:
                f.write(point)
    """
    for point in listofarray:
        XYZTrial.append('%s       %.5f        %.5f        %.5f\n' %(label, point[0], point[1], point[2]))
    with open("%s" %(fn),'w+') as f:
        for point in XYZTrial:
            f.write(point)
    del XYZTrial

def OOC_ChunkList(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def ChunkList(lst,n):
    return list(OOC_ChunkList(lst, n))


def KillInactiveKernels(cpu_threshold):



    from os import getpid, kill
    from time import sleep
    import signal
    import psutil

    sleep(1.0)

    inactive_kernels = []
    pids = psutil.pids()
    my_pid = getpid()

    for pid in pids:
        if pid == my_pid:
            continue
        try:
            p = psutil.Process(pid)
            cmd = p.cmdline()
            for arg in cmd:
                if arg.count('ipykernel'):
                    cpu = p.cpu_percent(interval=0.1)
                    if cpu < cpu_threshold:
                        tempappend = [cpu, pid]
                        tempappend.extend(cmd)
                        inactive_kernels.append(tempappend)
                        #print(pid, cpu)
        except:
            continue

    # NOTE May consider to remove this block altogether
    try:
        inactive_kernels.sort(key=lambda x: x[1])
    except:
        return [] 
    #print(inactive_kernels)


    # kill one by one
    for i in inactive_kernels:
        if "--stdin=9013" in i:
            try:
                kill(i[1], signal.SIGINT)
                #print(cmd)
                sleep(0.5)
            except:
                pass

    return inactive_kernels

# ==================
# torch
# ==================
def TorchEmptyCache():

    torch.cuda.empty_cache()    
    #torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.memory_allocated(0)
    torch.cuda.max_memory_allocated(0)



# ===================
# Basic Visualisation
# =====================

def DrawNxGraphPlotly(test_Gg, pos_, 
    VisualiseEdge = True, EdgeWeightGtThreshold = 0.0, EdgeWidthVisualisationWeight = 1.0):
    
    print("Finished layout")
    import tqdm
    import plotly
    import networkx as nx
    import plotly.graph_objects as go
    import pandas as pd
    import gc


    # Copy the graph
    test_Gg_ = test_Gg
    all_edges = test_Gg_.edges()


    # Create figure
    PART0_MakeFigure = True
    if PART0_MakeFigure:
        layout = go.Layout(
            xaxis =  {'showgrid': False, 'zeroline': False}, 
            yaxis = {'showgrid': False, 'zeroline': False}, 
            )
        fig = go.Figure(layout = layout)



    # ==========================
    # Edge
    # ===========================
    if VisualiseEdge:
        PART1_UpdatingEdgeFigure = True
    else:
        PART1_UpdatingEdgeFigure = False
    if PART1_UpdatingEdgeFigure:
        print("Updating Edge Figure")    
        for edge in tqdm.tqdm(test_Gg_.edges()):
            
            if all_edges[edge]['weight'] > EdgeWeightGtThreshold:
                char_1 = edge[0]
                char_2 = edge[1]
                x0, y0 = pos_[char_1]
                x1, y1 = pos_[char_2]

                # Update figure
                fig.add_trace( go.Scatter(
                                x         = [x0, x1, None],
                                y         = [y0, y1, None],
                                opacity   = .5,
                                line      = dict(width = EdgeWidthVisualisationWeight*all_edges[edge]['weight'],
                                            color = 'grey'),
                                mode      = 'lines')
                )


    # ==========================
    # Node
    # ==========================
    print("Updating Node Figure")
    PART2_UpdatingNodeFigure = True
    if PART2_UpdatingNodeFigure:
        # Make a node trace
        node_text = []
        node_x = []
        node_y = []
        node_color = []
        for k,v in tqdm.tqdm(pos_.items()):
            if k not in test_Gg_.nodes():
                continue       

            node_text.append(str(k))
            node_x.append(v[0])
            node_y.append(v[1])
            node_color.append(list(test_Gg_.degree([k]))[0][1])
        node_color = np.log10(np.array(node_color)+1.0).tolist()
        node_trace = go.Scatter(x         = node_x,
                                y         = node_y,
                                hovertext = node_text,
                                #hovermode ='y',
                                #hoverdistance=5,                                
                                mode      = 'markers',
                                hovertemplate = "%{hovertext}<extra></extra>", 
                                marker_colorscale = plotly.colors.sequential.deep,
                                marker    = dict(
                                                color = node_color,
                                                line  = None,
                                                colorbar = dict(thickness=20)
                                                )
                                )

        node_df = pd.DataFrame(zip(list(range(len(node_x))), node_x, node_y, node_color, node_text),
                                columns = ["Index", "X", "Y", "Color", "Text"])

        fig.add_trace(node_trace)
        del node_x, node_y, node_color, node_text
        gc.collect()


    # ===============================
    # Node Call back
    # ================================

    # ==============================
    # Display
    # ==============================
    print("Loading Figure")
    # Tidy up
    fig.update_layout(
                #hovermode="closest",

                autosize=False,
                width=1000,
                height=1000,
                showlegend = False
                )

    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    fig.show()
    #fig.close()




# ===================
# Basic sequence tools
# ===================


def SimpleEditDistance(l,    r):

    D = np.zeros((len(l)+1, len(r)+1))
    # Set dummy init
    D[1:,0 ] = range(1,len(l)+1)
    D[0, 1:] = range(1,len(r)+1)
    
    for i in range(1, len(l)+1):
        for j in range(1, len(r)+1):
            if l[i-1] != r[j-1]:
                mismatch = 1
            else:
                mismatch = 0
            # update DP
            D[i,j] = min([
                D[i-1,j-1] + mismatch ,    
                D[i-1,j]+1,   
                D[i,j-1]+1
                ])
    return D[-1,-1]



# ============================
# Configure Matplotlib 
# ============================
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties


#globscale = 1.35
LETTERS = { "U" : TextPath((-0.365, 0), "U", size=1, prop=FontProperties(family="sans-serif", weight="bold") ),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=FontProperties(family="sans-serif", weight="bold") ),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=FontProperties(family="sans-serif", weight="bold") ),
            "C" : TextPath((-0.325, 0), "C", size=1, prop=FontProperties(family="sans-serif", weight="bold") ) }
# RNAC color scheme
COLOR_SCHEME = {'G': '#FFAF00', 
                'A': '#06C806', 
                'C': '#0003C6', 
                'U': '#C90101'}

# Our NucleicNet Color scheme
#COLOR_SCHEME = {'G': '#AEF3EB', 
#                'A': '#5B9AE8', 
#                'C': '#E30F38', 
#                'U': '#F1BBC1'}

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*1.35, yscale*1.35) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p