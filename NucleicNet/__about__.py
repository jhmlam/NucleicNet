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
import time

_this_year = time.strftime("%Y")
__version__ = "1.1"
__authors__ = "Jordy Homing Lam"
__corresponding_author__ = "Xin Gao, Xuhui Huang, Lizhe Zhu"
# The current address of author
__license__ = "AGPL-3.0 license"
__copyright__ = f"Copyright (c) 2019-{_this_year}, {__authors__}."
__homepage__ = "https://github.com/"
__docs_url__ = "readthedocs.io"
__citation__ = "Lam, J.H., Li, Y., Zhu, L. et al. A deep learning framework to predict binding preference of RNA constituents on protein surface. Nat Commun 10, 4941 (2019). https://doi.org/10.1038/s41467-019-12920-0"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "NucleicNet"
)
__long_docs__ = """
Protein-RNA interaction is ubiquitous in cells and serves as the main mechanism for post-transcriptional regulation. Base-dominant interaction and backbone-dominant interaction categorize the two main modes of the way RNA interacting with proteins. Despite the advances in experimental technologies and computational methods to capture protein-RNA interactions, estimating binding preference of RNA backbone constituents and different bases on any location of a given protein surface is beyond the capacity of existing techniques. Here we show that these attributes can be predicted from the local physicochemical characteristics of the protein structure surface, by leveraging the power of deep learning.
"""

__all__ = ["__authors__", "__corresponding_author__", 
 "__copyright__", "__docs__", "__homepage__", "__license__", "__version__"]
