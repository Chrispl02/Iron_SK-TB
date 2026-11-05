import numpy as np 
import math
import sympy as sp
from crystal import *

class TetrahedronDOS:
    def __init__(self, energies, tetra, bz_weight=None):
        # energies: (Nk, nbands)
        # tetra: (Nt, 4)
        self.eps = np.sort(energies[tetra], axis=1) 
        self.w = 1.0 / len(tetra)

    def dos(self, E): 
        pass
    def integrated_dos(self, E): 
        pass