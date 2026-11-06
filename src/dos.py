#DoS and integred DoS with tetrahedral method

import numpy as np 
import math
import sympy as sp
from crystal import *

class TetrahedronDOS:
    

    def __init__(self, energies, tetra, tetra_w):
        energies = np.asarray(energies, float)
        if energies.ndim == 1:
            energies = energies[:, None]
        eps = np.sort(energies[np.asarray(tetra)], axis=1)      # (Nt, 4, nb)
        w = np.asarray(tetra_w, float)
        self.w = w / w.sum()
        self.nb = eps.shape[2]
        self.e1, self.e2, self.e3, self.e4 = (eps[:, i, :] for i in range(4))

        d21 = self.e2 - self.e1
        d31 = self.e3 - self.e1
        d41 = self.e4 - self.e1
        d32 = self.e3 - self.e2
        d42 = self.e4 - self.e2
        d43 = self.e4 - self.e3
        inv = lambda x: 1.0 / np.where(x > 0, x, 1.0)   # solo se usa donde x > 0
        self.d21 = d21
        self.c1 = inv(d21 * d31 * d41)
        self.c2 = inv(d31 * d41)
        self.c2b = (d31 + d42) * inv(d32 * d42)
        self.c3 = inv(d41 * d42 * d43)

    # --
    def _evaluate(self, E, integrated, chunk=128):
        E = np.atleast_1d(np.asarray(E, float))
        out = np.zeros((len(E), self.nb))
        e1, e2, e3, e4 = self.e1, self.e2, self.e3, self.e4
        # Chunk rutine to make it faster without spending to much memory
        for s in range(0, len(E), chunk):
            x = E[s:s + chunk, None, None]                       # (ne, 1, 1)
            m1 = (x >= e1) & (x < e2)
            m2 = (x >= e2) & (x < e3)
            m3 = (x >= e3) & (x < e4)
            y = x - e2
            if integrated:
                t1 = (x - e1) ** 3 * self.c1
                t2 = self.c2 * (self.d21**2 + 3 * self.d21 * y + 3 * y**2
                                - self.c2b * y**3)
                t3 = 1.0 - (e4 - x) ** 3 * self.c3
                val = np.where(m1, t1, 0) + np.where(m2, t2, 0) \
                    + np.where(m3, t3, 0) + (x >= e4)
            else:
                t1 = 3 * (x - e1) ** 2 * self.c1
                t2 = self.c2 * (3 * self.d21 + 6 * y - 3 * self.c2b * y**2)
                t3 = 3 * (e4 - x) ** 2 * self.c3
                val = np.where(m1, t1, 0) + np.where(m2, t2, 0) + np.where(m3, t3, 0)
            out[s:s + chunk] = np.einsum("t,etb->eb", self.w, val)
        return out

    def dos(self, E, per_band=False):
        g = self._evaluate(E, integrated=False)
        return g if per_band else g.sum(axis=1)

    def integrated_dos(self, E, per_band=False):
        n = self._evaluate(E, integrated=True)
        return n if per_band else n.sum(axis=1)

    def energy_grid(self, npts=1000, pad=0.5):
        return np.linspace(self.e1.min() - pad, self.e4.max() + pad, npts)