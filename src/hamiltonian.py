import json
import numpy as np 
import math
import sympy as sp
from crystal import *

ORBITAL_MAP = {
    "s": 0,
    "x": 1, "y": 2, "z": 3,
    "xy": 4, "yz": 5, "xz": 6,
    "d1": 7,  # d_x2-y2
    "d2": 8   # d_3z2-r2
}

ORBITAL_NUM = {
    0:"s",
    1: "x",2:"y",3:"z",
    4:"xy",5:"yz",6:"xz",7:"d1",8:"d2"}


def load(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_param(data, basis_type, category, neighbor, key):
    base = data.get(basis_type, {})
    if category not in base:
        raise KeyError(f"Category '{category}' not found.")
        
    section = base[category]
    
    if neighbor is None:
        for orbital, value in section:
            if orbital == key:
                return value
        raise ValueError(f"Orbital '{key}' not found.")
    
    if neighbor not in section:
        raise KeyError(f"Neighbor '{neighbor}' not found.")
        
    for pair, value in section[neighbor]:
        if pair == key:
            return value
            
    raise ValueError(f"Parameter '{key}' not found.")


    



class SlaterKoster:
    def __init__(self):
        self.l, self.m, self.n = sp.symbols('l m n')
        self.sss = sp.Symbol('sss')
        self.sps = sp.Symbol('sps')
        self.pps = sp.Symbol('pps')
        self.ppp = sp.Symbol('ppp')
        self.sds = sp.Symbol('sds')
        self.pds = sp.Symbol('pds')
        self.pdp = sp.Symbol('pdp')
        self.dds = sp.Symbol('dds')
        self.ddp = sp.Symbol('ddp')
        self.ddd = sp.Symbol('ddd')
        self._build_table()

    def _build_table(self):
        l, m, n = self.l, self.m, self.n
        sss, sps = self.sss, self.sps
        pps, ppp = self.pps, self.ppp
        sds = self.sds
        pds, pdp = self.pds, self.pdp
        dds, ddp, ddd = self.dds, self.ddp, self.ddd

        self.table = {
            ("s", "s"): sss,
            ("s", "x"): l * sps,
            ("s", "y"): m * sps,
            ("s", "z"): n * sps,
            ("x", "x"): l**2 * pps + (1 - l**2) * ppp,
            ("y", "y"): m**2 * pps + (1 - m**2) * ppp,
            ("z", "z"): n**2 * pps + (1 - n**2) * ppp,
            ("x", "y"): l * m * (pps - ppp),
            ("x", "z"): l * n * (pps - ppp),
            ("y", "z"): m * n * (pps - ppp),
            ("s", "xy"): sp.sqrt(3) * l * m * sds,
            ("s", "yz"): sp.sqrt(3) * m * n * sds,
            ("s", "zx"): sp.sqrt(3) * l * n * sds,
            ("s", "x2-y2"): (sp.sqrt(3) / 2) * (l**2 - m**2) * sds,
            ("s", "3z2-r2"): (n**2 - 0.5 * (l**2 + m**2)) * sds,
            ("x", "xy"): sp.sqrt(3) * (l**2) * m * pds + m * (1 - 2 * l**2) * pdp,
            ("x", "yz"): sp.sqrt(3) * l * m * n * pds - 2 * l * m * n * pdp,
            ("x", "zx"): sp.sqrt(3) * (l**2) * n * pds + n * (1 - 2 * l**2) * pdp,
            ("x", "x2-y2"): (sp.sqrt(3) / 2) * l * (l**2 - m**2) * pds + l * (1 - l**2 + m**2) * pdp,
            ("x", "3z2-r2"): l * (n**2 - 0.5 * (l**2 + m**2)) * pds - sp.sqrt(3) * l * n**2 * pdp,
            ("y", "xy"): sp.sqrt(3) * l * (m**2) * pds + l * (1 - 2 * m**2) * pdp,
            ("y", "yz"): sp.sqrt(3) * (m**2) * n * pds + n * (1 - 2 * m**2) * pdp,
            ("y", "zx"): sp.sqrt(3) * l * m * n * pds - 2 * l * m * n * pdp,
            ("y", "x2-y2"): (sp.sqrt(3) / 2) * m * (l**2 - m**2) * pds - m * (1 + l**2 - m**2) * pdp,
            ("y", "3z2-r2"): m * (n**2 - 0.5 * (l**2 + m**2)) * pds - sp.sqrt(3) * m * n**2 * pdp,
            ("z", "xy"): sp.sqrt(3) * l * m * n * pds - 2 * l * m * n * pdp,
            ("z", "yz"): sp.sqrt(3) * m * (n**2) * pds + m * (1 - 2 * n**2) * pdp,
            ("z", "zx"): sp.sqrt(3) * l * (n**2) * pds + l * (1 - 2 * n**2) * pdp,
            ("z", "x2-y2"): (sp.sqrt(3) / 2) * n * (l**2 - m**2) * pds - n * (l**2 - m**2) * pdp,
            ("z", "3z2-r2"): n * (n**2 - 0.5 * (l**2 + m**2)) * pds + sp.sqrt(3) * n * (l**2 + m**2) * pdp,
            ("xy", "xy"): 3 * (l**2) * (m**2) * dds + (l**2 + m**2 - 4 * l**2 * m**2) * ddp + (n**2 + l**2 * m**2) * ddd,
            ("yz", "yz"): 3 * (m**2) * (n**2) * dds + (m**2 + n**2 - 4 * m**2 * n**2) * ddp + (l**2 + m**2 * n**2) * ddd,
            ("zx", "zx"): 3 * (l**2) * (n**2) * dds + (l**2 + n**2 - 4 * l**2 * n**2) * ddp + (m**2 + l**2 * n**2) * ddd,
            ("xy", "yz"): 3 * l * (m**2) * n * dds + l * n * (1 - 4 * m**2) * ddp + l * n * (m**2 - 1) * ddd,
            ("xy", "zx"): 3 * (l**2) * m * n * dds + m * n * (1 - 4 * l**2) * ddp + m * n * (l**2 - 1) * ddd,
            ("yz", "zx"): 3 * l * m * (n**2) * dds + l * m * (1 - 4 * n**2) * ddp + l * m * (n**2 - 1) * ddd,
            ("xy", "x2-y2"): 1.5 * l * m * (l**2 - m**2) * dds + 2 * l * m * (m**2 - l**2) * ddp + 0.5 * l * m * (l**2 - m**2) * ddd,
            ("yz", "x2-y2"): 1.5 * m * n * (l**2 - m**2) * dds - m * n * (1 + 2 * (l**2 - m**2)) * ddp + m * n * (1 + 0.5 * (l**2 - m**2)) * ddd,
            ("zx", "x2-y2"): 1.5 * l * n * (l**2 - m**2) * dds + l * n * (1 - 2 * (l**2 - m**2)) * ddp - l * n * (1 - 0.5 * (l**2 - m**2)) * ddd,
            ("xy", "3z2-r2"): sp.sqrt(3) * l * m * (n**2 - 0.5 * (l**2 + m**2)) * dds - 2 * sp.sqrt(3) * l * m * n**2 * ddp + 0.5 * sp.sqrt(3) * l * m * (1 + n**2) * ddd,
            ("yz", "3z2-r2"): sp.sqrt(3) * m * n * (n**2 - 0.5 * (l**2 + m**2)) * dds + sp.sqrt(3) * m * n * (l**2 + m**2 - n**2) * ddp - 0.5 * sp.sqrt(3) * m * n * (l**2 + m**2) * ddd,
            ("zx", "3z2-r2"): sp.sqrt(3) * l * n * (n**2 - 0.5 * (l**2 + m**2)) * dds + sp.sqrt(3) * l * n * (l**2 + m**2 - n**2) * ddp - 0.5 * sp.sqrt(3) * l * n * (l**2 + m**2) * ddd,
            ("x2-y2", "x2-y2"): 0.75 * ((l**2 - m**2)**2) * dds + (l**2 + m**2 - (l**2 - m**2)**2) * ddp + (n**2 + 0.25 * ((l**2 - m**2)**2)) * ddd,
            ("x2-y2", "3z2-r2"): (sp.sqrt(3) / 2) * (l**2 - m**2) * (n**2 - 0.5 * (l**2 + m**2)) * dds + sp.sqrt(3) * n**2 * (m**2 - l**2) * ddp + (sp.sqrt(3) / 4) * (1 + n**2) * (l**2 - m**2) * ddd,
            ("3z2-r2", "3z2-r2"): ((n**2 - 0.5 * (l**2 + m**2))**2) * dds + 3 * n**2 * (l**2 + m**2) * ddp + 0.75 * ((l**2 + m**2)**2) * ddd
        }

    def get_equation(self, orb1, orb2, l_val, m_val, n_val):
        pair = (orb1, orb2)
        if pair not in self.table:
            pair = (orb2, orb1)
            
        if pair not in self.table:
            raise KeyError(f"Pair ({orb1}, {orb2}) not found.")

        expr = self.table[pair]
        return expr.subs({self.l: l_val, self.m: m_val, self.n: n_val})

    def get_all_equations(self, l_val, m_val, n_val):
        result = {}
        for (orb1, orb2), expr in self.table.items():
            result[(orb1, orb2)] = expr.subs({self.l: l_val, self.m: m_val, self.n: n_val})
        return result
        

class Hamiltonian(SlaterKoster):
    def __init__(self):
        self.sk = SlaterKoster()

    def on_site():
        return 0


    def first_NN():
        return 0





if __name__ == "__main__":
    data = load("data/SK_2-CP_Fe_bcc_up.json")

    sk = SlaterKoster()
    
    l_val, m_val , n_val = direction_cosines(pos=[2,0,0],  sym = False)

    tabla = sk.get_all_equations(l_val, m_val, n_val)

    # To check that we are obtaining the same results that Papaconstatinopolous section 1.4
    '''
    for par, eq in tabla.items():
        eq_simplificada = sp.simplify(eq)
        #print(f"{par[0]}-{par[1]}: {eq}")
        print(f"{par[0]}-{par[1]}")
        sp.pprint(eq_simplificada)'''

    a = 5.40570 # a.u.
    a = au_to_angstrom(a)
    crystal = Crystal(structure="bcc", a=a, NN = 3)
    crystal.build_NN()

    print("pos")
    print(crystal.pos)

    print( get_param(data, "orthogonal","onsite",None,"p") )