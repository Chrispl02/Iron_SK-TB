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

global ORBITAL
ORBITAL = {'s': 0, 'x': 1, 'y': 2, 'z': 3, 
            'xy': 4, 'yz': 5, 'zx': 6, 'x2-y2': 7, '3z2-r2': 8}

def ry_to_eV(x):
    return x * 13.6057

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
        
# No spin option is not proved
class Hamiltonian(SlaterKoster):
    def __init__(self,crystal_pos, spin = False):
        super().__init__()
        self.sk = SlaterKoster()
        self.crystal = crystal_pos
        self.spin = spin
        self.basis = "orthogonal"
        # Assume H 9x9 (structure until d orbital)
        if spin == True:
            self.H_on_site = np.zeros([2,9,9])
            self.H_R_up = {}
            self.H_R_dn = {}
        else: # spin False
            self.H_on_site = np.zeros([9,9])
            self.H_R = {}

    def load_param(self, data):
        if self.spin == True:
            self.param_up = load(data[0])
            self.param_dn = load(data[1])
        else:
            self.param = load(data)


    def on_site(self):
        category = "onsite"
        
        if self.spin:
            for s_idx, p_data in enumerate([self.param_up, self.param_dn]):
                e_s = dict(p_data[self.basis][category])['s']
                e_p = dict(p_data[self.basis][category])['p']
                e_d1 = dict(p_data[self.basis][category])['d1']
                e_d2 = dict(p_data[self.basis][category])['d2']

                self.H_on_site[s_idx, 0, 0] = e_s
                for i in range(1, 4): self.H_on_site[s_idx, i, i] = e_p
                for i in range(4, 7): self.H_on_site[s_idx, i, i] = e_d1
                for i in range(7, 9): self.H_on_site[s_idx, i, i] = e_d2
            
        else:

            e_s = dict(self.param_up[self.basis][category])['s']
            e_p = dict(self.param_up[self.basis][category])['p']
            e_d1 = dict(self.param_up[self.basis][category])['d1']
            e_d2 = dict(self.param_up[self.basis][category])['d2']

            self.H_on_site[0, 0] = e_s
            for i in range(1, 4): self.H_on_site[i, i] = e_p
            for i in range(4, 7): self.H_on_site[i, i] = e_d1
            for i in range(7, 9): self.H_on_site[i, i] = e_d2

    def _build_neighbor(self, category_name, crystal_key):
        # To construct hamiltonian for a given a N neighbord

        parameters_up = dict(self.param_up[self.basis]['hopping'][category_name])
        if self.spin:
            parameters_dn = dict(self.param_dn[self.basis]['hopping'][category_name])

        sum_H_up = np.zeros([9, 9])
        sum_H_dn = np.zeros([9, 9])
        if self.spin:
                H_R_dn = np.zeros([9, 9])
        for pos in self.crystal[crystal_key]:
            l_val, m_val, n_val = direction_cosines(pos=pos)
            table = self.get_all_equations(l_val, m_val, n_val)
            
            H_R_up = np.zeros([9, 9])
            if self.spin:
                H_R_dn = np.zeros([9, 9])

            for (orb1, orb2), expr in table.items():
                i = ORBITAL[orb1]
                j = ORBITAL[orb2]

                val_up = float(expr.subs(parameters_up))
                H_R_up[i, j] = val_up
                if i != j:
                    H_R_up[j, i] = val_up

                if self.spin:
                    val_dn = float(expr.subs(parameters_dn))
                    H_R_dn[i, j] = val_dn
                    if i != j:
                        H_R_dn[j, i] = val_dn

            if self.spin:
                self.H_R_up[tuple(pos)] = H_R_up
                self.H_R_dn[tuple(pos)] = H_R_dn
                sum_H_up += H_R_up
                sum_H_dn += H_R_dn
            else:
                self.H_R[tuple(pos)] = H_R_up
                sum_H_up += H_R_up

        if self.spin:
            return sum_H_up, sum_H_dn
        else:
            return sum_H_up



    def first_NN(self):
        return self._build_neighbor("first_neighbor", "first_NN")

    def second_NN(self):
        return self._build_neighbor("second_neighbor", "second_NN")

    def third_NN(self):
        return self._build_neighbor("third_neighbor", "third_NN")

    def build_H_k(self, k_vec):
        # To apply bloch sum
        if self.spin:
            # On site
            H_k_up = np.array(self.H_on_site[0], dtype=complex)
            H_k_dn = np.array(self.H_on_site[1], dtype=complex)
            # + Neighbors
            for pos_tuple, H_R in self.H_R_up.items():
                phase = np.exp(1j * np.dot(k_vec, pos_tuple))
                H_k_up += phase * H_R
                
            for pos_tuple, H_R in self.H_R_dn.items():
                phase = np.exp(1j * np.dot(k_vec, pos_tuple))
                H_k_dn += phase * H_R

            return H_k_up, H_k_dn
        
        else:
            H_k = np.array(self.H_on_site, dtype=complex)
            
            for pos_tuple, H_R in self.H_R.items():
                phase = np.exp(1j * np.dot(k_vec, pos_tuple))
                H_k += phase * H_R
                
            return H_k


    def calculate_bands(self, k_path):
        # solve the secular equation for a k path
        num_k = len(k_path)
        
        if self.spin:
            energies_up = np.zeros((num_k, 9))
            energies_dn = np.zeros((num_k, 9))
            
            for i, k_vec in enumerate(k_path):
                H_k_up, H_k_dn = self.build_H_k(k_vec)
                # return eigenvalues for each hamiltonian
                energies_up[i, :] = np.linalg.eigh(H_k_up)[0]
                energies_dn[i, :] = np.linalg.eigh(H_k_dn)[0]
                
            return energies_up, energies_dn
        
        else:
            energies = np.zeros((num_k, 9))
            
            for i, k_vec in enumerate(k_path):
                H_k = self.build_H_k(k_vec)
                energies[i, :] = np.linalg.eigh(H_k)[0]
                
            return energies


def fast_band_plot(energies_up, energies_dn, k_path, k_points):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))


    for i in range(9):
        label_up = 'Spin Up' if i == 0 else ""
        label_dn = 'Spin Down' if i == 0 else ""
        
        plt.plot(energies_up[:, i], color='blue', linestyle='-', label=label_up)
        plt.plot(energies_dn[:, i], color='red', linestyle='--', label=label_dn)

    nodos_idx = [0, k_points, 2 * k_points, 3 * k_points, len(k_path) - 1]
    etiquetas = ['$\Gamma$', 'H', 'N', '$\Gamma$', 'P']

    for idx in nodos_idx:
        plt.axvline(x=idx, color='black', linewidth=0.8, linestyle=':')

    plt.xticks(nodos_idx, etiquetas, fontsize=12)
    plt.ylabel("Energy (eV)", fontsize=12)
    plt.xlim(0, len(k_path) - 1)
    # plt.ylim(-1.5, 2.0) # 
    plt.title("Band Structure", fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

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

    # lattice parameter hardcoded
    a = 5.40570 # a.u.
    a = au_to_angstrom(a)
    crystal = Crystal(structure="bcc", a=a, NN = 3)
    crystal.build_NN()

    print("pos")
    print(crystal.pos)

    data = ["data/SK_2-CP_Fe_bcc_up.json", # up data
            "data/SK_2-CP_Fe_bcc_dn.json"] # dn data

    H = Hamiltonian(crystal.pos, spin = True)
    H.load_param(data)
    H.on_site()
    H.first_NN()
    H.second_NN()
    H.third_NN()

    # Debug with high symmetry points
    factor = 2 * np.pi / a
    Gamma = np.array([0.0, 0.0, 0.0]) * factor
    H_pt  = np.array([0.0, 1.0, 0.0]) * factor
    N     = np.array([0.5, 0.5, 0.0]) * factor
    P     = np.array([0.5, 0.5, 0.5]) * factor

    path_nodes = [Gamma, H_pt, N, Gamma, P]
    k_points = 40
    k_path = []

    for i in range(len(path_nodes) - 1):
        segment = np.linspace(path_nodes[i], path_nodes[i+1], k_points, endpoint=(i == len(path_nodes) - 2))
        k_path.append(segment)

    k_path = np.vstack(k_path)

    energies_up, energies_dn = H.calculate_bands(k_path)
    
    energies_up = ry_to_eV(energies_up)
    energies_dn = ry_to_eV(energies_dn)

    fast_band_plot(energies_up, energies_dn, k_path, k_points)
    print(f"Forma de las bandas up: {energies_up.shape}")
