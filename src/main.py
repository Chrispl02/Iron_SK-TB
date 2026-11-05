import numpy as np 
from crystal import *
from hamiltonian import *
from dos import *


def band_structure(a, NN, structure, data):
    
    crystal = Crystal(structure=structure, a=a, NN = NN)
    crystal.build_NN()
    
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

#####
a = 5.40570 # a.u.
a = au_to_angstrom(a)

NN = 3
structure = 'bcc'

data = ["data/SK_2-CP_Fe_bcc_up.json", # up data
            "data/SK_2-CP_Fe_bcc_dn.json"] # dn data


# ---- Calculate band structures 

band_structure(a, NN, structure, data)


# --- Calculate DoS


