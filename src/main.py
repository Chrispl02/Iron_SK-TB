from ast import get_docstring
import numpy as np 
from crystal import *
from hamiltonian import *
from dos import *
from scipy.optimize import brentq


def band_structure(H, a):

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


def plot_dos(energies, g_up, g_dn, e_fermi=0.0):
    e_shifted = energies - e_fermi
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(e_shifted, g_up, label='Spin Up', color='crimson', lw=2)
    plt.fill_between(e_shifted, 0, g_up, color='crimson', alpha=0.3)
    
    plt.plot(e_shifted, -g_dn, label='Spin Down', color='royalblue', lw=2)
    plt.fill_between(e_shifted, 0, -g_dn, color='royalblue', alpha=0.3)
    
    plt.axhline(0, color='black', lw=0.8, ls='--')
    plt.axvline(0, color='darkgreen', lw=1.5, ls=':', label='$E_F$')
    
    plt.xlabel(f'Energía - $E_F$ (eV)', fontsize=12)
    plt.ylabel('Densidad de Estados (Estados/eV)', fontsize=12)
    plt.title('Densidad de Estados (DoS) Resuelta por Espín', fontsize=14, fontweight='bold')
    
    y_ticks, _ = plt.yticks()
    plt.yticks(y_ticks, [str(abs(int(tick))) for tick in y_ticks])
    
    plt.xlim(min(e_shifted), max(e_shifted))
    plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.4, ls=':')
    plt.tight_layout()
    plt.show()


#####
a = 5.40570 # a.u.
a = au_to_angstrom(a)

NN = 3
structure = 'bcc'

data = ["data/SK_2-CP_Fe_bcc_up.json", # up data
            "data/SK_2-CP_Fe_bcc_dn.json"] # dn data

# ---- Create system
crystal = Crystal(structure=structure, a=a, NN = NN)
crystal.build_NN()

H = Hamiltonian(crystal.pos, spin = True)
H.load_param(data)
H.on_site()
H.first_NN()
H.second_NN()
H.third_NN()

# ---- Calculate band structures 
# band_structure(H, a)


# --- Calculate DoS
crystal.build_lattice()
crystal.load_irreducible_mesh("data/spts_bcc_v2.txt", "data/tetra_bcc.txt")

E_up, E_dn = H.calculate_bands(crystal.k_irr_cart)
E_up, E_dn = ry_to_eV(E_up), ry_to_eV(E_dn)

dos_up = TetrahedronDOS(E_up, crystal.tetra, crystal.tetra_w)
dos_dn = TetrahedronDOS(E_dn, crystal.tetra, crystal.tetra_w)

# Energy delimitation for DoS calculation
Egrid = np.linspace(min(E_up.min(), E_dn.min()) - 0.5, max(E_up.max(), E_dn.max()) + 0.5, 1000)

# DoS integrated from -infinity to E (to calculate the Fermi energy)
N_tot = lambda E: dos_up.integrated_dos(E)[0] + dos_dn.integrated_dos(E)[0]
 
# Fe: [ar] 4s2 3d6, valence electrons = 8.0
valance_e = 8.0
E_F = brentq(lambda E: N_tot(E) - valance_e, Egrid[0], Egrid[-1])

g_up, g_dn = dos_up.dos(Egrid), dos_dn.dos(Egrid)
m = dos_up.integrated_dos(E_F)[0] - dos_dn.integrated_dos(E_F)[0]

print("m", m)
print("E_F", E_F)

import matplotlib.pyplot as plt
plot_dos(Egrid, g_up, g_dn, E_F)


# Calculate Fermi surface 