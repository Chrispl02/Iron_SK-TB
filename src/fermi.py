import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from hamiltonian import ry_to_eV
from matplotlib.patches import Patch

def fermi_surface(H, crystal, E_F, N=61, pad=1.05):
    """E_F in eV. Gives {(spin, band): (vertices, faces)} cutted to BZ."""
    lim = pad * np.abs(crystal.bz_vertices).max()
    ax = np.linspace(-lim, lim, N)
    dk = ax[1] - ax[0]
    K = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    E = [ry_to_eV(e) for e in H.eigvals_batch(K)]

    # k·G <= |G|**2/2 for G short define the Wigner-Seitz zone
    idx = [i for i in itertools.product((-1, 0, 1), repeat=3) if any(i)]
    G = np.array(idx) @ crystal.b_vec

    surfaces = {}
    for spin, Es in zip(("up", "dn"), E):
        Es = Es.reshape(N, N, N, -1)
        for n in range(Es.shape[-1]):
            band = Es[..., n]
            if not (band.min() < E_F < band.max()):
                continue                                   # this band does not cross E_F
            v, f, _, _ = marching_cubes(band, level=E_F, spacing=(dk, dk, dk))
            v = v - lim
            cen = v[f].mean(axis=1)
            keep = (cen @ G.T <= 0.5 * (G**2).sum(axis=1) + 1e-9).all(axis=1)
            surfaces[(spin, n)] = (v, f[keep])
    return surfaces


def plot_fermi_surface(surfaces, crystal, spin="up", bands=None, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(7, 7)).add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(crystal.bz_faces, facecolors=(0, 0, 0, 0),
                                         edgecolors="gray", linewidths=0.6))
    cmap = plt.get_cmap("tab10")
    handles = []
    for (s, n), (v, f) in surfaces.items():
        if s != spin or (bands is not None and n not in bands) or len(f) == 0:
            continue
        color = cmap(n % 10)
        ax.add_collection3d(Poly3DCollection(v[f], facecolors=color, alpha=0.6,
                                             edgecolors="none"))
        handles.append(Patch(facecolor=color, alpha=0.6, label=f"Banda {n + 1}"))
    lim = np.abs(crystal.bz_vertices).max() * 1.05
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
           xlabel="kx (1/Å)", ylabel="ky (1/Å)", zlabel="kz (1/Å)",
           title=f"Superficie de Fermi, spin {spin}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend(handles=handles, loc="upper left")
    return ax

def write_bxsf(basename, H, crystal, E_F, N=40):
    # Xcrysden has .bxsf example files, this function writes a similar file for the Fermi surface
    frac = np.stack(np.meshgrid(*[np.arange(N) / N] * 3, indexing="ij"), -1).reshape(-1, 3)
    k = frac @ crystal.b_vec                          # 1/A
    E_up, E_dn = (ry_to_eV(e) for e in H.eigvals_batch(k))

    files = []
    for spin, E in (("up", E_up), ("dn", E_dn)):
        crossing = [n + 1 for n in range(E.shape[1]) if E[:, n].min() < E_F < E[:, n].max()]
        fname = f"{basename}_{spin}.bxsf"
        with open(fname, "w") as f:
            f.write("BEGIN_INFO\n")
            f.write(f"  # Band-XCRYSDEN-Structure-File, spin {spin}\n")
            f.write(f"  # Bandas que cruzan E_F: {' '.join(map(str, crossing))}\n")
            f.write(f"  Fermi Energy: {E_F:.8f}\n")
            f.write("END_INFO\n\n")
            f.write("BEGIN_BLOCK_BANDGRID_3D\n band_energies\n BEGIN_BANDGRID_3D\n")
            f.write(f" {E.shape[1]}\n {N} {N} {N}\n 0.0 0.0 0.0\n")
            for v in crystal.b_vec:                   # vectors in mesh: rows b_i
                f.write(" %.10f %.10f %.10f\n" % tuple(v))
            for n in range(E.shape[1]):
                f.write(f" BAND: {n + 1}\n")
                for s in range(0, len(E), 6):
                    f.write(" " + " ".join("%.8f" % x for x in E[s:s + 6, n]) + "\n")
            f.write(" END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3D\n")
        files.append(fname)
        print(f"{fname}: spin {spin}, bandas que cruzan E_F = {crossing}")
    return files