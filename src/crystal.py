import numpy as np
import matplotlib.pyplot as plt
import itertools
import sympy as sp
from scipy.spatial import Voronoi, ConvexHull

def au_to_angstrom(x):
    return x * 0.5291772109

def direction_cosines(pos, sym=False):
    x, y, z = pos
    if not sym:
        r = np.sqrt(x**2 + y**2 + z**2)
        return x / r, y / r, z / r
    else:
        x_sym, y_sym, z_sym = sp.sympify(x), sp.sympify(y), sp.sympify(z)
        r_sym = sp.sqrt(x_sym**2 + y_sym**2 + z_sym**2)
        return x_sym / r_sym, y_sym / r_sym, z_sym / r_sym
        
class Crystal():
    def __init__(self,structure:str, a, NN:int):
        if structure == None:
            raise Exception("No structure given, free structure is not implemented!") 
        if NN < 1:
            raise Exception("Your number of neighbors doesnt make sense")
        
        self.structure = structure
        self.NN = NN
        self.a = a # lattice parameter

    def build_NN(self): # The correct way to create this position is with the real basis (To DO later)
        if self.structure == "bcc":
            self.pos = {"self": [0,0,0]}
            if self.NN >= 1: # 1st neighbord
                self.pos["first_NN"] = self.a/2 * np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]])
            if self.NN >= 2: # 2st neighbord
                self.pos["second_NN"] = self.a * np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
            if self.NN >= 3: # 3rd neighbord
                self.pos["third_NN"] = self.a * np.array([[1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
                                            [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
                                            [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]])

        # possible To DO? Implement one more layer to use a same function to create neighbors in each symmetry structure

    def build_lattice(self):
        
        if self.structure == "bcc":
            self.a_vec = self.a / 2 * np.array([[-1,  1,  1],[ 1, -1,  1],[ 1,  1, -1]])
        else:
            raise NotImplementedError(f"{self.structure} not implemented")

        self.volume = abs(np.dot(self.a_vec[0], np.cross(self.a_vec[1], self.a_vec[2])))
        self.b_vec = 2 * np.pi * np.linalg.inv(self.a_vec).T
        self.bz_volume = (2 * np.pi) ** 3 / self.volume

        # to check a_i · b_j = 2π delta_ij
        assert np.allclose(self.a_vec @ self.b_vec.T, 2 * np.pi * np.eye(3))

    def load_irreducible_mesh(self, spts_file, tetra_file):
        spts = np.loadtxt(spts_file) # Irreducible points
        self.k_irr_frac = spts[:, :3]                    # coeficientes en b_i
        self.k_irr_mult = spts[:, 3].astype(int)         # degeneración
        self.k_irr_cart = self.k_irr_frac @ self.b_vec   # (30, 3) en 1/Å

        tet = np.loadtxt(tetra_file, dtype=int) # Tetrahedas
        self.tetra = tet[:, :4] - 1                      # índices 0-based
        self.tetra_w = tet[:, 4] / tet[:, 4].sum()       # fracción de BZ por línea

        # chequeo: 6 tetraedros por punto de la malla completa
        assert tet[:, 4].sum() == 6 * self.k_irr_mult.sum()

    # --- Plotting functions ---

    def plot_3d(self):
        if not hasattr(self, "pos"):
            raise Exception("cannot plot not created structure")

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        styles = {
            "self": ("black", "o"),
            "first_NN": ("red", "o"),
            "second_NN": ("blue", "s"),
            "third_NN": ("green", "^")
        }

        for shell, coordinates in self.pos.items():
            coordinates = np.array(coordinates).reshape(-1, 3)
            color, marker = styles[shell]

            ax.scatter(*coordinates.T, color=color, marker=marker,
                    s=80, label=shell)

            if shell != "self":
                for point in coordinates:
                    ax.plot(*np.array([[0, 0, 0], point]).T,
                            color=color, alpha=0.25)

        r = self.NN * self.a
        ax.set(xlim=(-r, r), ylim=(-r, r), zlim=(-r, r),
            xlabel="X (Å)", ylabel="Y (Å)", zlabel="Z (Å)",
            title=f"{self.structure.upper()} | a={self.a} Å | NN={self.NN}")

        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        plt.tight_layout()
        plt.show()


    def build_BZ(self, nshell=2):
        # Just to create the brillouin zone shape for plotting
        n = np.arange(-nshell, nshell + 1)
        idx = np.array(list(itertools.product(n, n, n)))
        G = idx @ self.b_vec
        i0 = np.where((idx == 0).all(axis=1))[0][0]

        vor = Voronoi(G)
        region = vor.regions[vor.point_region[i0]]
        self.bz_vertices = vor.vertices[region]

        def order_face(v, normal):
            """Ordena los vértices de una cara cíclicamente (para Poly3DCollection)."""
            c = v.mean(axis=0)
            n = normal / np.linalg.norm(normal)
            u = v[0] - c
            u -= n * np.dot(u, n)
            u /= np.linalg.norm(u)
            w = np.cross(n, u)
            ang = np.arctan2((v - c) @ w, (v - c) @ u)
            return v[np.argsort(ang)]

        faces = []
        for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
            if i0 in (p, q) and -1 not in rv:
                normal = G[q] if p == i0 else G[p]
                faces.append(order_face(vor.vertices[rv], normal))
        self.bz_faces = faces

        # check: for bcc
        assert np.isclose(ConvexHull(self.bz_vertices).volume, self.bz_volume)

    

    def high_symmetry_points(self):
        f = 2 * np.pi / self.a
        if self.structure == "bcc":
            return {"Gamma": f * np.array([0.0, 0.0, 0.0]),
                    "H":     f * np.array([0.0, 1.0, 0.0]),
                    "N":     f * np.array([0.5, 0.5, 0.0]),
                    "P":     f * np.array([0.5, 0.5, 0.5])}
        raise NotImplementedError

    '''def build_kpath(self, labels=("Gamma", "H", "N", "Gamma", "P"), npts=40):
        pts = self.high_symmetry_points()
        nodes = [pts[l] for l in labels]
        path = [np.linspace(nodes[i], nodes[i + 1], npts, endpoint=(i == len(nodes) - 2))
                for i in range(len(nodes) - 1)]
        return np.vstack(path)'''
    
    def plot_BZ(self, kpoints = False, kmesh=False):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(self.bz_faces, facecolors="cyan",
                                             edgecolors="k", alpha=0.2))
        # Plot high symmetry points
        for name, k in self.high_symmetry_points().items():
            ax.scatter(*k, color="red")
            ax.text(*k, f" {name}")
        # Plot irreducible k-points of spts.txt -> Given kpoints
        if kpoints == True:
            print("self.b_vec", self.b_vec)
            for i,k in enumerate(self.k_irr_cart):
                print(i,": ", k)
                ax.scatter(*k, color="blue")
                ax.text(*k, f" {i}")
        # Plot irreducible k-points of spts.txt -> Given kpoints
        if kmesh == True:
            for i,k in enumerate(self.k_cart):
                ax.scatter(*k, color="yellow", marker='*')
        
        lim = np.abs(self.bz_vertices).max() * 1.1
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
               xlabel="kx (1/Å)", ylabel="ky (1/Å)", zlabel="kz (1/Å)")
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()

    # ---------- Experimental: about creating the mesh ----------
    def build_kmesh(self, N):
        """Mesh centred in Gamma"""
        N = (N, N, N) if isinstance(N, int) else tuple(N)
        self.mesh_shape = N
        frac = np.stack(np.meshgrid(*[np.arange(n) / n for n in N], indexing="ij"), axis=-1)
        self.k_frac = frac.reshape(-1, 3)          # índice plano = (i*N2 + j)*N3 + l
        self.k_cart = self.k_frac @ self.b_vec

    def build_tetrahedra(self):
        """Divide cada subcelda en 6 tetraedros (triangulación de Kuhn) sobre la
        diagonal más corta en cartesianas. Devuelve índices planos de la malla, con
        condiciones periódicas. Cada tetraedro pesa 1/(6·N1·N2·N3) de la BZ."""
        N = np.array(self.mesh_shape)
        signs = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
        s = np.array(min(signs, key=lambda s: np.linalg.norm(np.array(s) @ self.b_vec)))
        start = (1 - s) // 2                       # esquina inicial de la diagonal

        local = []
        for perm in itertools.permutations(range(3)):
            v = start.copy()
            verts = [v.copy()]
            for ax in perm:
                v[ax] += s[ax]
                verts.append(v.copy())
            local.append(verts)
        local = np.array(local)                    # (6, 4, 3)

        cells = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"),
                         axis=-1).reshape(-1, 3)
        idx = (cells[:, None, None, :] + local[None]) % N      # (Ncell, 6, 4, 3)
        flat = (idx[..., 0] * N[1] + idx[..., 1]) * N[2] + idx[..., 2]
        self.tetra = flat.reshape(-1, 4)           # (6·Ncell, 4)
        self.tetra_weight = 1.0 / len(self.tetra)

            
            

        



if __name__ == "__main__":
    a = 5.40570 # a.u.
    a = au_to_angstrom(a)
    crystal = Crystal(structure="bcc", a=a, NN = 3)
    crystal.build_NN()
    crystal.plot_3d()

    # Test Brillouin zone

    crystal.build_lattice()
    crystal.build_BZ()
    crystal.build_kmesh(6) #24
    crystal.build_tetrahedra()
    crystal.load_irreducible_mesh("data/spts_bcc_v2.txt", "data/tetra_bcc.txt")

    
    crystal.plot_BZ(kpoints = True, kmesh = False)