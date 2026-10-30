import numpy as np
import matplotlib.pyplot as plt

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

    def build_NN(self):
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

            
            

        



if __name__ == "__main__":
    a = 5.40570 # a.u.
    a = au_to_angstrom(a)
    crystal = Crystal(structure="bcc", a=a, NN = 3)
    crystal.build_NN()
    crystal.plot_3d()
