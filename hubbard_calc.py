from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from .geometry import *
from .tools import *
from hubbard import HubbardHamiltonian, sp2, density



def hub(g:Geometry, t, U, S:int=0):
    """
    Create the Hubbard Hamiltonian for a given geometry.
    Args:
        g: Geometry object
        t: hopping parameter
        U: Hubbard U
        S: spin quantum number
    Returns:
        HubbardHamiltonian object
    """
    h0 = sp2(g, t1=t, t2=0, t3=0, spin='polarized')
    hh = HubbardHamiltonian(h0, q=(len(h0)/2+S, len(h0)/2-S), U=U, kT=0.025)
    # randomly select six numbers from 1 to g.na
    randints = np.random.randint(1, g.na, 6)
    hh.set_polarization(randints[:3], dn=randints[3:])
    dn = hh.converge(density.calc_n, tol=1e-10, print_info=False)
    return hh




def plot_spin_density(g: Geometry, h: HubbardHamiltonian,
                      save: bool=False,
                      save_name='tmp',
                      save_format='svg',
                      save_path='.'):
    """
    Plot the spin density from Hubbard Hamiltonian calculation.
    Args:
        g: Geometry object
        h: HubbardHamiltonian object
        save: whether to save the figure
        save_name: name of the figure
        save_format: format of the figure
        save_path: path to save the figure
    """

    # Create a custom colormap
    colors = ["#D41159", "white", "#1A85FF"]
    cmap_custom = LinearSegmentedColormap.from_list("custom", colors)

    size = get_size(g)
    fig, ax = plt.subplots(figsize=[5*size[0]/size[1], 5])

    C_list = [i for i in range(g.na)]
    CC_bonds = dict()
    for cc in combinations(C_list, 2):
        r = g.rij(*cc)
        if r < 1.6:
            CC_bonds.update({cc: r})

    for cc, bond_length in CC_bonds.items():
        xy = g.xyz[cc,:2]
        plt.plot(xy[:,0], xy[:,1], color='k', linewidth=1, zorder=1)

    # plot spin density
    dotsize = 1000
    n = h.n
    SD = n[0,:] - n[1,:]
    colorbar_max = np.abs(SD).max()
    colorbar_min = -np.abs(SD).max()

    for a in g:
        norm_SD = (SD[a] - colorbar_min) / (colorbar_max-colorbar_min)
        color = cmap_custom(norm_SD)
        sc = ax.scatter(*g.xyz[a,:2], dotsize*np.abs(SD[a]), color=color, zorder=2)

    plt.xlim(g.xyz[:,0].min()-2, g.xyz[:,0].max()+2)
    ax.set_aspect('equal')
    plt.axis('off')
    # Create a symmetric Normalize instance for the colormap range with specified limits
    norm = mcolors.Normalize(vmin=colorbar_min, vmax=colorbar_max)
    # Create the colorbar using the custom colormap and the normalization instance
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_custom), ax=ax,
                      shrink=0.5)
    cb.set_label(r'$Q_\uparrow-Q_\downarrow$ ($e$)')
    if save:
        for fmt in save_format.split(','):
            filename = os.path.join(save_path, save_name+'.HubbardModelSpinDensity.'+fmt)
            plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
