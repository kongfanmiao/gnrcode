import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import py3Dmol
from sisl import Geometry, Atom, AtomicOrbital, plot
from sklearn.preprocessing import scale
from typing import Union

import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from .tools import deprecated, timer


@deprecated
def adjust_axes(
    geom: Geometry,
    ao: int,
    ax: int,
    ay: int=None,
    CC_bond_length=1.42,
    plot_geom=False,
):
    """
    Rotate the geometry in xy plane to align the periodic direction along x axis
    by using linear transformation (basic linear algebra). You should provide the
    indices of three atoms to define periodic direction as well as direction 
    perpendicular to it
    Argument:
        geom: sisl Geometry object
        ao: index of atom that sits in the origin
        ax: index of atom that you want to be aligned along x axis
        ay: index of atom that you want to be aligned along y axis
        CC_bond_length: set the C-C bond_length. It works as a scale factor. If you 
            want to specify the rx and ry vector length
        plot_geom: choose if to plot the geometry after rotation
    """
    adjust_bond_length(geom, bond_length=CC_bond_length)
    geom = rotate_gnr(geom, [ao,ax], plot_geom=plot_geom)



def set_cell(geom: Geometry, 
             a, b=30, c=30, 
             bond_length=1.42):
    """
    set the length of the unit cell vector along the ribbon direction.
    Adjust the bond length if needed
    Argument:
        geom: sisl Geometry object
        a: length of the unit cell vector along the ribbon direction
        b: length of the unit cell vector along the y perpendicular direction
        c: length of the unit cell vector along the x perpendicular direction
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        geom.cell[0, :] = a
    elif isinstance(a, (int, float)):
        geom.cell[0, :] = [a, 0, 0]
    if isinstance(b, (list, tuple, np.ndarray)):
        geom.cell[1, :] = b
    elif isinstance(b, (int, float)):
        geom.cell[1, :] = [0, b, 0]
    if isinstance(c, (list, tuple, np.ndarray)):
        geom.cell[2, :] = c
    elif isinstance(c, (int, float)):
        geom.cell[2, :] = [0, 0, c]
    geom.set_nsc([3, 1, 1])


def adjust_bond_length(geom: Geometry, bond_length=1.42):
    # if the bond length is not 1.42, adjust it
    min_bond = 10000
    n = None
    for i,a,_ in geom.iter_species():
        if a.Z == 6:
            if not n:
                n = i
            else:
                min_bond = min(min_bond, geom.rij(n,i))
    if min_bond != bond_length:
        geom.xyz = geom.xyz * bond_length / min_bond
    return geom


def write_coord(geom: Geometry, name, path=None):
    """
    write the standardised coordinates to new xyz file
    """
    file_name = name + "_st.xyz"
    if not path:
        file_path = "./files/" + file_name
    else:
        file_path = os.path.join(path, file_name)
    coords = geom.xyz
    with open(file_path, "w") as fout:
        fout.write(str(len(coords)) + "\n\n")
        for i in range(len(coords)):
            fout.write(
                "{}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(
                    geom.atoms[i].symbol, *geom.xyz[i]
                )
            )


def create_geometry(
    name, path=None,
    cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
    plot_geom=True,
    aid=True,
    sc=False,
    adjust_bond=True,
    bond=1.42,
    text_color='green', 
    text_font_size=16,
    figsize=[5,5]
) -> Geometry:
    """
    Read coordinates from .xyz file, move the geometry center to origin
    then create a geometry object, the initial lattice vectors are
    [[10,0,0],[0,10,0],[0,0,10]]
    User set_cell method later to correct it.
    by default, for tight binding model. 
    Adjust bond length to 1.42 
    """
    file_name = name + ".xyz"
    if not path:
        file_path = "./files/" + file_name
    else:
        file_path = os.path.join(path, file_name)

    coordinates = []
    raw_atom = []
    atoms = []
    with open(file_path, "r") as fin:
        for i in range(2):
            fin.readline()
        for line in fin:
            line = line.strip()
            line = line.split()
            raw_atom.append(str(line[0]))
            atoms.append(Atom(str(line.pop(0)), R=bond*1.01))
            coordinates.append([float(i) for i in line])
    coordinates = np.array(coordinates)
    coordinates = coordinates - np.mean(coordinates, 0)

    geom = Geometry(coordinates, atoms, cell)
    if adjust_bond:
        adjust_bond_length(geom, bond_length=bond)
    if plot_geom:
        display2D(geom, aid=aid, sc=sc, text_color=text_color, 
        text_font_size=text_font_size, figsize=figsize)

    return geom


def move_to_origo(gnr, plot_geom=True, sc=True, aid=False):
    """
    Move the geometry center to origin
    """

    gnr = gnr.translate(-gnr.center())
    
    if plot_geom:
        display2D(gnr, sc=sc, aid=aid)

    return gnr


def move_to_center(g, axis="xyz", plot_geom=True,
    sc=True, aid=False):
    """
    Move the geometry to the center of supercell
    """
    cell_center = g.center(what="cell")
    # The internal method of xyz center is stupid, don't use it
    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    maxxyz = np.amax(xyz, 0)
    geom_center = (minxyz + maxxyz) / 2
    xvector = cell_center[0] - geom_center[0] if "x" in axis else 0
    yvector = cell_center[1] - geom_center[1] if "y" in axis else 0
    zvector = cell_center[2] - geom_center[2] if "z" in axis else 0
    g = g.translate([xvector, yvector, zvector])

    if plot_geom:
        display2D(g, sc=sc, aid=aid)

    return g


def move_to_xcenter(g, plot_geom=True):
    """
    Move the geometry to the centre in x direction
    """
    gx = move_to_center(g, "x", plot_geom=plot_geom)
    return gx


def move_to_ycenter(g, plot_geom=True):
    """
    Move the geometry to the centre in y direction
    """
    gy = move_to_center(g, "y", plot_geom=plot_geom)
    return gy


def move_to_xycenter(g, plot_geom=True):
    """
    Move the geometry to the centre in x and y direction
    """
    gxy = move_to_center(g, "xy", plot_geom=plot_geom)
    return gxy


def move_to_zcenter(g, plot_geom=True):
    """
    Move the geometry to the centre in z direction
    """
    gz = move_to_center(g, "z", plot_geom=plot_geom)
    return gz


def guess_figsize(g):
    """
    Guess the figure size according to the geometry
    """
    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    maxxyz = np.amax(xyz, 0)
    length = maxxyz - minxyz
    ratio = length[0] / length[1]
    figsize = (4 * ratio, 4)
    return figsize



def display2D(g, aid=False, sc=True, rotate=False, figsize=None,
              text_color='green', text_font_size=16, **kwargs):

    mpl.rcParams['text.color'] = text_color
    mpl.rcParams['font.size'] = text_font_size

    if figsize is None:
        figsize = guess_figsize(g)

    plt.figure(figsize=figsize)
    if rotate:
        if round(g.cell[0, 1], 5) != 0:
            angle = np.degrees(np.arctan(g.cell[0, 1] / g.cell[0, 0]))
            g = g.rotate(-angle, [0, 0, 1])
    plot(g, atom_indices=aid, supercell=sc, **kwargs)
    plt.axis("equal")

    # if length[0] > length[1]:  # x is larger
    #     plt.xlim(minxyz[0] - 6, maxxyz[0] + 6)
    # else:
    #     plt.ylim(minxyz[1] - 2, maxxyz[1] + 2)
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.show()




def display3D(what, width=500, height=500, rotation=0, zoom=1,
    index=False, index_font='sans-serif', index_font_size=15,
    index_color='black', style='ball-stick'):
    if isinstance(what, str):
        with open(what, 'r') as f:
            xyzstr = f.read()
    elif isinstance(what, Geometry):    
        xyzstr = "{}\n\n".format(what.na)
        xyz = what.xyz
        xyzstr = xyzstr + '\n'.join(["{}\t{:.8f}\t{:.8f}\t{:.8f}".format(
            a.tag, *xyz[ia]) for ia, a, _ in what.iter_species()])
    xyzview = py3Dmol.view(width=width, height=height)
    xyzview.addModel(xyzstr, 'xyz')
    if style == 'ball':
        xyzview.setStyle('sphere')
    elif style == 'ball-stick':
        xyzview.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3},
                          'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})
    if index:
        for i in range(xyz.shape[0]):
            _x,_y,_z = xyz[i,:]
            xyzview.addLabel(i, {'position':{'x':_x,'y':_y,'z':_z},
                                'showBackground': False,
                                'font': index_font,
                                'fontSize':index_font_size,
                                'fontColor':index_color,
                                'alignment': 'center'})
    
    xyzview.rotate(rotation)
    xyzview.zoomTo()
    xyzview.zoom(zoom)
    xyzview.show()



def slice_show(g, xlim=[0, 10], ylim=None, figsize=(8, 5), aid=True, sc=False):
    display2D(g, aid=aid, sc=sc, figsize=figsize)
    plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    else:
        ylim = [-1e3, 1e3]
    for i in g:
        if (xlim[0] < g.xyz[i, 0] < xlim[1]) and (ylim[0] < g.xyz[i, 1] < ylim[1]):
            plt.annotate(i, g.xyz[i, 0:2])


def connect(g1, g2, a1, a2, bond=[1.42, 0, 0]):
    """
    Connect two geometries, by default using a horizontal single C-C bond.
    """
    # first move the atom 0 to the origo
    g1 = g1.translate(-g1.xyz[0])
    g2 = g2.translate(-g2.xyz[0])
    Ra1 = g1.Rij(0, a1)
    Ra2 = g2.Rij(a2, 0)
    bond = np.array(bond)
    displ = Ra1 + bond + Ra2
    g = g1.add(g2, displ)

    return g


def attach_pz(g, basis="DZP"):
    """
    Attach the Pz orbitals to the geometry
    """
    if isinstance(g, Geometry):
        pass
    elif isinstance(g, Hamiltonian):
        g = g.geometry
        
    r = np.linspace(0, 5, 100)
    epsilon = 1.625
    f = np.exp(-epsilon * r)
    # Normalize
    A = 1 / (f.sum() * (r[1] - r[0]))
    f = A * f

    if basis == "DZP":
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=2, P=1)
    elif basis == "DZ":
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=2)
    elif basis == "SZP":
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=1, P=1)
    elif basis == "SZ":
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=1)
    C = Atom(6, pz)
    g.atoms.replace(g.atoms[0], C)


def find_sublattice(g:Geometry, bond_length=1.42):
    """
    Find the sublattice of a geometry and return the indices
    """
    # Start with atom 0
    # Find the direction which has length of sqrt(3)*a
    n = 1
    while not (abs(g.rij(n,0)-bond_length*np.sqrt(3)) < 0.05):
        n += 1
    # all atoms in the row that is parallel to v1 belong to 
    # same sublattice
    v1 = g.Rij(n,0)
    # the vector that is perpendicular to v2
    v2 = np.array([-v1[1], v1[0], 0])
    v2 = v2/np.linalg.norm(v2)
    # project vector Rij(0,a) to v2
    # if it's 0, 1.5*1.42, 3*1.42, ..., it's sublattice A
    # if it's 1.42, 2.5*1.42, 4*1.42, ..., it's sublattice B
    Asublat = [0]
    Bsublat = []
    for i in range(g.na-1):
        # atom index
        a = i+1
        proj = g.Rij(0,a).dot(v2)
        tmp = (proj/1.42)/1.5
        tmp = abs(tmp-np.floor(tmp))
        if abs(tmp-1/3)<0.05 or abs(tmp-2/3)<0.05:
            Bsublat.append(int(a))
        else:
            Asublat.append(int(a))
    return Asublat, Bsublat


def mark_sublattice(g, figsize=None):
    """
    Mark the sublattice of a geometry
    """
    if not figsize:
        figsize = guess_figsize(g)
    A, B = find_sublattice(g)
    plt.figure(figsize=figsize)
    plt.axis('equal')
    xyz = g.xyz
    Axyz = xyz[A,:]
    Bxyz = xyz[B,:]
    plt.scatter(Axyz[:,0], Axyz[:,1], 20, color='r')
    plt.scatter(Bxyz[:,0], Bxyz[:,1], 20, color='b')
    for i in range(len(A)):
        plt.text(Axyz[i,0], Axyz[i,1], 'A', fontsize=12, color='r')
    for i in range(len(B)):
        plt.text(Bxyz[i,0], Bxyz[i,1], 'B', fontsize=12, color='b')



def plot_gnr(gnr: Geometry,
            repetitions: int,
            bond_color='k',
            bond_thickness=0.5,
            highlight_middle=True,
            highlight_color='r',
            desired_bond_length_mm=4,
            save=True,
            save_name='tmp',
            save_path='.',
            gradient_color=False):
    """
    Plot a graphene nanoribbon.
    Args:
        gnr (Geometry): The graphene nanoribbon to plot.
        repetitions (int): The number of repetitions of the unit cell.
        bond_color (str): The color of the bonds.
        bond_thickness (float): The thickness of the bonds.
        highlight_middle (bool): Whether to highlight the middle unit cell.
        highlight_color (str): The color of the highlighted unit cell.
        desired_bond_length_mm (float): The desired bond length in mm.
        save (bool): Whether to save the plot.
        save_name (str): The name of the saved plot.
        save_path (str): The path to save the plot.
    Returns:
        None
    """
    # rotate the gnr to along the x-axis
    gnr = rotate_gnr(gnr, plot_geom=False)

    mm_to_inches = 0.0393701
    unit_cell_xyz = gnr.xyz
    translation_vector = gnr.cell[0]
    
    # Calculate the real size of the molecule
    total_translation_vector = np.array(translation_vector) * (repetitions - 1)
    min_coords = np.min(unit_cell_xyz, axis=0)
    max_coords = np.max(unit_cell_xyz, axis=0)
    molecule_size = max_coords - min_coords + total_translation_vector
    print(molecule_size)
    
    # Calculate the average C-C bond length
    avg_bond_length = np.mean([np.linalg.norm(atom1 - atom2) for atom1 in unit_cell_xyz for atom2 in unit_cell_xyz if 1.3 < np.linalg.norm(atom1 - atom2) < 1.7])

    # Determine the scaling factor
    scaling_factor = (desired_bond_length_mm / avg_bond_length) * mm_to_inches

    # Adjust the figure size based on the scaling factor
    fig_width, fig_height = molecule_size[:2] * scaling_factor
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    

    if isinstance(bond_color,str):
        bond_color = mcolors.to_rgb(bond_color)
    if isinstance(highlight_color, str):
        highlight_color = mcolors.to_rgb(highlight_color)

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [highlight_color, bond_color])

    def draw_bond(atom1, atom2, color, thickness, linestyle='-'):
        xs = [atom1[0], atom2[0]]
        ys = [atom1[1], atom2[1]]
        ax.plot(xs, ys, color=color, linewidth=thickness, linestyle=linestyle, zorder=1)

    def draw_gradient_bond(atom1, atom2, thickness, reverse_gradient=False):
        x = np.linspace(atom1[0], atom2[0], 100)
        y = np.linspace(atom1[1], atom2[1], 100)
        cols = np.linspace(0, 1, len(x))

        if reverse_gradient:
            cols = np.flip(cols)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=custom_cmap)
        lc.set_array(cols)
        lc.set_linewidth(thickness)
        line = ax.add_collection(lc)


    for i in range(repetitions):
        translated_xyz = unit_cell_xyz + np.array(translation_vector) * i
        is_middle = highlight_middle and (i == repetitions // 2)

        for atom1 in translated_xyz:
            for atom2 in translated_xyz:
                distance = np.linalg.norm(atom1 - atom2)
                if 1.3 < distance < 1.7:
                    if is_middle:
                        draw_bond(atom1, atom2, highlight_color, bond_thickness)
                    else:
                        draw_bond(atom1, atom2, bond_color, bond_thickness)
            
            # connection bonds between two unit cells
            if i < repetitions - 1:
                translated_xyz_next = unit_cell_xyz + np.array(translation_vector) * (i + 1)
                for atom2 in translated_xyz_next:
                    distance = np.linalg.norm(atom1 - atom2)
                    if 1.3 < distance < 1.7:
                        if gradient_color:
                            if is_middle or (highlight_middle and i == (repetitions // 2) - 1):
                                reverse_gradient = (i == (repetitions // 2) - 1)
                                draw_gradient_bond(atom1, atom2, bond_thickness, reverse_gradient)
                            else:
                                draw_bond(atom1, atom2, bond_color, bond_thickness)
                        else:
                            if is_middle or (highlight_middle and i == (repetitions // 2) - 1):
                                # Change bond color at the middle of the bond
                                middle_point = (atom1 + atom2) / 2
                                if i == (repetitions // 2) - 1:
                                    draw_bond(atom1, middle_point, bond_color, bond_thickness)
                                    draw_bond(middle_point, atom2, highlight_color, bond_thickness)
                                else:
                                    draw_bond(atom1, middle_point, highlight_color, bond_thickness)
                                    draw_bond(middle_point, atom2, bond_color, bond_thickness)
                            else:
                                draw_bond(atom1, atom2, bond_color, bond_thickness)

            # Add virtual connection bonds for leftmost and rightmost unit cells
            if i == 0:
                translated_xyz_prev = unit_cell_xyz - np.array(translation_vector)
                for atom2 in translated_xyz_prev:
                    distance = np.linalg.norm(atom1 - atom2)
                    if 1.3 < distance < 1.7:
                        draw_bond(atom1, atom2, bond_color, bond_thickness, linestyle='--')

            elif i == repetitions - 1:
                translated_xyz_next2 = translated_xyz_next + np.array(translation_vector)
                for atom2 in translated_xyz_next2:
                    distance = np.linalg.norm(atom1 - atom2)
                    if 1.3 < distance < 1.7:
                        draw_bond(atom1, atom2, bond_color, bond_thickness, linestyle='--')


    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    if save:
        png = os.path.join(save_path, f'{save_name}.png')
        svg = os.path.join(save_path, f'{save_name}.svg')
        plt.savefig(png, dpi=600, bbox_inches='tight', transparent=True)
        plt.savefig(svg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()



def rotate_gnr(g: Geometry, axis:list=[], plot_geom:bool=False):
    """
    Rotate a GNR to align the given axis along the x-axis.
    The axis is defined by two points given in axis argument, 
    where the two points are indices of atoms in the GNR.
    If axis is not given, the axis will be defined by the first
    row of the g.cell.
    """
    if not axis:
        v = g.cell[0]
    else:
        # get the two points
        p1 = g.xyz[axis[0]]
        p2 = g.xyz[axis[1]]
        # get the vector
        v = p2 - p1
    # get the angle
    theta = -np.degrees(np.arctan2(v[1], v[0]))
    # rotate the geometry
    g = g.rotate(theta, [0,0,1])
    # set the y component of the cell to 20

    g.cell[1] = np.array([0, 20, 0])
    g = move_to_center(g, plot_geom=plot_geom)
    return g



def vec(length, angle):
    """
    Return a vector of given length and angle (in degrees).
    The vector is three dimensional, with the z component being zero.
    """
    angle = np.radians(angle)
    return np.array([length * np.cos(angle), length * np.sin(angle), 0])



def add_hydrogen(g:Geometry, index:int, direction:Union[list, np.ndarray]):
    """
    add hydrogen passivation to the gnr at atom with given index. The C-H bond 
    direction is given by direction, which is a 3D vector, eg. [1,0,0] for x direction.
    The C-H bond length is determined by the average of other existing C-H bonds
    in the geometry, if no other C-H bond exist, then by default the new C-H 
    bond length is 1.09 angstrom.
    """
    # get the C-H bond length
    # if no C-H bond exist, use 1.09 angstrom
    bond_length = 1.09
    # if C-H bond exist, use the average length
    # get the C-H bond length, the C atom index and H atom index in a C-H bond
    # are not necessarily adjacent
    # get a hyodrogen atom at the same time
    hydrogen = None
    bond_length_list = []
    for i in range(g.na):
        if g.atoms[i].tag == 'C':
            for j in range(g.na):
                if g.atoms[j].tag == 'H':
                    if not hydrogen:
                        hydrogen = g.atoms[j]
                    if np.linalg.norm(g.xyz[i] - g.xyz[j]) < 1.4:
                        bond_length_list.append(np.linalg.norm(g.xyz[i] - g.xyz[j]))
                # if no H atom found, then use the default bond length
                else:
                    hydrogen = Atom('H', [0,0,0])
    if len(bond_length_list) > 0:
        bond_length = np.mean(bond_length_list)
    
    # create geometry for hydrogen atom
    g1 = Geometry([[0,0,0]], atoms=[hydrogen])
    # then get the C-H bond vector
    bond_vector = np.array(direction) * bond_length
    # then add the hydrogen atom to the geometry using connect method
    g = connect(g, g1, index, 0, bond_vector)
        
    return g
