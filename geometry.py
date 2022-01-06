import os
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
from sisl import Geometry, Atom, AtomicOrbital, plot


def adjust_axes(
    geom: Geometry,
    ao: int,
    ax: int,
    ay: int,
    rx=None,
    ry=None,
    bond_length=1.0,
    plot_geom=True,
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
        rx: distance between ao and ax
        ry: distance between ao and ay
        bond_length: set the bond_length. It works as a scale factor. If you 
            want to specify the rx and ry vector length, remeber that the default 
            bond_length is 1.0, not 1.42
        plot_geom: choose if to plot the geometry after rotation
    """
    # make sure it is a sisl Geometry object
    if not isinstance(geom, Geometry):
        raise TypeError("Please give a Geometry object as input")
    # calculate the vectors pointing from ao to ax and ay respectively
    Rx = geom.Rij(ao, ax)
    Ry = geom.Rij(ao, ay)
    # calculate the normal direction that is perpendicular to both x and y direction
    Rz = np.cross(Rx, Ry)
    # array of three directions
    xyz = np.array([Rx, Ry, Rz])
    # length of Rz vector
    rz = np.linalg.norm(Rz)
    # calculate the length of Rx and Ry vector if not provided
    if not (rx or ry):
        rx = geom.rij(ao, ax)
        ry = geom.rij(ao, ay)
    # do linear transformation
    xyz_new = np.array([[rx * bond_length, 0, 0], [0, ry * bond_length, 0], [0, 0, rz]])
    trans_matrix = np.dot(np.linalg.inv(xyz), xyz_new)
    coords = geom.xyz
    coords_new = np.dot(coords, trans_matrix)
    geom.xyz = coords_new

    if plot_geom:
        plot(geom, atom_indices=True)
        plt.axis("equal")


def set_cell(geom: Geometry, a, b=15, c=15):
    """
    set the length of the unit cell vector along the ribbon direction
    """
    if isinstance(a, (list, tuple, np.ndarray)):
        geom.cell[0, :] = a
    elif isinstance(a, (int, float)):
        geom.cell[0, :] = [a, 0, 0]
    geom.cell[1, :] = [0, b, 0]
    geom.cell[2, :] = [0, 0, c]
    geom.set_nsc([3, 1, 1])


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
    name, path=None, cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]], plot_geom=True
) -> Geometry:
    """
    Read coordinates from .xyz file, move the geometry center to origin
    then create a geometry object, the initial lattice vectors are
    [[10,0,0],[0,10,0],[0,0,10]]
    User set_cell method later to correct it.
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
            atoms.append(Atom(str(line.pop(0))))
            coordinates.append([float(i) for i in line])
    coordinates = np.array(coordinates)
    coordinates = coordinates - np.mean(coordinates, 0)

    geom = Geometry(coordinates, atoms, cell)
    if plot_geom:
        plot(geom, atom_indices=True)
        plt.axis("equal")

    return geom


def move_to_origo(gnr):
    """
    Move the geometry center to origin
    """

    gnr = gnr.translate([-gnr.center()[0], -gnr.center()[1], 0])
    plot(gnr)
    plt.axis("equal")

    return gnr


def move_to_center(g, axis="xyz", plot_geom=True):
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
        plot(g)
        plt.axis("equal")

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


def display2D(g, aid=False, sc=True, rotate=False, figsize=(10, 5), **kwargs):

    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    print("minxyz:", minxyz)
    maxxyz = np.amax(xyz, 0)
    print("maxxyz:", maxxyz)
    length = maxxyz - minxyz
    print("size:", length)

    plt.figure(figsize=figsize)
    if rotate:
        if round(g.cell[0, 1], 5) != 0:
            angle = np.degrees(np.arctan(g.cell[0, 1] / g.cell[0, 0]))
            g = g.rotate(-angle, [0, 0, 1])
    plot(g, atom_indices=aid, supercell=sc, **kwargs)
    plt.axis("equal")

    if length[0] > length[1]:  # x is larger
        plt.xlim(minxyz[0] - 6, maxxyz[0] + 6)
    else:
        plt.ylim(minxyz[1] - 2, maxxyz[1] + 2)


def SetView(xyzview, rotation, zoom):
    xyzview.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3},
                      'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})
    xyzview.rotate(rotation)
    xyzview.zoomTo()
    xyzview.zoom(zoom)
    xyzview.show()



def display3D(what, width=500, height=300, rotation=0, zoom=1):
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
    SetView(xyzview, rotation, zoom)



def slice_show(g, xlim=[0, 10], ylim=None, figsize=(8, 5)):
    display2D(g, aid=False, sc=False, figsize=figsize)
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
