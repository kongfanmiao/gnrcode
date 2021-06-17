from sisl import *
from sisl import Geometry, Atom
import os
import numpy as np
import matplotlib.pyplot as plt


def create_geometry(name, path=None,
                    cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                    plot_geom=True) -> Geometry:
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
    with open(file_path, 'r') as fin:
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
        plt.axis('equal')

    return geom


def adjust_axes(geom: Geometry, ao: int, ax: int, ay: int, rx, ry, bond_length=1.42,
                plot_geom=True):
    """
    Put the geometry in xy plane and align the one dimensional ribbon along x axis.
    """
    # give three atom indices and define the x, y axis
    if not isinstance(geom, Geometry):
        raise TypeError('Please give a Geometry object as input')
    Rx = geom.Rij(ao, ax)
    Ry = geom.Rij(ao, ay)
    Rz = np.cross(Rx, Ry)
    xyz = np.array([Rx, Ry, Rz])
    rz = np.linalg.norm(Rz)
    xyz_new = np.array(
        [[rx*bond_length, 0, 0], [0, ry*bond_length, 0], [0, 0, rz]])
    trans_matrix = np.dot(np.linalg.inv(xyz), xyz_new)

    coords = geom.xyz
    coords_new = np.dot(coords, trans_matrix)
    geom.xyz = coords_new

    if plot_geom:
        plot(geom, atom_indices=True)
        plt.axis('equal')


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
    # write the standardised coordinates to new xyz file

    file_name = name + "_st.xyz"
    if not path:
        file_path = './files/' + file_name
    else:
        file_path = os.path.join(path, file_name)
    coords = geom.xyz
    with open(file_path, "w") as fout:
        fout.write(str(len(coords)) + "\n\n")
        for i in range(len(coords)):
            fout.write("{}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(
                geom.atoms[i].symbol, *geom.xyz[i]))
