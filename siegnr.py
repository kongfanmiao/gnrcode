from sisl import *
from sisl import Geometry, Atom
import os
import numpy as np
import matplotlib.pyplot as plt


def create_geometry(name, path=None,
                    cell=[[10,0,0],[0,10,0],[0,0,10]],
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
    
    coordinates=[]
    raw_atom=[]
    atoms=[]
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



def adjust_axes(geom: Geometry, a0:int, ax:int, ay:int, plot_geom=True):
    """
    Put the geometry in xy plane and align the one dimensional ribbon along x axis.
    """
    # give three atom indices and define the x, y axis
    if not isinstance(geom, Geometry):
        raise TypeError('Please give a Geometry object as input')
    Rx = geom.Rij(a0, ax)
    Ry = geom.Rij(a0, ay)
    Rz = np.cross(Rx, Ry)
    xyz = np.array([Rx, Ry, Rz])
    #if not (1.2 < geom.rij(a0, ax) < 1.7):
    #    raise ValueError('Please choose two nearest atoms as the x-axis')
    #if not (1.2*np.sqrt(3) < geom.rij(a0, ay) < 1.7*np.sqrt(3)):
    #    raise ValueError('Please choose next nearest atoms as the y axis')
    #rx = geom.rij(a0, ax)
    rx = 1.42
    #ry = geom.rij(a0, ay)
    ry = 1.42*np.sqrt(3)
    rz = np.linalg.norm(Rz)
    xyz_new = np.array([[rx, 0, 0],[0, ry, 0], [0, 0, rz]])
    trans_matrix = np.dot(np.linalg.inv(xyz), xyz_new)
    
    coords = geom.xyz
    coords_new = np.dot(coords, trans_matrix)
    geom.xyz = coords_new
    
    if plot_geom:
        plot(geom, atom_indices=True)
        plt.axis('equal')



def set_cell(geom: Geometry, length):
    """
    set the length of the unit cell vector along the ribbon direction 
    """   
    
    geom.cell[0,0] = length
    geom.cell[1,1], geom.cell[2,2] = 30, 30
    geom.set_nsc([3,1,1])



def write_coord(geom: Geometry, name, path=None):
    # write the standardised coordinates to new xys file    
    
    file_name =  name + "_st.xyz"
    if not path:
        file_path = './files/' + file_name
    else:
        file_path = os.path.join(path, file_name)
    coords = geom.xyz
    with open(file_path, "w") as fout:
        fout.write(str(len(coords)) + "\n\n")
        for i in range(len(coords)):
            fout.write("{}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(
                geom.atoms[i].symbol,*geom.xyz[i]))




