from ase.data import atomic_numbers
from ase.data import atomic_masses_iupac2016 as atomic_masses
from itertools import combinations, permutations
from sisl import *
import os
import time
import re
import regex
import datetime
import functools
import numpy as np
import matplotlib.pyplot as plt
import warnings
from glob import glob


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and may be removed in future versions.",
            category=DeprecationWarning,
            stacklevel=3,
        )
        return func(*args, **kwargs)
    return wrapper


def timer(func):
    """
    Print the elapsed time of running the decorated function.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Running {func.__name__} takes {run_time:.4f} seconds")
        return value

    return wrapper_timer


def read_geom_and_ham(name, path):
    """
    Read geometry and Hamiltonian from Siesta output files
    """
    runFile = os.path.join(path, f'{name}_RUN.fdf')
    xvFile = os.path.join(path, f'{name}.XV')
    H = get_sile(runFile).read_hamiltonian()
    g = get_sile(xvFile).read_geometry()
    H.geometry.cell[:] = g.cell[:]
    H.geometry.xyz[:] = g.xyz[:]
    g = H.geometry
    return g, H


# cast the hamiltonian copy into a function
def copy_hamiltonian(H, shape=3):

    if shape == 3:
        a, b, c = H.shape
        h = np.empty([a, b, c])
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    h[i, j, k] = H[i, j, k]
    elif shape == 2:
        a, b = H.shape
        h = np.empty([a, b])
        for i in range(a):
            for j in range(b):
                h[i, j] = H[i, j]
    return h


def construct_hamiltonian(gnr, spin=None):

    H = Hamiltonian(gnr)
    if spin:
        H = Hamiltonian(gnr, spin=spin)
    r = (0.1, 1.44)
    t = (0.0, -2.7)
    H.construct([r, t])
    # print(H)
    return H


def get_orb_list(geom):
    """
    Get the indices of all the orbitals for each atom species
    """
    # The index of first orbitals for each atoms, classifed based on atomic species
    aidx_dict = {}
    for ia, a, isp in geom.iter_species():
        if a.symbol not in aidx_dict.keys():
            aidx_dict[a.symbol] = []
        aidx_dict[a.symbol].append(geom.a2o(ia))
    oidx_dict = {}
    idx_dict = {}
    orb_list = ["s", "pxy", "pz", "d", "f"]
    for atom in geom.atoms.atom:
        a = atom.symbol
        oidx_dict[a] = dict(zip(orb_list, [[], [], [], [], []]))
        idx_dict[a] = dict(zip(orb_list, [[], [], [], [], []]))
        for i, orb in enumerate(atom):
            if orb.l == 0:
                oidx_dict[a]["s"].append(i)
            elif orb.l == 1 and (orb.m in [-1, 1]):
                oidx_dict[a]["pxy"].append(i)
            elif orb.l == 1 and orb.m == 0:
                oidx_dict[a]["pz"].append(i)
            elif orb.l == 2:
                oidx_dict[a]["d"].append(i)
            elif orb.l == 3:
                oidx_dict[a]["f"].append(i)
        for orb in orb_list:
            all_idx = np.add.outer(aidx_dict[a], oidx_dict[a][orb]).ravel()
            idx_dict[a][orb] = all_idx
    return idx_dict


def all_pz(H):
    C_atoms = []
    H_atoms = []
    for i, at in enumerate(H.atoms):
        if at.Z == 6:
            C_atoms.append(i)
        elif at.Z == 1:
            H_atoms.append(i)

    idx_pz = []
    for i, orb in enumerate(H.geometry.atoms[C_atoms[0]]):
        if orb.l == 1 and orb.m == 0:
            idx_pz.append(i)

    all_pz = np.add.outer(H.geometry.a2o(C_atoms), idx_pz).ravel()
    return all_pz


def convert_formated_str_to_dict(s: str):
    """
    Convert the string of the following format to dictionary:
    Eg:
        Input: 'C: pz,pz; N: pxy'
        Output: {'C': ['pxy','pz'],
                 'N': ['pxy']}
    """
    sl = s.split(";")
    d = {}
    for i in sl:
        k, v = i.split(":")
        key, v = k.strip(), v.strip()
        v = v.split(",")
        value = [i.strip() for i in v]
        d[key] = value
    return d



def convert_dict_to_formatted_str(d: dict) -> str:
    """
    Convert the dictionary to a string in the following format:
    Eg:
        Input: {'C': ['pxy','pz'],
                'N': ['pxy']}
        Output: 'C: pxy,pz; N: pxy'
    """
    formatted_str_list = []

    for key, values in d.items():
        values_str = ','.join((value.strip() for value in values))
        formatted_str_list.append(f"{key}:{values_str}")

    formatted_str = ';'.join(formatted_str_list)

    return formatted_str



def list_str(l):
    """
    Convert the list to a string and shorten it if it's too long
    """
    if len(l) <= 4:
        lnew = str(l)
    else:
        lnew = "[" + ",".join((f"{l[0]},{l[1]}", "...",
                              f"{l[-2]},{l[-1]}")) + "]"

    return lnew


def read_forces(name, path, which=None):
    """
    Read max force from .FA file
    ouput max force, total force, and residual force in eV/Ang"""

    fa = os.path.join(path, f'{name}.FA')
    forces = np.loadtxt(fa, skiprows=1)[:, 1:]
    maxForce = np.max(np.abs(forces))
    totForce = np.linalg.norm(np.sum(forces, axis=0))
    resForce = np.sqrt(np.sum(np.square(forces))/forces.size)  # residual
    forceDict = {'max': maxForce,
                 'total': totForce,
                 'residue': resForce}
    if not which:
        print('Max force: {:6f} eV/Ang'.format(maxForce))
        print('Total force: {:6f} eV/Ang'.format(totForce))
        print('Residue force: {:6f} eV/Ang'.format(resForce))
    elif which.lower() == 'all':
        return maxForce, totForce, resForce
    else:
        which = which.split(",")
        retlist = []
        for s in which:
            try:
                retlist.append(forceDict[s.strip().lower()])
            except:
                print(
                    "Wrong keyword. Must be one of ['max', 'total', 'residue']")
        if len(retlist) == 1:
            retlist = retlist[0]
        return retlist


def read_calc_time(name, path, which=None):

    # There is only .times file for parallel run. Try to read .times file first
    try:
        ftime = os.path.join(path, f'{name}.times')
        with open(ftime) as f:
            lines = f.read()
        numCores = int(
            re.search('Number of nodes\s*=\s*(\d+)', lines).group(1))
        wallTime = float(
            re.search('Total elapsed wall-clock time.+=\s+(.+)', lines).group(1))
        CPUtime = float(
            re.search('Tot:  Sum, Avge,.+=([\d\.]+)\s+', lines).group(1))
    except:
        fout = glob(os.path.join(path, f'{name}*.out'))[0]
        with open(fout) as f:
            lines = f.read()
        tStart = re.search(
            'Start of run:\s*(\d+\-\w+\-\d{4}\s*\d+:\d+:\d+)\n', lines).group(1)
        tEnd = re.search(
            'End of run:\s*(\d+\-\w+\-\d{4}\s*\d+:\d+:\d+)\n', lines).group(1)
        tStart = datetime.datetime.strptime(tStart, '%d-%b-%Y %H:%M:%S')
        tEnd = datetime.datetime.strptime(tEnd, '%d-%b-%Y %H:%M:%S')
        wallTime = (tEnd - tStart).total_seconds()
        CPUtime = wallTime
        numCores = re.search(
            'Running on\s*(\d+)\s+nodes in parallel', lines).group(1)
    if not which:
        wallTime = str(datetime.timedelta(seconds=wallTime))
        CPUtime = str(datetime.timedelta(seconds=CPUtime))
        print(f"Number of cores (processors): {numCores}")
        print(f"Total elapsed wall-clock time: {wallTime}")
        print(f"Total CPU time: {CPUtime}")
    elif which.lower() == 'cores':
        return numCores
    elif which.lower() == 'walltime':
        return wallTime
    elif which.lower() == 'cputime':
        return CPUtime
    elif which.lower() == 'all':
        return numCores, wallTime, CPUtime
    else:
        raise ValueError("which must be None, or one of ['cores', 'walltime',\
            'cputime', 'all']")


def read_final_energy(name, path="./opt", which=None):
    outfile = name + ".out"
    filepath = os.path.join(path, outfile)
    with open(filepath) as fout:
        lines = fout.read()
    ergStr = regex.search('(?<=siesta: Final energy \(eV\):\n)(.+=.+\n)+',
                          lines).group(0)
    rawList = regex.findall('siesta: .+=.+', ergStr)
    ergDict = dict()
    for s in rawList:
        [key, value] = s[7:].split('=')
        key = key.strip().lower()
        value = float(value)
        ergDict[key] = value
    if which:
        which = which.split(",")
        retlist = []
        for s in which:
            try:
                retlist.append(ergDict[s.strip().lower()])
            except:
                print("Wrong keyword. Must be one of {}".format([
                    i for i in ergDict.keys()]))
        if len(retlist) == 1:
            retlist = retlist[0]
        return retlist
    else:
        print("Total energy: {} eV".format(ergDict["total"]))
        print("Fermi energy: {} eV".format(ergDict["fermi"]))


def read_bond_length(name, path):
    "Read C-C and C-H bond length"
    g = get_sile(os.path.join(path, f'{name}.XV')).read_geometry()

    C_list = []
    H_list = []
    for i, a, _ in g.iter_species():
        if a.Z == 6:
            C_list.append(i)
        elif a.Z == 1:
            H_list.append(i)

    CC_bonds = []
    CH_bonds = []
    for cc in combinations(C_list, 2):
        r = g.rij(*cc)
        if r < 1.9:
            CC_bonds.append(r)
    CC_bond = sum(CC_bonds)/len(CC_bonds)
    for c in C_list:
        for h in H_list:
            r = g.rij(c, h)
            if r < 1.2:
                CH_bonds.append(r)
    CH_bond = sum(CH_bonds)/len(CH_bonds)
    return CC_bond, CH_bond


def read_interpolated_ham(name, path):
    """
    Read interpolated Hamiltonian from Wannier90 calculation
    """
    win = get_sile(os.path.join(path, f'{name}.win'))
    Hint = win.read_hamiltonian()
    return Hint


def modify_wann_xsf_file(filename, translation_vector):
    """
    Modify the XSF file generated by Wannier90, translate the datagrid
    to the home geometry.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    grid_dimensions = []
    grid_origin = []
    spanning_vectors = []
    grid_data = []
    lines_before_datagrid_3d = []

    datagrid_3d_started = False

    for i, line in enumerate(lines):
        if "BEGIN_DATAGRID_3D" in line:
            datagrid_3d_started = True
        elif not datagrid_3d_started:
            lines_before_datagrid_3d.append(line)
        else:
            if len(grid_dimensions) == 0 and len(line.split()) == 3:
                grid_dimensions = [int(x) for x in line.split()]
            elif len(grid_origin) == 0 and len(line.split()) == 3:
                grid_origin = [float(x) for x in line.split()]
            elif len(spanning_vectors) < 3 and len(line.split()) == 3:
                spanning_vectors.append([float(x) for x in line.split()])
            else:
                try:
                    float_values = [float(x) for x in line.split()]
                    grid_data.extend(float_values)
                except ValueError:
                    pass

    grid_origin = np.array(grid_origin)
    grid_origin = grid_origin + translation_vector

    with open(filename, 'w') as file:
        # Write lines before BEGIN_DATAGRID_3D
        for line in lines_before_datagrid_3d:
            file.write(line)

        # Write BEGIN_DATAGRID_3D
        file.write("BEGIN_DATAGRID_3D\n")

        # Write grid dimensions
        file.write("\t"+"\t ".join(str(x) for x in grid_dimensions) + "\n")

        # Write grid origin
        file.write("\t"+"\t ".join("{:.6f}".format(x)
                   for x in grid_origin) + "\n")

        # Write spanning vectors
        for vector in spanning_vectors:
            file.write("\t"+"\t ".join("{:.6f}".format(x)
                       for x in vector) + "\n")

    # Write grid data
        for i, value in enumerate(grid_data):
            file.write("{:.6E} ".format(value))
            if (i + 1) % grid_dimensions[0] == 0:
                file.write("\n")

        # Write END_DATAGRID_3D
        file.write("END_DATAGRID_3D\n")
        file.write("END_BLOCK_DATAGRID_3D\n")


def modify_wann_centres_file(path, n, translation_vector):
    """ 
    Modify the centres file generated by Wannier90, translate by vector
    Argument:
        n: the n-th centre
        translation_vector: a list of three floats"""

    # parent_directory = os.path.dirname(filename)
    centres_file = glob(f'{path}/*_centres.xyz')[0]
    # pattern = r'_(\d+)\.xsf'
    # match = re.search(pattern, filename)
    # n = int(match.group(1))

    # Read the file and store the lines in a list
    with open(centres_file, "r") as file:
        lines = file.readlines()

    # Update the n-th line starting with "X"
    x_count = 0
    for i, line in enumerate(lines):
        if line.startswith("X"):
            x_count += 1
            if x_count == n:
                parts = line.split()
                coords = [float(x) for x in parts[1:]]
                updated_coords = [coords[j] + translation_vector[j]
                                  for j in range(3)]
                lines[i] = "{}\t\t\t{}\n".format(parts[0], '\t\t'.join(
                    f'{coord:.8f}' for coord in updated_coords))
                break

    # Write the updated lines back to the file
    with open(centres_file, "w") as file:
        file.writelines(lines)


def read_orca_final_energy(name, path):
    last_energy = None

    with open(os.path.join(path, f'{name}.out'), 'r') as file:
        for line in file:
            if "FINAL SINGLE POINT ENERGY" in line:
                # Extract the float value
                last_energy = line.split()[-1]

    last_energy = float(last_energy.split()[-1])
    last_energy *= 27.211386245988  # convert from Hartree to eV
    if last_energy is not None:
        return last_energy
    else:
        print("No 'FINAL SINGLE POINT ENERGY' line found in the file.")


def get_masses(elements):
    return np.array([atomic_masses[atomic_numbers[e]] for e in elements])
