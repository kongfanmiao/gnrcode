from sisl import *
import os
import time
import re, regex
import glob, datetime
import functools
import numpy as np
import matplotlib.pyplot as plt


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


def construct_hamiltonian(gnr):

    H = Hamiltonian(gnr)
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
        Input: 'C: pz; N: pxy'
        Output: {'C': ['pz'],
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


def list_str(l):
    """
    Convert the list to a string and shorten it if it's too long
    """
    if len(l) <= 4:
        lnew = str(l)
    else:
        lnew = "[" + ",".join((f"{l[0]},{l[1]}", "...", f"{l[-2]},{l[-1]}")) + "]"

    return lnew


def read_forces(name,path):
    """
    Read max force from .FA file
    ouput max force, total force, and residual force in eV/Ang"""

    fa = os.path.join(path,f'{name}.FA')
    forces = np.loadtxt(fa, skiprows=1)[:,1:]
    maxForce = np.max(np.abs(forces))
    totForce = np.linalg.norm(np.sum(forces, axis=0))
    resForce = np.sqrt(np.sum(np.square(forces))/forces.size) # residual

    return maxForce, totForce, resForce


def read_calc_time(name, path, which='all'):

    # There is only .times file for parallel run. Try to read .times file first
    try:
        ftime = os.path.join(path, f'{name}.times')
        with open(ftime) as f:
            lines = f.read()
        numNodes = int(re.search('Number of nodes\s*=\s*(\d+)', lines).group(1))
        wallTime = float(re.search('Total elapsed wall-clock time.+=\s+(.+)', lines).group(1))
        CPUtime = float(re.search('Tot:  Sum, Avge,.+=\s+([\d\.]+)\s+', lines).group(1))
    except:
        fout = glob(os.path.join(path,'*.out'))[0]
        with open(fout) as f:
            lines = f.read()
        tStart = re.search('Start of run:\s*(\d+\-\w+\-\d{4}\s*\d+:\d+:\d+)\n', lines).group(1)
        tEnd = re.search('End of run:\s*(\d+\-\w+\-\d{4}\s*\d+:\d+:\d+)\n', lines).group(1)
        tStart = datetime.strptime(tStart, '%d-%b-%Y %H:%M:%S')
        tEnd = datetime.strptime(tEnd, '%d-%b-%Y %H:%M:%S')
        wallTime = (tEnd - tStart).total_seconds()
        CPUtime = wallTime
        numNodes = 1
    if which.lower()=='nodes':
        return numNodes
    elif which.lower()=='walltime':
        return wallTime
    elif which.lower()=='cputime':
        return CPUtime
    elif which.lower() == 'all':
        return numNodes, wallTime, CPUtime
    else:
        wallTime = time.strftime('%H:%M:%S', time.gmtime(wallTime))
        CPUtime = time.strftime('%H:%M:%S', time.gmtime(CPUtime))
        print(f"Number of nodes (processors): {numNodes}")
        print(f"Total elapsed wall-clock time: {wallTime}")
        print(f"Total CPU time: {CPUtime}")



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
        key = key.strip()
        value = float(value)
        ergDict[key] = value
    if which:
        which = which.split(",")
        retlist = []
        for s in which:
            for key in ergDict.keys():
                if key.lower() == s.strip().lower():
                    retlist.append(ergDict[key])
        if len(retlist) == 1:
            retlist = retlist[0]
        return retlist
    else:
        print("Total energy: {} eV".format(ergDict["Total"]))
        print("Fermi energy: {} eV".format(ergDict["Fermi"]))


from itertools import combinations, permutations

def read_bond_length(name, path):

    "Read C-C and C-H bond length"
    g = get_sile(os.path.join(path, f'{name}.XV')).read_geometry()

    C_list = []
    H_list = []
    for i,a,_ in g.iter_species():
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
            r = g.rij(c,h)
            if r < 1.2:
                CH_bonds.append(r)
    CH_bond = sum(CH_bonds)/len(CH_bonds)
    return CC_bond, CH_bond