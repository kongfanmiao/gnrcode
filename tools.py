from sisl import *
import time
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
    t = (0., -2.7)
    H.construct([r, t])
    print(H)
    return H


def get_atom_list(geom):
    aidx_dict = {}
    for ia, a, isp in geom.iter_species():
        if a.symbol not in aidx_dict.keys():
            aidx_dict[a.symbol] = []
        aidx_dict[a.symbol].append(geom.a2o(ia))
    return aidx_dict


def get_orb_list(geom):
    aidx_dict = get_atom_list(geom)
    oidx_dict = {}
    idx_dict = {}
    orb_list = ['s', 'pxy', 'pz', 'd', 'f']
    for atom in geom.atoms.atom:
        a = atom.symbol
        oidx_dict[a] = dict(zip(orb_list, [[], [], [], [], []]))
        idx_dict[a] = dict(zip(orb_list, [[], [], [], [], []]))
        for i, orb in enumerate(atom):
            if orb.l == 0:
                oidx_dict[a]['s'].append(i)
            elif orb.l == 1 and (orb.m in [-1, 1]):
                oidx_dict[a]['pxy'].append(i)
            elif orb.l == 1 and orb.m == 0:
                oidx_dict[a]['pz'].append(i)
            elif orb.l == 2:
                oidx_dict[a]['d'].append(i)
            elif orb.l == 3:
                oidx_dict[a]['f'].append(i)
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
    sl = s.split(';')
    d = {}
    for i in sl:
        k, v = i.split(':')
        key, v = k.strip(), v.strip()
        v = v.split(',')
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
        lnew = '[' + ','.join((f"{l[0]},{l[1]}", '...',
                               f"{l[-2]},{l[-1]}")) + ']'

    return lnew