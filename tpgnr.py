from sisl import *
import numpy as np
import matplotlib.pyplot as plt
import os

import time
import functools


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


def move_to_origo(gnr):
    """
    Move the geometry center to origin
    """

    gnr = gnr.translate([-gnr.center()[0], -gnr.center()[1], 0])
    plot(gnr)
    plt.axis('equal')

    return gnr


def move_to_center(g, axis='xyz', plot_geom=True):
    """
    Move the geometry to the center of supercell
    """
    cell_center = g.center(what='cell')
    # The internal method of xyz center is stupid, don't use it
    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    maxxyz = np.amax(xyz, 0)
    geom_center = (minxyz + maxxyz) / 2
    xvector = cell_center[0]-geom_center[0] if 'x' in axis else 0
    yvector = cell_center[1]-geom_center[1] if 'y' in axis else 0
    zvector = cell_center[2]-geom_center[2] if 'z' in axis else 0
    g = g.translate([xvector, yvector, zvector])

    if plot_geom:
        plot(g)
        plt.axis('equal')

    return g


def move_to_xcenter(g, plot_geom=True):
    """
    Move the geometry to the centre in x direction
    """
    gx = move_to_center(g, 'x', plot_geom=plot_geom)
    return gx


def move_to_ycenter(g, plot_geom=True):
    """
    Move the geometry to the centre in y direction
    """
    gy = move_to_center(g, 'y', plot_geom=plot_geom)
    return gy


def move_to_xycenter(g, plot_geom=True):
    """
    Move the geometry to the centre in x and y direction
    """
    gxy = move_to_center(g, 'xy', plot_geom=plot_geom)
    return gxy


def move_to_zcenter(g, plot_geom=True):
    """
    Move the geometry to the centre in z direction
    """
    gz = move_to_center(g, 'z', plot_geom=plot_geom)
    return gz


def display(g, aid=False, sc=True, rotate=False, figsize=(10, 5), **kwargs):

    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    print('minxyz:', minxyz)
    maxxyz = np.amax(xyz, 0)
    print('maxxyz:', maxxyz)
    length = maxxyz - minxyz
    print('size:', length)

    plt.figure(figsize=figsize)
    if rotate:
        if round(g.cell[0, 1], 5) != 0:
            angle = np.degrees(np.arctan(g.cell[0, 1] / g.cell[0, 0]))
            g = g.rotate(-angle, [0, 0, 1])
    plot(g, atom_indices=aid, supercell=sc, **kwargs)
    plt.axis('equal')

    if length[0] > length[1]:  # x is larger
        plt.xlim(minxyz[0]-6, maxxyz[0]+6)
    else:
        plt.ylim(minxyz[1]-2, maxxyz[1]+2)


def slice_show(g, xlim=[0, 10], figsize=(8, 5)):
    display(g, aid=False, sc=False, figsize=figsize)
    plt.xlim(*xlim)
    for i in g:
        if xlim[0] < g.xyz[i, 0] < xlim[1]:
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


def attach_pz(g, basis='DZP'):
    """
    Attach the Pz orbitals to the geometry
    """

    r = np.linspace(0, 5, 100)
    epsilon = 1.625
    f = np.exp(-epsilon*r)
    # Normalize
    A = 1/(f.sum() * (r[1] - r[0]))
    f = A * f

    if basis == 'DZP':
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=2, P=1)
    elif basis == 'DZ':
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=2)
    elif basis == 'SZP':
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=1, P=1)
    elif basis == 'SZ':
        pz = AtomicOrbital((r, f), n=2, l=1, m=0, Z=1)
    C = Atom(6, pz)
    g.atoms.replace(g.atoms[0], C)


def construct_hamiltonian(gnr):

    H = Hamiltonian(gnr)
    r = (0.1, 1.44)
    t = (0., -2.7)
    H.construct([r, t])
    print(H)
    return H


def find_fermi_energy(name, path):
    """
    Read Fermi energy from siesta .out file
    If the Hamiltonian is read from win file but not fdf file,
    The Fermi energy won't be shifted automatically.
    """
    file_path = os.path.join(path, name+'.out')
    with open(file_path) as fout:
        for line in fout:
            if "Fermi energy" in line:
                fe_str = line.strip().split()[-2]
                fe = float(fe_str)
    # print(f"Fermi energy is: {fe_str} eV")
    return fe


@timer
def band_structure(H, Erange=[-3, 3], index=False, figsize=(6, 4), tick_labels='XGX', **kwargs):

    labels_dict = {
        'G': ('$\Gamma$', [0, 0, 0]),
        'X': ('$X$', [0.5, 0, 0]),  # always put periodic direction in x
        'M': ('$M$', [0.5, 0.5, 0]),
        'K': ('$K$', [2./3, 1./3, 0])
    }
    tkls = list(tick_labels)
    tks = []
    # TO DO this is problematic for time reversal symmetry broken system
    for i, v in enumerate(tick_labels):
        tkls[i] = labels_dict[v][0]
        tks.append(labels_dict[v][1])

    bs = BandStructure(H, tks, 400, tkls)
    bsar = bs.apply.array
    eigh = bsar.eigh()

    lk, kt, kl = bs.lineark(True)

    plt.figure(figsize=figsize)
    plt.xticks(kt, kl)
    plt.ylim(Erange[0], Erange[-1])
    plt.xlim(0, lk[-1])
    plt.ylabel('$E-E_F$ (eV)')

    for i, ek in enumerate(eigh.T):
        plt.plot(lk, ek, **kwargs)
        if index:
            if Erange[0] < ek[-1] < Erange[1]:
                plt.annotate(i+1, (lk[-1], ek[0]))



@timer
def interpolated_bs(name, path, Hint=None, Hpr=None, Erange=[-5, 0], figsize=(6, 4), tick_labels='XGX', overlap=False, npts=30, marker_size=30, marker_color='g', marker='o', facecolors='none', **kwargs):
    """
    PLot the interpolated band structure from Wannier90 output file
    Arguments:
        Hint: interpolated Hamiltonian
        Hpr: pristine Hamiltonian
    """
    if Hint:
        ham_int = Hint
    else:
        win_path = os.path.join(path, name+'.win')
        fwin = get_sile(win_path)
        ham_int = fwin.read_hamiltonian()
 
    labels_dict = {
        'G': ('$\Gamma$', [0, 0, 0]),
        'X': ('$X$', [0.5, 0, 0]),  # always put periodic direction in x
        'M': ('$M$', [0.5, 0.5, 0]),
        'K': ('$K$', [2./3, 1./3, 0])
    }
    tkls = list(tick_labels)
    tks = []
    # TO DO this is problematic for time reversal symmetry broken system
    for i, v in enumerate(tick_labels):
        tkls[i] = labels_dict[v][0]
        tks.append(labels_dict[v][1])
    
    if not overlap:
        knpts_int = 400
    else:
        knpts_int = npts
        knpts_pr = 400
        if not Hpr:
            raise ValueError("Please provide the pristine Hamiltonian overlap is True")
        bs_pr = BandStructure(Hpr, tks, knpts_pr, tkls)
        bsar_pr = bs_pr.apply.array
        eigh_pr = bsar_pr.eigh()
        lk_pr = bs_pr.lineark(ticks=False)

    bs_int = BandStructure(ham_int, tks, knpts_int, tkls)
    bsar_int = bs_int.apply.array
    eigh_int = bsar_int.eigh()
    fe = find_fermi_energy(name=name, path=path)
    eigh_int -= fe

    lk_int, kt, kl = bs_int.lineark(ticks=True)
    plt.figure(figsize=figsize)
    plt.xticks(kt, kl)
    plt.ylim(Erange[0], Erange[-1])
    plt.ylabel('$E-E_F$ (eV)')

    for i, ek in enumerate(eigh_int.T):
        plt.plot(lk_int, ek, **kwargs)

    if overlap:
        for i, ek_pr in enumerate(eigh_pr.T):
            plt.plot(lk_pr, ek_pr, color='k', **kwargs)
        for j, ek_int in enumerate(eigh_int.T):
            plt.scatter(lk_int, ek_int, s=marker_size, marker=marker, facecolors=facecolors, edgecolors='g')
        plt.xlim(0, lk_pr[-1])
    else:
        for i, ek_int in enumerate(eigh_int.T):
            plt.plot(lk_int, ek_int,**kwargs)
        plt.xlim(0, lk_int[-1])
    



@timer
def unfold_band(H, lat_vec, kmesh=500, ky=0, marker_size: float = None,
                marker_size_range=[2, 10], cmap='Reds'):
    """
    Unfold the bandstructure
    Only applies to 1-D structure at the moment
    Arguments:
        lat_vec: lattive vector of the primitive cell
        ky: perpendicular component of the wavenumber to the periodic direction
        kmesh: number of supercell
        marker_size: if not provide, then marker size use normalized weight, if provide,
            then use the provided value.
        cmap: color map
    """
    N = len(H)
    xyz = H.geometry.xyz
    # band lines scale
    bdlscale = np.pi/lat_vec
    G = H.rcell[0, 0]
    for red_k in np.linspace(-1, 1, kmesh):
        k = red_k*bdlscale
        k_vec = np.array([k, ky, 0])
        red_K = k/G
        if red_K > 0:
            red_K = red_K - np.floor(red_K)
        else:
            red_K = red_K - np.ceil(red_K)
        red_K_vec = np.array([red_K, 0, 0])
        # k vectors use reduced one here
        eigh = H.eigh(k=red_K_vec)
        eigenstate = H.eigenstate(k=red_K_vec).state
        phase_left = np.exp(xyz.dot(k_vec)*1j)
        phase_right = np.exp(xyz.dot(k_vec)*(-1j))
        phase = np.outer(phase_left, phase_right)
        weight = (1/N)*np.conj(eigenstate).dot(phase).dot(eigenstate.T)
        weight = np.abs(weight).diagonal()
        weight = weight/weight.max()
        weight = weight
        kpts = np.repeat(k, N)
        if not marker_size:
            smin, smax = marker_size_range
            msize = weight*(smax-smin)+smin
        else:
            msize = marker_size
        plt.scatter(kpts, eigh, s=msize,
                    c=weight, cmap=cmap)
        plt.ylabel('$E-E_F (eV)$')
        plt.xlabel('wavenumber ($1 /\AA$)')


@timer
def band_gap(H):

    bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
                       '$\Gamma$', '$X$', '$\Gamma$'])

    bsar = bs.apply.array
    eigh = bsar.eigh()
    bg = functools.reduce(lambda x, y: x if x <= y else y,
                          (ek[ek > 0].min() - ek[ek < 0].max() for ek in eigh))
    return bg


@timer
def energy_levels(H, Erange=[-5, 5], figsize=(1, 5), index=False,
                  color='darkslategrey', **kwargs):
    """
    should be a single molecule, nsc=[1,1,1]
    """
    eig = H.eigh()
    plt.figure(figsize=figsize)
    # use Hermitian solver, read values
    plt.hlines(eig, 0, 1, color=color, **kwargs)
    plt.ylim(Erange[0], Erange[-1])
    plt.ylabel('$E-E_F$ (eV)')
    plt.xticks([])

    if index:  # label the index of energy levels
        which = np.where(np.logical_and(eig < Erange[-1],
                                        eig > Erange[0]))[0]
        for i in which:
            plt.text(11, eig[i], str(i))


@timer
def dos(H, Erange=[-3, 3], figsize=(4, 6), ret=False, color='k', **kwargs):

    from functools import partial
    nsc = H.nsc
    if (nsc[0] > 1 & nsc[1] == 1 & nsc[2] == 1):
        bs = BandStructure(H, [[0, 0, 0], [1, 0, 0]],
                           400, ['$\Gamma$', '$\Gamma$'])
    elif (nsc[0] == 1 & nsc[1] > 1 & nsc[2] == 1):
        bs = BandStructure(H, [[0, 0, 0], [0, 1, 0]],
                           400, ['$\Gamma$', '$\Gamma$'])

    bsav = bs.apply.average
    dis = partial(gaussian, sigma=0.05)
    E = np.linspace(Erange[0], Erange[-1], 500)
    plt.figure(figsize=figsize)
    DOS = bsav.DOS(E, distribution=dis)
    plt.plot(DOS, E, color='k', **kwargs)
    plt.ylim(Erange[0], Erange[-1])
    plt.xlim(0, np.amax(DOS) + 2)
    plt.ylabel('$E-E_F$ (eV)')
    plt.xlabel('DOS')
    if ret:
        return DOS


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


@timer
def pdos(H, Erange=(-10, 10), figsize=(4, 6), Emesh=300,
         projected_orbitals=['s', 'pxy', 'pz']):
    """
    Projected density of states
    by default plot selected projected orbitals for all the atom species
    """
    geom = H.geometry
    bs = BandStructure(H, [[0, 0, 0], [1, 0, 0]],
                       400, ['$\Gamma$', '$\Gamma$'])
    bsav = bs.apply.average

    orb_idx_dict = get_orb_list(geom)
    pdos_dict = {}

    def wrap(PDOS):
        nonlocal pdos_dict
        i = 0
        for a, all_idx in orb_idx_dict.items():
            for orb in projected_orbitals:
                if orb == 'pxy':
                    label = f'{a}: $p_x+p_y$'
                elif orb == 'pz':
                    label = f'{a}: $p_z$'
                else:
                    label = f'{a}: ${orb}$'
                if all_idx[orb].size != 0:  # if it's not empty
                    pdos = PDOS[all_idx[orb], :].sum(0)
                    pdos_dict.update({i: [label, pdos]})
                    i += 1
        return np.stack([v[1] for v in pdos_dict.values()])

    E = np.linspace(Erange[0], Erange[-1], Emesh)
    pDOS = bsav.PDOS(E, wrap=wrap)
    plt.figure(figsize=figsize)
    for i in range(pDOS.shape[0]):
        plt.plot(pDOS[i, :], E, color=f'C{i}', label=pdos_dict[i][0])
    plt.ylim(E[0], E[-1])
    plt.xlim(0, None)
    plt.ylabel('$E - E_F$ [eV]')
    plt.xlabel('DOS [1/eV]')
    plt.legend(bbox_to_anchor=[1.1, 0.9])


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


@timer
def pzdos(H, Erange=[-10, 20], plot_pzdos=True):

    import numpy as np

    bs = BandStructure(H, [[0, 0, 0], [1, 0, 0]],
                       400, ['$\Gamma$', '$\Gamma$'])
    bsav = bs.apply.average

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

    def wrap(PDOS):
        pdos_pz = PDOS[all_pz, :]
        return pdos_pz
    Emin, Emax = Erange
    E = np.linspace(Emin, Emax, 100)
    pDOS = bsav.PDOS(E, wrap=wrap)
    if plot_pzdos:
        for i, label in enumerate(all_pz):
            plt.plot(E, pDOS[i, :], label=label)
        plt.xlim(E[0], E[-1])
        plt.ylim(0, None)
        plt.xlabel(r'$E - E_F$ [eV]')
        plt.ylabel(r'pz_DOS [1/eV]')
        plt.title("Project DOS on pz orbitals")
        plt.legend(loc="best", bbox_to_anchor=[1.4, 0.9])
    return pDOS


def pzweight(H, Erange=[-10, 0]):
    # plot the weight of pz orbitals in a specific energy range, by default the occupied band
    pzd = pzdos(H, Erange, plot_pzdos=False)
    pzwt = np.multiply(pzd, pzd).sum(-1)
    pzidx = all_pz(H)
    plt.bar(pzidx, pzwt, width=1)
    plt.ylim(0, 1)
    for i, idx in enumerate(list(pzidx)):
        plt.annotate(idx, (pzidx[i], pzwt[i]))


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


@timer
def fat_bands(H, Erange=(-20, 20), figsize=(10, 8),
              projected_atoms='all',
              projected_orbitals=['s', 'pxy', 'pz'],
              specify_atoms_and_orbitals=None,
              index=False,
              alpha=0.75):
    """
    Plot the fat bands, showing the weight of each kinds of orbital of every band.
    specify_atoms_and_orbitals should follow the following format:
        'C: pz; N: pxy'
    If the specify_atosm_and_orbitals argument is not None, then it will overwrite 
    the projected_atoms and projected_orbitals arguments.
    If not, the projected orbitals will be all the orbitals in projected_orbitals of
    each atoms in projected_atoms.
    """
    geom = H.geometry
    orb_idx_dict = get_orb_list(geom)
    # initialize the weight dictionary
    wt_dict = {}
    for a, orbs in orb_idx_dict.items():
        wt_dict[a] = {}
        for orb in orbs.keys():
            wt_dict[a][orb] = []

    # Generate the atoms and corresponding orbitals that you want to project on
    if not specify_atoms_and_orbitals:
        if projected_atoms == 'all':
            proj_atoms = list(orb_idx_dict.keys())
        elif isinstance(projected_atoms, (list, tuple)):
            proj_atoms = projected_atoms
        if projected_orbitals == 'all':
            proj_orbs = ['s', 'pxy', 'pz', 'd', 'f']
        elif isinstance(projected_orbitals, (list, tuple)):
            proj_orbs = projected_orbitals
        proj_ats_orbs = dict(zip(proj_atoms,
                                 [proj_orbs]*len(proj_atoms)))
    else:
        proj_ats_orbs = convert_formated_str_to_dict(
            specify_atoms_and_orbitals)
    print(proj_ats_orbs)

    def wrap_fat_bands(eigenstate):
        """
        <psi_{i,v}|S(k)|psi_i>
        return the eigenvalue for a specify eigenstat and calculate
        the weight for each orbitals.
        """
        nonlocal wt_dict
        norm2 = eigenstate.norm2(sum=False)
        for a, orbs in orb_idx_dict.items():
            for orb, indices in orbs.items():
                if len(indices) != 0:
                    wt_k = norm2[:, indices].sum(-1)
                    wt_dict[a][orb].append(wt_k)
        return eigenstate.eig

    bs = BandStructure(H, [[-0.5, 0, 0], [0, 0, 0], [0.5, 0, 0]],
                       400, ['$X$', '$\Gamma$', '$X$'])
    bsar = bs.apply.array
    eig = bsar.eigenstate(wrap=wrap_fat_bands).T
    # after transposition, k is as column, e is as row, consistent with eig
    for value in wt_dict.values():
        for k in value.keys():
            value[k] = np.array(value[k]).T
    linear_k, k_tick, k_label = bs.lineark(True)
    Emin, Emax = Erange
    dE = (Emax - Emin)/(figsize[1]*5)
    plt.figure(figsize=figsize)
    plt.ylabel('$E-E_F$ [eV]')
    plt.xlim(linear_k[0], linear_k[-1])
    plt.xticks(k_tick, k_label)
    plt.ylim(Emin, Emax)

    legend_dict = {}
    for i, e in enumerate(eig):
        if np.any(np.logical_and(e < Emax, e > Emin)):
            t = 0
            filled_range = np.array([e, e])
            plt.plot(linear_k, e, color='k')
            if index:
                plt.annotate(i+1, (linear_k[-1], e[-1]))
            # plt.fill_between(linear_k, e-dE, e+dE, color='k', alpha=0.4)

            # To ensure the color sequence are the same for the same geometry
            # no matter what projected atoms and orbitals you choose, always
            # iterate all the atoms and all the orbitals. Change their transparency
            # based on whether you want to see it or not
            for a, orbs in wt_dict.items():
                for orb, wt in orbs.items():
                    if wt.size != 0:
                        try:
                            # make sure the orbital of this atom is meant to
                            # be projected
                            assert orb in proj_ats_orbs[a]
                            # if yes, plot it, alpha is not zero
                            alp = alpha
                            # and define the legend patch
                            if orb == 'pxy':
                                label = f'{a}: $p_x+p_y$'
                            elif orb == 'pz':
                                label = f'{a}: $p_z$'
                            else:
                                label = f'{a}: ${orb}$'
                            legend_dict[t] = label
                        except:
                            # if not, it will be totally transparent
                            alp = 0
                        weight = np.abs(wt[i, :]*dE)
                        plt.fill_between(
                            linear_k, filled_range[0]-weight,
                            filled_range[0], color=f'C{t}', alpha=alp)
                        plt.fill_between(
                            linear_k, filled_range[1],
                            filled_range[1]+weight, color=f'C{t}', alpha=alp)
                        # update the already filled range
                        filled_range = filled_range + \
                            np.array([-weight, weight])
                        t += 1

    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=f'C{t}', label=label)
                       for t, label in legend_dict.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=[1.1, 0.9])


# @timer
# def fat_bands_pz(H, Erange=(-10, 0), index=False, figsize=(10, 8)):
#     """
#     Plot the fat bands, showing the weight of each kinds of orbital of every band.
#     """

#     C_atoms = []
#     H_atoms = []
#     for i, at in enumerate(H.atoms):
#         if at.Z == 6:
#             C_atoms.append(i)
#         elif at.Z == 1:
#             H_atoms.append(i)

#     idx_s = []
#     idx_pxy = []
#     idx_pz = []
#     idx_d = []
#     for i, orb in enumerate(H.geometry.atoms[C_atoms[0]]):
#         if orb.l == 0:
#             idx_s.append(i)
#         elif orb.l == 1 and (orb.m in [-1, 1]):
#             idx_pxy.append(i)
#         elif orb.l == 1 and orb.m == 0:
#             idx_pz.append(i)
#         elif orb.l == 2:
#             idx_d.append(i)

#     idx_hs = []
#     idx_hpz = []
#     idx_hpxy = []
#     for i, orb in enumerate(H.geometry.atoms[H_atoms[0]]):
#         if orb.l == 1 and orb.m == 0:
#             idx_hpz.append(i)
#         elif orb.l == 1 and (orb.m in [-1, 1]):
#             idx_hpxy.append(i)
#         elif orb.l == 0:
#             idx_hs.append(i)

#     all_s = np.add.outer(H.geometry.a2o(C_atoms), idx_s).ravel()
#     all_pxy = np.add.outer(H.geometry.a2o(C_atoms), idx_pxy).ravel()
#     all_pz = np.add.outer(H.geometry.a2o(C_atoms), idx_pz).ravel()
#     all_d = np.add.outer(H.geometry.a2o(C_atoms), idx_d).ravel()
#     all_hs = np.add.outer(H.geometry.a2o(H_atoms), idx_hs).ravel()
#     all_hpxy = np.add.outer(H.geometry.a2o(H_atoms), idx_hpxy).ravel()
#     all_hpz = np.add.outer(H.geometry.a2o(H_atoms), idx_hpz).ravel()

#     weight_s = []
#     weight_pxy = []
#     weight_pz = []
#     weight_d = []
#     weight_hs = []
#     weight_hpxy = []
#     weight_hpz = []

#     def wrap_fat_bands(eigenstate):
#         """
#         <psi_{i,v}|S(k)|psi_i>
#         """
#         norm2 = eigenstate.norm2(sum=False)
#         weight_s.append(norm2[:, all_s].sum(-1))
#         weight_pxy.append(norm2[:, all_pxy].sum(-1))
#         weight_pz.append(norm2[:, all_pz].sum(-1))
#         weight_d.append(norm2[:, all_d].sum(-1))
#         weight_hs.append(norm2[:, all_hs].sum(-1))
#         weight_hpxy.append(norm2[:, all_hpxy].sum(-1))
#         weight_hpz.append(norm2[:, all_hpz].sum(-1))
#         return eigenstate.eig

#     kpoints = 400
#     bs = BandStructure(H, [[0, 0, 0], [1, 0, 0]],
#                        kpoints, ['$\Gamma$', '$\Gamma$'])
#     bsar = bs.apply.array
#     eig = bsar.eigenstate(wrap=wrap_fat_bands).T

#     linear_k, k_tick, k_label = bs.lineark(True)

#     Emin, Emax = Erange
#     dE = (Emax - Emin)/(figsize[1]*5)

#     weight_s = np.array(weight_s).T
#     weight_pxy = np.array(weight_pxy).T
#     weight_pz = np.array(weight_pz).T
#     weight_d = np.array(weight_d).T
#     weight_hs = np.array(weight_hs).T
#     weight_hpxy = np.array(weight_hpxy).T
#     weight_hpz = np.array(weight_hpz).T

#     plt.figure(figsize=figsize)
#     plt.ylabel('$E-E_F$ [eV]')
#     plt.xlim(linear_k[0], linear_k[-1])
#     plt.xticks(k_tick, k_label)
#     plt.ylim(Emin, Emax)

#     fatpzk = 0
#     for i, e_all_k in enumerate(eig):
#         s_abs = np.abs(weight_s[i, :] * dE)
#         pxy_abs = np.abs(weight_pxy[i, :] * dE)
#         pz_abs = np.abs(weight_pz[i, :] * dE)
#         d_abs = np.abs(weight_d[i, :] * dE)
#         hs_abs = np.abs(weight_hs[i, :] * dE)
#         hpxy_abs = np.abs(weight_hpxy[i, :] * dE)
#         hpz_abs = np.abs(weight_hpz[i, :] * dE)
#         plt.plot(linear_k, e_all_k, color='k')
#         if index:
#             if Erange[0] < e_all_k[-1] < Erange[1]:
#                 plt.annotate(i+1, (linear_k[-1], e_all_k[-1]))

#         if np.any(pz_abs/dE > 0.5):
#             # select k-points where pz makes major contribution
#             where_pz = np.where(pz_abs/dE > 0.5)[0]
#             if where_pz.size:
#                 fatpzk += len(where_pz)

#             # split the k-points array into seperate segments
#             klist = []  # list of k-segments
#             kseg = []  # k-segments
#             for i in range(len(where_pz)):
#                 if where_pz[i] - where_pz[i-1] > 1:
#                     klist.append(kseg)
#                     kseg = []
#                 kseg.append(where_pz[i])
#             klist.append(kseg)

#             for i in range(len(klist)):
#                 k_pz = linear_k[klist[i]]
#                 e = e_all_k[klist[i]]
#                 s = s_abs[klist[i]]
#                 pxy = pxy_abs[klist[i]]
#                 pz = pz_abs[klist[i]]
#                 d = d_abs[klist[i]]
#                 hs = hs_abs[klist[i]]
#                 hpxy = hpxy_abs[klist[i]]
#                 hpz = hpz_abs[klist[i]]

#                 # Full fat-band
#                 plt.fill_between(k_pz, e-dE, e+dE, color='k', alpha=0.1)
#                 # s
#                 plt.fill_between(k_pz, e-(s), e+(s), color='C0', alpha=0.5)
#                 # pxy
#                 plt.fill_between(k_pz, e+(s), e+(s+pxy), color='C1', alpha=0.5)
#                 plt.fill_between(k_pz, e-(s+pxy), e-(s), color='C1', alpha=0.5)
#                 # pz
#                 plt.fill_between(k_pz, e+(s+pxy), e+(s+pxy+pz),
#                                  color='C2', alpha=0.5)
#                 plt.fill_between(k_pz, e-(s+pxy+pz), e -
#                                  (s+pxy), color='C2', alpha=0.5)
#                 # d
#                 plt.fill_between(k_pz, e+(s+pxy+pz), e +
#                                  (s+pxy+pz+d), color='C3', alpha=0.5)
#                 plt.fill_between(k_pz, e-(s+pxy+pz+d), e -
#                                  (s+pxy+pz), color='C3', alpha=0.5)
#                 # hs
#                 plt.fill_between(k_pz, e+(s+pxy+pz+d), e +
#                                  (s+pxy+pz+d+hs), color='C4', alpha=0.5)
#                 plt.fill_between(k_pz, e-(s+pxy+pz+d+hs), e -
#                                  (s+pxy+pz+d), color='C4', alpha=0.5)
#                 # hpxy
#                 plt.fill_between(k_pz, e+(s+pxy+pz+d+hs), e +
#                                  (s+pxy+pz+d+hs+hpxy), color='C5', alpha=0.5)
#                 plt.fill_between(k_pz, e-(s+pxy+pz+d+hs+hpxy),
#                                  e-(s+pxy+pz+d+hs), color='C5', alpha=0.5)
#                 # hpz
#                 plt.fill_between(k_pz, e+(s+pxy+pz+d+hs+hpxy), e +
#                                  (s+pxy+pz+d+hs+hpxy+hpz), color='C6', alpha=0.5)
#                 plt.fill_between(k_pz, e-(s+pxy+pz+d+hs+hpxy+hpz),
#                                  e-(s+pxy+pz+d+hs+hpxy), color='C6', alpha=0.5)

#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor='C0', label='C: $s$'),
#                        Patch(facecolor='C1', label='C: $p_x+p_y$'),
#                        Patch(facecolor='C2', label='C: $p_z$'),
#                        Patch(facecolor='C3', label='C: $d$'),
#                        Patch(facecolor='C4', label='H: $hs$'),
#                        Patch(facecolor='C5', label='H: $hp_x + hp_y$'),
#                        Patch(facecolor='C6', label='H: $hp_z$')]
#     plt.legend(handles=legend_elements, bbox_to_anchor=[1.1, 0.9])

#     num_pz_band = fatpzk/kpoints
#     print(f"Total number of pz fat bands: {round(num_pz_band)} (true value\
#           {num_pz_band}")


@timer
def plot_eigst_band(H, offset: list = [0], k=None, figsize=(15, 5), dotsize=500):
    """
    Plot the eigenstate of a band, by default the topmost valence band
    - offset: offset from the fermi level, or, the topmost valence band
    """

    es = H.eigenstate(k=k) if k else H.eigenstate()

    occn = len(H) // 2 - 1
    print("Index of the HOMO: ", occn)
    bands = []
    offset.sort()
    for i in offset:
        bands.append(occn+i)
    print("Bands that are taken into account: ", bands)
    print("Energy: ", [H.eigh()[i] for i in bands])
    plt.figure(figsize=figsize)
    esnorm = es.sub(bands).norm2(sum=False).sum(0)
    plt.scatter(H.xyz[:, 0], H.xyz[:, 1], dotsize*esnorm)
    plt.axis('equal')


@timer
def plot_eigst_energy(H, E=0.0, Ewidth=0.1, k=None, figsize=(15, 5), dotsize=100):
    """
    Plot the eigenstates whose eigenvalues are in a specific range, by default around fermi level
    Note that this method sums all the orbitals of one atom and plot it as a circle,
    therefore the orbital information is hidden here. To visualize the orbitals, 
    use ldos_map method instead.
    pdos adds all the atoms for an orbital, while plot_eigst adds
    all the orbitals for an atom.
    """

    geom = H.geometry
    bs = BandStructure(H, [[0, 0, 0], [1, 0, 0]],
                       100, ['$\Gamma$', '$\Gamma$'])
    bsav = bs.apply.average

    Emin = E-Ewidth/2
    Emax = E+Ewidth/2
    mesh_pts = int(Ewidth/0.01)
    Emesh = np.linspace(Emin, Emax, mesh_pts)
    lpdos = np.zeros((geom.na, mesh_pts))

    def wrap(PDOS):
        # local projected dos
        # sum all the orbitals for each atom
        for io in range(PDOS.shape[0]):
            ia = geom.o2a(io)
            lpdos[ia, :] += PDOS[io, :]
        return lpdos

    lpdos = bsav.PDOS(Emesh, wrap=wrap)
    lpdos = lpdos.sum(-1)
    plt.figure(figsize=figsize)
    plt.scatter(geom.xyz[:, 0], geom.xyz[:, 1], dotsize*lpdos)
    plt.axis('equal')


@timer
def ldos(H, location, Erange=[-3, 3], figsize=None,
         rescale=[0, 1], color='coral', ret=False,  **kwargs):
    """
    Plot the local density of states
    Args:
        location: index of the atom
    """
    es = H.eigenstate()
    Emin, Emax = Erange
    eig = H.eigh()
    sub = np.where(np.logical_and(eig > Emin, eig < Emax))
    es_sub = es.sub(sub)
    eig_sub = eig[sub]
    es_sub_loc = es_sub.state[:, location]
    ldos = np.multiply(es_sub_loc, np.conj(es_sub_loc))
    ldos = ldos/ldos.max()  # from 0 to 1
    # change the scale, now from rescale[0] to rescale[1]
    ldos = ldos*(rescale[1]-rescale[0]) + rescale[0]

    if len(ldos.shape) == 1:
        m, n = ldos.shape[0], 1
    else:
        m, n = ldos.shape
    if not figsize:
        figsize = (1*n, 5)
    fig, axes = plt.subplots(
        1, n, sharex=True, sharey=True, figsize=figsize, gridspec_kw={'wspace': 0})
    if n > 1:
        for i in range(n):
            ax = axes[i]
            for j in range(m):
                ax.hlines(eig_sub[j], 0, 1, alpha=ldos[j, i],
                          color='coral', **kwargs)
            ax.set_xticks([])
            ax.set_ylim(Emin, Emax)
            ax.set_title(location[i])
        axes[0].set_ylabel('$E-E_F$ (eV)')
    elif n == 1:
        for j in range(m):
            ax = axes
            ax.hlines(eig_sub[j], 0, 1, alpha=ldos[j], color='coral', **kwargs)
            ax.set_xticks([])
            ax.set_ylim(Emin, Emax)
            ax.set_title(location)
        ax.set_ylabel('$E-E_F$ (eV)')
    if ret:
        return eig_sub, ldos


@timer
def ldos_map(H, E=0.0, Ewidth=0.1, height=3.0, mesh=0.1, figsize=(15, 5), colorbar=False):
    """
    Localized Density of States
    - E: the median of the energy range that you want to investigate
    - Ewidth: width of the energy range
    - height: height above the plane of the ribbon, default 1.0 angstrom
    - mesh: mesh size of the grid, default 0.1 angstrom
    """

    Emin = E-Ewidth/2
    Emax = E+Ewidth/2

    es = H.eigenstate()
    eig = H.eigh()
    sub = np.where(np.logical_and(eig > Emin, eig < Emax))[0]

    dos = 0
    for b in sub:
        grid = Grid(mesh, sc=H.sc)
        index = grid.index([0, 0, height])
        es_sub = es.sub(b)
        es_sub.wavefunction(grid)  # add the wavefunction to grid
        dos += grid.grid[:, :, index[2]].T ** 2

    plt.figure(figsize=figsize)
    plt.imshow(dos, cmap='hot')
    plt.xticks([])
    plt.yticks([])
    if colorbar:
        plt.colorbar()
    return dos


@timer
def zak_phase(H):

    bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
                       '$\Gamma$', '$X$', '$\Gamma$'])

    occ = [i for i in range(len(H.eig())//2)]
    gamma = electron.berry_phase(bs, sub=occ, method='zak')

    return gamma


def zak(contour, sub=None, gauge='R'):

    from sisl import Hamiltonian
    import sisl._array as _a
    from sisl.linalg import det_destroy
    import numpy as np
    from numpy import dot, angle

    if not isinstance(contour.parent, Hamiltonian):
        raise TypeError(
            'Requires the Brillouine zone object to contain a Hamiltonian')

    if not contour.parent.orthogonal:
        raise TypeError('Requires the Hamiltonian to use orthogonal basis')

    if not sub:
        raise ValueError('Calculate only the occupied bands!')

    def _zak(eigenstates):
        first = next(eigenstates).sub(sub)
        if gauge == 'r':
            first.change_gauge('r')
        prev = first
        prd = 1
        for second in eigenstates:
            second = second.sub(sub)
            if gauge == 'r':
                second.change_gauge('r')
            prd = dot(prd, prev.inner(second, diagonal=False))
            prev = second
        if gauge == 'r':
            g = contour.parent.geometry
            axis = contour.k[1] - contour.k[0]
            axis /= axis.dot(axis) ** 0.5
            phase = dot(g.xyz[g.o2a(_a.arangei(g.no)), :],
                        dot(axis, g.rcell)).reshape(1, -1)
            prev.state *= np.exp(-1j*phase)
        prd = dot(prd, prev.inner(first, diagonal=False))
        return prd

    d = _zak(contour.apply.iter.eigenstate())
    ddet = det_destroy(d)
    result = -angle(ddet)

    return result


@timer
def inter_zak(H, offset=0):

    bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
                       '$\Gamma$', '$X$', '$\Gamma$'])

    occ = [i for i in range(len(H.eig())//2)]
    if offset == 0:
        occ = occ
    elif offset > 0:
        for i in range(offset):
            occ.append(occ[-1]+1)
    elif offset < 0:
        for i in range(-offset):
            occ.pop()

    gamma_inter = zak(bs, sub=occ, gauge='R')

    return round(gamma_inter, 10)


@timer
def ssh(H, occ):
    """
    Please provide the offset from homo as occ
    """
    bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
                       '$\Gamma$', '$X$', '$\Gamma$'])

    occn = len(H) // 2 - 1
    print("Index of the HOMO: ", occn)
    bands = []
    occ.sort()
    for i in occ:
        bands.append(occn+i)
    print("Bands that are taken into account: ", bands)
    gamma_inter = zak(bs, sub=bands, gauge='R')

    return round(gamma_inter, 10)


@timer
def zak_band(H, occ):

    bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
                       '$\Gamma$', '$X$', '$\Gamma$'])

    gamma_inter = zak(bs, sub=occ, gauge='R')

    return round(gamma_inter, 10)


def zak_dft(contour, sub=None, gauge='R'):

    from sisl import Hamiltonian, Overlap
    import sisl._array as _a
    from sisl.linalg import det_destroy
    import numpy as np
    from numpy import dot, angle, multiply, conj

    if not isinstance(contour.parent, Hamiltonian):
        raise TypeError(
            'Requires the Brillouine zone object to contain a Hamiltonian')

    if sub is None:
        raise ValueError('Calculate only the occupied bands!')

    def _zak(eigenstates):

        H = contour.parent
        k = contour.k
        dk = k[1] - k[0]

        first = next(eigenstates).sub(sub)
        if gauge == 'r':
            first.change_gauge('r')
        prev = first

        prd = 1
        for second in eigenstates:
            if gauge == 'r':
                second.change_gauge('r')
            second = second.sub(sub)

            k_sec = second.info['k']
            ovlpm = H.Sk(k=k_sec-dk/2, gauge='R', format='array')
            prev_state = prev.state
            second_state = second.state
            inner_prd = dot(dot(conj(prev_state), ovlpm), second_state.T)
            prd = dot(prd, inner_prd)
            prev = second

        if gauge == 'r':
            g = contour.parent.geometry
            axis = contour.k[1] - contour.k[0]
            axis /= axis.dot(axis) ** 0.5
            coords = g.xyz[g.o2a(_a.arangei(g.no)), :]
            phase = dot(coords, dot(axis, g.rcell)).reshape(1, -1)
            prev.state *= np.exp(-1j*phase)

        # in case the last state and first state are not equal
        ovlpm_last = H.Sk(k=[0]*3, gauge='R', format='array')
        last_state = prev.state
        first_state = first.state
        inner_prd_last = dot(dot(conj(last_state), ovlpm_last), first_state.T)
        prd = dot(prd, inner_prd_last)
        return prd

    d = _zak(contour.apply.iter.eigenstate())

    d_det = det_destroy(d)
    result = -angle(d_det)

    return d_det, result


def zak_band_dft(H, occ):

    bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
                       '$\Gamma$', '$X$', '$\Gamma$'])

    d_det, gamma_inter = zak_dft(bs, sub=occ, gauge='R')

    return d_det, round(gamma_inter, 10)


def get_zak_dict(H, method='from_top'):
    """
    Get the Zak phase list (by default, we mean intercellualr Zak phase)
    Format of list:
        {index: (bands_list, determinant, Zak_phase)}
    -method:
        'from_top': occupied bands counted from top (fermi) to bottom
        'from_bottom': occupied bands counted from bottom to top (fermi)
    """

    occ = np.where(H.eigh() < 0)[0]

    zak_dict = {}

    for i in range(len(occ)):
        if method == 'from_top':
            sub = occ[-i-1:]
        elif method == 'from_bottom':
            sub = occ[:i+1]
        d, z = zak_band_dft(H, sub)
        zak_dict.update({i: (sub, d, z)})
    return zak_dict


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


def plot_zak_polar(zdict):
    """
    Put Zak phase in a complex plane, which is a polar plot here
    """
    from matplotlib import cm

    plt.figure(figsize=(10, 10))
    for i, v in zdict.items():
        bwr = cm.get_cmap('bwr')
        rho, r = np.angle(v[1]), np.abs(v[1])
        plt.polar([0, rho], [0, r], marker='o',
                  color=bwr(-r*np.cos(rho)+1),
                  label=list_str(v[0])+': '+str(round(r*np.cos(rho), 2)),
                  alpha=r*0.8+0.05)
        #plt.text(rho, r, str(i), color="red", fontsize=10)
        plt.legend(bbox_to_anchor=[1.2, 0.9])


def plot_wannier_centers(geom, name, path=None, figsize=(6, 4), sc=False, marker='*', marker_size=5, marker_color='green'):
    if not path:
        path = './s2w/'
    cell = geom.cell
    gcenter = geom.center()
    abc = cell.diagonal()
    file_path = os.path.join(path, name+'_centres.xyz')
    with open(file_path) as f:
        contents = f.readlines()
        # wannier centres in raw coordinates strings
        wc_raw = contents[2:2+num_wann]
        # convert it to an array
        wc = np.array(list(map(lambda x: list(map(
            float, x.strip().split()[1:])), wc_raw)))
        wc_hc = wc - np.floor((wc-(gcenter-abc/2))/abc).dot(cell)
        # sum of wannier centers
        wcs = wc_hc.sum(0)
        temp = wcs.dot(np.linalg.inv(cell))
        wcs = np.dot(temp - np.floor(temp), cell)
        print('Sum of Wannier centres: ', wcs)
    plt.figure(figsize=figsize)
    plot(geom, supercell=sc)
    plt.scatter(wc_hc[:, 0], wc_hc[:, 1], s=marker_size,
                c=marker_color, marker=marker)
    plt.axis('equal')
