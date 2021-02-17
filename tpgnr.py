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
def copy_hamiltonian(H):
    
    if not isinstance(H, sisl.physics.Hamiltonian):
        raise TypeError('H must be a Hamiltonian object')
    a, b, c = H.shape
    h = np.empty([a, b, c])
    for i in range(a):
        for j in range(b):
            for k in range(c):
                h[i,j,k] = H[i,j,k]
    return h


def move_to_origo(gnr):
    """
    Move the geometry center to origin
    """
    
    gnr = gnr.translate([-gnr.center()[0], -gnr.center()[1], 0])
    plot(gnr)
    plt.axis('equal')
    
    return gnr


def move_to_center(g):
    """
    Move the geometry to the center of supercell
    """
    cell_center = g.center(what='cell')
    # The internal method of xyz center is stupid, don't use it
    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    maxxyz = np.amax(xyz, 0)
    geom_center = (minxyz + maxxyz) / 2
    g = g.translate([cell_center[0]-geom_center[0],
                     cell_center[1]-geom_center[1], 0])
    plot(g)
    plt.axis('equal')
    
    return g


def align_hcenter(g):
    """
    Align the geometry to the horizontal center
    """
    cell = g.cell
    g = g.translate([cell[0,0]/2-g.center()[0], 0, 0])
    plot(g)
    plt.axis('equal')
    
    return g


def align_vcenter(g):
    """
    Align the geometry to the horizontal center
    """
    cell = g.cell
    g = g.translate([0, cell[1,1]/2-g.center()[1], 0])
    plot(g)
    plt.axis('equal')
    
    return g


def display(g, aid=False, sc=True, rotate=False, figsize=(10,5)):
    
    xyz = g.xyz
    minxyz = np.amin(xyz, 0)
    print('minxyz:', minxyz)
    maxxyz = np.amax(xyz, 0)
    print('maxxyz:', maxxyz)
    length = maxxyz - minxyz
    print('size:', length)
    
    plt.figure(figsize=figsize)
    if rotate:
        if round(g.cell[0,1],5) != 0:
            angle = np.degrees(np.arctan(g.cell[0,1] / g.cell[0,0]))
            g = g.rotate(-angle, [0,0,1])
    plot(g, atom_indices=aid, supercell=sc)
    plt.axis('equal')
    
    if length[0] > length[1]: # x is larger
        plt.xlim(minxyz[0]-6, maxxyz[0]+6)
    else:
        plt.ylim(minxyz[1]-2, maxxyz[1]+2)


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
    
    r = np.linspace(0,5,100)
    epsilon = 1.625
    f = np.exp(-epsilon*r)
    # Normalize
    A = 1/(f.sum() * (r[1] - r[0]))
    f = A * f
    
    if basis == 'DZP':
        pz = AtomicOrbital((r,f), n=2, l=1, m=0, Z=2, P=1)
    elif basis == 'DZ':
        pz = AtomicOrbital((r,f), n=2, l=1, m=0, Z=2)
    elif basis == 'SZP':
        pz = AtomicOrbital((r,f), n=2, l=1, m=0, Z=1, P=1)
    elif basis == 'SZ':
        pz = AtomicOrbital((r,f), n=2, l=1, m=0, Z=1)
    C = Atom(6, pz)
    g.atoms.replace(g.atoms[0], C)


def construct_hamiltonian(gnr):
    
    H = Hamiltonian(gnr)
    r = (0.1, 1.44)
    t = (0., -2.7)
    H.construct([r,t])
    print(H)
    return H



@timer
def band_structure(H, Erange=[-10,10], figsize=(6,4)):
    
    bs = BandStructure(H, [[0,0,0],[0.5,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$','$\Gamma$'])
        
    bsar = bs.apply.array
    eigh = bsar.eigh()

    lk, kt, kl = bs.lineark(True)
    
    plt.figure(figsize=figsize)
    plt.xticks(kt, kl)
    plt.ylim(Erange[0], Erange[-1])
    plt.xlim(0, lk[-1])
    plt.ylabel('$E-E_F$ (eV)')

    for ek in eigh.T:
        plt.plot(lk, ek)


@timer
def energy_levels(H, Erange=[-10,10], figsize=(6,4), index=False):
    """
    should be a single molecule, nsc=[1,1,1]
    """
    eig = H.eigh()
    plt.figure(figsize=figsize)
    plt.hlines(eig, 0, 10) # use Hermitian solver, read values
    plt.ylim(Erange[0], Erange[-1])
    plt.xticks([])
    
    if index: # label the index of energy levels  
        which = np.where(np.logical_and(eig<Erange[-1],
                                        eig>Erange[0]))[0]
        for i in which:
            plt.text(11,eig[i], str(i))


@timer
def dos(H, Erange=[-10,10], figsize=(6,4)):
    
    if all(H.nsc == [3,1,1]):
        bs = BandStructure(H, [[0,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$'])
    elif all(H.nsc == [1,3,1]):
        bs = BandStructure(H, [[0,0,0],[0,1,0]], 400, ['$\Gamma$', '$X$'])
        
    bsav = bs.apply.average
    
    E = np.linspace(Erange[0], Erange[-1], 1000)
    plt.figure(figsize=figsize)
    plt.plot(E, bsav.DOS(E))
    plt.xlim(-10, 10)
    plt.ylim(0, np.amax(bsav.DOS(E)) + 1)



@timer
def pdos(H, Erange=(-20,20), figsize=(6,4)):
    """
    Projected density of states
    Here we consider s, px, py, pz, d orbitals of C, and s, px, py, pz orbitals of H
    """
        
    bs = BandStructure(H, [[0,0,0],[1,0,0]], 400, ['$\Gamma$', '$\Gamma$'])
    bsav = bs.apply.average
    
    C_atoms = []
    H_atoms = []
    for i, at in enumerate(H.atoms):
        if at.Z == 6:
            C_atoms.append(i)
        elif at.Z == 1:
            H_atoms.append(i)
    
    idx_s = []
    idx_pxy = []
    idx_pz = []
    idx_d = []
    for i, orb in enumerate(H.geometry.atoms[C_atoms[0]]):
        if orb.l == 0:
            idx_s.append(i)
        elif orb.l == 1 and (orb.m in [-1, 1]):
            idx_pxy.append(i)
        elif orb.l == 1 and orb.m == 0:
            idx_pz.append(i)
        elif orb.l == 2:
            idx_d.append(i)
    
    idx_hs = []
    idx_hpz = []
    idx_hpxy = []
    for i, orb in enumerate(H.geometry.atoms[H_atoms[0]]):
        if orb.l == 1 and orb.m == 0:
            idx_hpz.append(i)
        elif orb.l == 1 and (orb.m in [-1,1]):
            idx_hpxy.append(i)
        elif orb.l == 0:
            idx_hs.append(i)
    
    all_s = np.add.outer(H.geometry.a2o(C_atoms), idx_s).ravel()
    all_pxy = np.add.outer(H.geometry.a2o(C_atoms), idx_pxy).ravel()
    all_pz = np.add.outer(H.geometry.a2o(C_atoms), idx_pz).ravel()
    all_d = np.add.outer(H.geometry.a2o(C_atoms), idx_d).ravel()
    all_hs = np.add.outer(H.geometry.a2o(H_atoms), idx_hs).ravel()
    all_hpxy = np.add.outer(H.geometry.a2o(H_atoms), idx_hpxy).ravel()
    all_hpz = np.add.outer(H.geometry.a2o(H_atoms), idx_hpz).ravel()
    
    def wrap(PDOS):
        pdos_s = PDOS[all_s, :].sum(0)
        pdos_pxy = PDOS[all_pxy, :].sum(0)
        pdos_pz = PDOS[all_pz, :].sum(0)
        pdos_d = PDOS[all_d, :].sum(0)
        pdos_hs = PDOS[all_hs, :].sum(0)
        pdos_hpxy = PDOS[all_hpxy, :].sum(0)
        pdos_hpz = PDOS[all_hpz, :].sum(0)
        return np.stack((pdos_s,
                         pdos_pxy, pdos_pz,
                         pdos_d, pdos_hs,
                         pdos_hpxy, pdos_hpz))
    
    E = np.linspace(Erange[0],Erange[-1],500)
    pDOS = bsav.PDOS(E, wrap=wrap)
    plt.figure(figsize=figsize)
    plt.plot(E, pDOS[0, :], color='C0', label='C: $s$')
    plt.plot(E, pDOS[1, :], color='C1', label='C: $p_x+p_y$')
    plt.plot(E, pDOS[2, :], color='C2', label='C: $p_z$')
    plt.plot(E, pDOS[3, :], color='C3', label='C: $d$')
    plt.plot(E, pDOS[4, :], color='C4', label='H: $hs$')
    plt.plot(E, pDOS[5, :], color='C5', label='H: $hp_x + hp_y$')
    plt.plot(E, pDOS[6, :], color='C6', label='H: $hp_z$')
    plt.xlim(E[0], E[-1])
    plt.ylim(0, None)
    plt.xlabel(r'$E - E_F$ [eV]')
    plt.ylabel(r'DOS [1/eV]')
    plt.legend(bbox_to_anchor=[1.1,0.9])



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
def pzdos(H, Erange=[-10,20], plot_pzdos=True):
    
    import numpy as np
    
    bs = BandStructure(H, [[0,0,0],[1,0,0]], 400, ['$\Gamma$', '$\Gamma$'])
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
    E = np.linspace(Emin, Emax,100)
    pDOS = bsav.PDOS(E, wrap=wrap)
    if plot_pzdos:
        for i, label in enumerate(all_pz):
            plt.plot(E, pDOS[i,:],label=label)
        plt.xlim(E[0], E[-1])
        plt.ylim(0, None)
        plt.xlabel(r'$E - E_F$ [eV]')
        plt.ylabel(r'pz_DOS [1/eV]')
        plt.title("Project DOS on pz orbitals")
        plt.legend(loc="best", bbox_to_anchor=[1.4,0.9])
    return pDOS


def pzweight(H, Erange=[-10,0]):
    # plot the weight of pz orbitals in a specific energy range, by default the occupied band
    pzd = pzdos(H, Erange,plot_pzdos=False)
    pzwt = np.multiply(pzd, pzd).sum(-1)
    pzidx = all_pz(H)
    plt.bar(pzidx, pzwt, width=1)
    plt.ylim(0,1)
    for i, idx in enumerate(list(pzidx)):
        plt.annotate(idx, (pzidx[i], pzwt[i]))




@timer
def fat_bands(H, Erange=(-20,20), figsize=(10,8)):
    """
    Plot the fat bands, showing the weight of each kinds of orbital of every band.
    """
    
    C_atoms = []
    H_atoms = []
    for i, at in enumerate(H.atoms):
        if at.Z == 6:
            C_atoms.append(i)
        elif at.Z == 1:
            H_atoms.append(i)
    
    idx_s = []
    idx_pxy = []
    idx_pz = []
    idx_d = []
    for i, orb in enumerate(H.geometry.atoms[C_atoms[0]]):
        if orb.l == 0:
            idx_s.append(i)
        elif orb.l == 1 and (orb.m in [-1, 1]):
            idx_pxy.append(i)
        elif orb.l == 1 and orb.m == 0:
            idx_pz.append(i)
        elif orb.l == 2:
            idx_d.append(i)
    
    idx_hs = []
    idx_hpz = []
    idx_hpxy = []
    for i, orb in enumerate(H.geometry.atoms[H_atoms[0]]):
        if orb.l == 1 and orb.m == 0:
            idx_hpz.append(i)
        elif orb.l == 1 and (orb.m in [-1,1]):
            idx_hpxy.append(i)
        elif orb.l == 0:
            idx_hs.append(i)
    
    all_s = np.add.outer(H.geometry.a2o(C_atoms), idx_s).ravel()
    all_pxy = np.add.outer(H.geometry.a2o(C_atoms), idx_pxy).ravel()
    all_pz = np.add.outer(H.geometry.a2o(C_atoms), idx_pz).ravel()
    all_d = np.add.outer(H.geometry.a2o(C_atoms), idx_d).ravel()
    all_hs = np.add.outer(H.geometry.a2o(H_atoms), idx_hs).ravel()
    all_hpxy = np.add.outer(H.geometry.a2o(H_atoms), idx_hpxy).ravel()
    all_hpz = np.add.outer(H.geometry.a2o(H_atoms), idx_hpz).ravel()
    
    weight_s = []
    weight_pxy = []
    weight_pz = []
    weight_d = []
    weight_hs = []
    weight_hpxy = []
    weight_hpz = []
    
    def wrap_fat_bands(eigenstate):
        """
        <psi_{i,v}|S(k)|psi_i>
        """
        norm2 = eigenstate.norm2(sum=False)
        weight_s.append(norm2[:, all_s].sum(-1))
        weight_pxy.append(norm2[:, all_pxy].sum(-1))
        weight_pz.append(norm2[:, all_pz].sum(-1))
        weight_d.append(norm2[:, all_d].sum(-1))
        weight_hs.append(norm2[:, all_hs].sum(-1))
        weight_hpxy.append(norm2[:, all_hpxy].sum(-1))
        weight_hpz.append(norm2[:, all_hpz].sum(-1))
        return eigenstate.eig
        
    bs = BandStructure(H, [[0,0,0],[1,0,0]], 400, ['$\Gamma$', '$\Gamma$'])
    bsar = bs.apply.array
    eig = bsar.eigenstate(wrap=wrap_fat_bands).T
    
    linear_k, k_tick, k_label = bs.lineark(True)
    
    Emin, Emax = Erange
    dE = (Emax - Emin)/40.
    
    weight_s = np.array(weight_s).T
    weight_pxy = np.array(weight_pxy).T
    weight_pz = np.array(weight_pz).T
    weight_d = np.array(weight_d).T
    weight_hs = np.array(weight_hs).T
    weight_hpxy = np.array(weight_hpxy).T
    weight_hpz = np.array(weight_hpz).T
    
    plt.figure(figsize=figsize)
    plt.ylabel('$E-E_F$ [eV]')
    plt.xlim(linear_k[0], linear_k[-1])
    plt.xticks(k_tick, k_label)
    plt.ylim(Emin, Emax)
    
    for i, e in enumerate(eig):
        s = np.abs(weight_s[i, :] * dE)
        pxy = np.abs(weight_pxy[i, :] * dE)
        pz = np.abs(weight_pz[i, :] * dE)
        d = np.abs(weight_d[i, :] * dE)
        hs = np.abs(weight_hs[i, :] * dE)
        hpxy = np.abs(weight_hpxy[i, :] * dE)
        hpz = np.abs(weight_hpz[i, :] * dE)
        plt.plot(linear_k, e, color='k')
        # Full fat-band
        plt.fill_between(linear_k, e-dE, e+dE, color='k', alpha=0.1)
        # s
        plt.fill_between(linear_k, e-(s), e+(s), color='C0', alpha=0.5)
        # pxy
        plt.fill_between(linear_k, e+(s), e+(s+pxy), color='C1', alpha=0.5)
        plt.fill_between(linear_k, e-(s+pxy), e-(s), color='C1', alpha=0.5)
        # pz
        plt.fill_between(linear_k, e+(s+pxy), e+(s+pxy+pz), color='C2', alpha=0.5)
        plt.fill_between(linear_k, e-(s+pxy+pz), e-(s+pxy), color='C2', alpha=0.5)
        # d
        plt.fill_between(linear_k, e+(s+pxy+pz), e+(s+pxy+pz+d), color='C3', alpha=0.5)
        plt.fill_between(linear_k, e-(s+pxy+pz+d), e-(s+pxy+pz), color='C3', alpha=0.5)
        # hs
        plt.fill_between(linear_k, e+(s+pxy+pz+d), e+(s+pxy+pz+d+hs), color='C4', alpha=0.5)
        plt.fill_between(linear_k, e-(s+pxy+pz+d+hs), e-(s+pxy+pz+d), color='C4', alpha=0.5)
        # hpxy
        plt.fill_between(linear_k, e+(s+pxy+pz+d+hs), e+(s+pxy+pz+d+hs+hpxy), color='C5', alpha=0.5)
        plt.fill_between(linear_k, e-(s+pxy+pz+d+hs+hpxy), e-(s+pxy+pz+d+hs), color='C5', alpha=0.5)
        # hpz
        plt.fill_between(linear_k, e+(s+pxy+pz+d+hs+hpxy), e+(s+pxy+pz+d+hs+hpxy+hpz), color='C6', alpha=0.5)
        plt.fill_between(linear_k, e-(s+pxy+pz+d+hs+hpxy+hpz), e-(s+pxy+pz+d+hs+hpxy), color='C6', alpha=0.5)
      
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='C0', label='C: $s$'),
                       Patch(facecolor='C1', label='C: $p_x+p_y$'),
                       Patch(facecolor='C2', label='C: $p_z$'),
                       Patch(facecolor='C3', label='C: $d$'),
                       Patch(facecolor='C4', label='H: $hs$'),
                       Patch(facecolor='C5', label='H: $hp_x + hp_y$'),
                       Patch(facecolor='C6', label='H: $hp_z$')]
    plt.legend(handles=legend_elements, bbox_to_anchor=[1.1,0.9])
        

@timer
def fat_bands_pz(H, Erange=(-10,0), figsize=(10,8)):
    """
    Plot the fat bands, showing the weight of each kinds of orbital of every band.
    """
    
    C_atoms = []
    H_atoms = []
    for i, at in enumerate(H.atoms):
        if at.Z == 6:
            C_atoms.append(i)
        elif at.Z == 1:
            H_atoms.append(i)
    
    idx_s = []
    idx_pxy = []
    idx_pz = []
    idx_d = []
    for i, orb in enumerate(H.geometry.atoms[C_atoms[0]]):
        if orb.l == 0:
            idx_s.append(i)
        elif orb.l == 1 and (orb.m in [-1, 1]):
            idx_pxy.append(i)
        elif orb.l == 1 and orb.m == 0:
            idx_pz.append(i)
        elif orb.l == 2:
            idx_d.append(i)
    
    idx_hs = []
    idx_hpz = []
    idx_hpxy = []
    for i, orb in enumerate(H.geometry.atoms[H_atoms[0]]):
        if orb.l == 1 and orb.m == 0:
            idx_hpz.append(i)
        elif orb.l == 1 and (orb.m in [-1,1]):
            idx_hpxy.append(i)
        elif orb.l == 0:
            idx_hs.append(i)
    
    all_s = np.add.outer(H.geometry.a2o(C_atoms), idx_s).ravel()
    all_pxy = np.add.outer(H.geometry.a2o(C_atoms), idx_pxy).ravel()
    all_pz = np.add.outer(H.geometry.a2o(C_atoms), idx_pz).ravel()
    all_d = np.add.outer(H.geometry.a2o(C_atoms), idx_d).ravel()
    all_hs = np.add.outer(H.geometry.a2o(H_atoms), idx_hs).ravel()
    all_hpxy = np.add.outer(H.geometry.a2o(H_atoms), idx_hpxy).ravel()
    all_hpz = np.add.outer(H.geometry.a2o(H_atoms), idx_hpz).ravel()
    
    weight_s = []
    weight_pxy = []
    weight_pz = []
    weight_d = []
    weight_hs = []
    weight_hpxy = []
    weight_hpz = []
    
    def wrap_fat_bands(eigenstate):
        """
        <psi_{i,v}|S(k)|psi_i>
        """
        norm2 = eigenstate.norm2(sum=False)
        weight_s.append(norm2[:, all_s].sum(-1))
        weight_pxy.append(norm2[:, all_pxy].sum(-1))
        weight_pz.append(norm2[:, all_pz].sum(-1))
        weight_d.append(norm2[:, all_d].sum(-1))
        weight_hs.append(norm2[:, all_hs].sum(-1))
        weight_hpxy.append(norm2[:, all_hpxy].sum(-1))
        weight_hpz.append(norm2[:, all_hpz].sum(-1))
        return eigenstate.eig
        
    bs = BandStructure(H, [[0,0,0],[1,0,0]], 400, ['$\Gamma$', '$\Gamma$'])
    bsar = bs.apply.array
    eig = bsar.eigenstate(wrap=wrap_fat_bands).T
    
    linear_k, k_tick, k_label = bs.lineark(True)
    
    Emin, Emax = Erange
    dE = (Emax - Emin)/40.
    
    weight_s = np.array(weight_s).T
    weight_pxy = np.array(weight_pxy).T
    weight_pz = np.array(weight_pz).T
    weight_d = np.array(weight_d).T
    weight_hs = np.array(weight_hs).T
    weight_hpxy = np.array(weight_hpxy).T
    weight_hpz = np.array(weight_hpz).T
    
    plt.figure(figsize=figsize)
    plt.ylabel('$E-E_F$ [eV]')
    plt.xlim(linear_k[0], linear_k[-1])
    plt.xticks(k_tick, k_label)
    plt.ylim(Emin, Emax)
    
    for i, e_all_k in enumerate(eig):
        s_abs = np.abs(weight_s[i, :] * dE)
        pxy_abs = np.abs(weight_pxy[i, :] * dE)
        pz_abs = np.abs(weight_pz[i, :] * dE)
        d_abs = np.abs(weight_d[i, :] * dE)
        hs_abs = np.abs(weight_hs[i, :] * dE)
        hpxy_abs = np.abs(weight_hpxy[i, :] * dE)
        hpz_abs = np.abs(weight_hpz[i, :] * dE)
        plt.plot(linear_k, e_all_k, color='k')
        
        if np.any(pz_abs/dE>0.5): 
            # select k-points where pz makes major contribution
            where_pz = np.where(pz_abs/dE > 0.5)[0]
            
            # split the k-points array into seperate segments
            klist = [] # list of k-segments
            kseg = [] #  k-segments
            for i in range(len(where_pz)):
                if where_pz[i] - where_pz[i-1] > 1:
                    klist.append(kseg)
                    kseg = []
                kseg.append(where_pz[i])
            klist.append(kseg)

            for i in range(len(klist)):
                k_pz = linear_k[klist[i]]
                e = e_all_k[klist[i]]
                s = s_abs[klist[i]]
                pxy = pxy_abs[klist[i]]
                pz = pz_abs[klist[i]]
                d = d_abs[klist[i]]
                hs = hs_abs[klist[i]]
                hpxy = hpxy_abs[klist[i]]
                hpz = hpz_abs[klist[i]]
            
                # Full fat-band
                plt.fill_between(k_pz, e-dE, e+dE, color='k', alpha=0.1)
                # s
                plt.fill_between(k_pz, e-(s), e+(s), color='C0', alpha=0.5)
                # pxy
                plt.fill_between(k_pz, e+(s), e+(s+pxy), color='C1', alpha=0.5)
                plt.fill_between(k_pz, e-(s+pxy), e-(s), color='C1', alpha=0.5)
                # pz
                plt.fill_between(k_pz, e+(s+pxy), e+(s+pxy+pz), color='C2', alpha=0.5)
                plt.fill_between(k_pz, e-(s+pxy+pz), e-(s+pxy), color='C2', alpha=0.5)
                # d
                plt.fill_between(k_pz, e+(s+pxy+pz), e+(s+pxy+pz+d), color='C3', alpha=0.5)
                plt.fill_between(k_pz, e-(s+pxy+pz+d), e-(s+pxy+pz), color='C3', alpha=0.5)
                # hs
                plt.fill_between(k_pz, e+(s+pxy+pz+d), e+(s+pxy+pz+d+hs), color='C4', alpha=0.5)
                plt.fill_between(k_pz, e-(s+pxy+pz+d+hs), e-(s+pxy+pz+d), color='C4', alpha=0.5)
                # hpxy
                plt.fill_between(k_pz, e+(s+pxy+pz+d+hs), e+(s+pxy+pz+d+hs+hpxy), color='C5', alpha=0.5)
                plt.fill_between(k_pz, e-(s+pxy+pz+d+hs+hpxy), e-(s+pxy+pz+d+hs), color='C5', alpha=0.5)
                # hpz
                plt.fill_between(k_pz, e+(s+pxy+pz+d+hs+hpxy), e+(s+pxy+pz+d+hs+hpxy+hpz), color='C6', alpha=0.5)
                plt.fill_between(k_pz, e-(s+pxy+pz+d+hs+hpxy+hpz), e-(s+pxy+pz+d+hs+hpxy), color='C6', alpha=0.5)
      
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='C0', label='C: $s$'),
                       Patch(facecolor='C1', label='C: $p_x+p_y$'),
                       Patch(facecolor='C2', label='C: $p_z$'),
                       Patch(facecolor='C3', label='C: $d$'),
                       Patch(facecolor='C4', label='H: $hs$'),
                       Patch(facecolor='C5', label='H: $hp_x + hp_y$'),
                       Patch(facecolor='C6', label='H: $hp_z$')]
    plt.legend(handles=legend_elements, bbox_to_anchor=[1.1,0.9])
        
        
@timer
def plot_eigst_band(H, offset:list=[0], k=None, figsize=(15,5), dotsize=500):
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
    plt.figure(figsize=figsize)
    esnorm = es.sub(bands).norm2(sum=False).sum(0)
    plt.scatter(H.xyz[:,0], H.xyz[:,1],dotsize*esnorm)
    plt.axis('equal')        
        
        
@timer
def plot_eigst_energy(H, E=0.0, Ewidth=0.1, k=None, figsize=(15,5), dotsize=100):
    """
    Plot the eigenstates whose eigenvalues are in a specific range, by default around fermi level
    """
    
    es = H.eigenstate(k=k) if k else H.eigenstate()
    Emin = E-Ewidth/2
    Emax = E+Ewidth/2
    
    es = H.eigenstate()
    eig = H.eigh()
    sub = np.where(np.logical_and(eig>Emin, eig<Emax))
    es_sub = es.sub(sub)
    es_sub_norm = es_sub.norm2(sum=False).sum(0)
    
    plt.figure(figsize=figsize) 
    plt.scatter(H.xyz[:,0], H.xyz[:,1],dotsize*es_sub_norm)
    plt.axis('equal')        
        
        
@timer
def ldos(H, E=0.0, Ewidth=0.1, height=3.0, mesh=0.1, figsize=(15,5), colorbar=False):
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
    sub = np.where(np.logical_and(eig>Emin, eig<Emax))[0]
    
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
    
    bs = BandStructure(H, [[0,0,0],[0.5,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$','$\Gamma$'])
        
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
        raise TypeError('Requires the Brillouine zone object to contain a Hamiltonian')
        
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
            prd = dot(prd, prev.inner(second,diagonal=False))
            prev = second
        if gauge == 'r':
            g = contour.parent.geometry
            axis = contour.k[1] - contour.k[0]
            axis /= axis.dot(axis) ** 0.5
            phase = dot(g.xyz[g.o2a(_a.arangei(g.no)),:], dot(axis, g.rcell)).reshape(1,-1)
            prev.state *= np.exp(-1j*phase)
        prd = dot(prd, prev.inner(first, diagonal=False))
        return prd
    
    d = _zak(contour.apply.iter.eigenstate())
    ddet = det_destroy(d)
    result = -angle(ddet)
     
    return result



@timer
def inter_zak(H, offset=0):
    
    bs = BandStructure(H, [[0,0,0],[0.5,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$','$\Gamma$'])
        
    occ = [i for i in range(len(H.eig())//2)]
    if offset==0:
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
    
    bs = BandStructure(H, [[0,0,0],[0.5,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$','$\Gamma$'])
    
    gamma_inter = zak(bs, sub=occ, gauge='R')

    return round(gamma_inter, 10)



#@timer
def zak_band(H, occ):
    
    bs = BandStructure(H, [[0,0,0],[0.5,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$','$\Gamma$'])
        
    gamma_inter = zak(bs, sub=occ, gauge='R')

    return round(gamma_inter, 10)


def zak_dft(contour, sub=None, gauge='R'):
    
    from sisl import Hamiltonian, Overlap
    import sisl._array as _a
    from sisl.linalg import det_destroy
    import numpy as np
    from numpy import dot, angle, multiply, conj
    
    if not isinstance(contour.parent, Hamiltonian):
        raise TypeError('Requires the Brillouine zone object to contain a Hamiltonian')
    
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
            coords = g.xyz[g.o2a(_a.arangei(g.no)),:]
            phase = dot(coords, dot(axis, g.rcell)).reshape(1,-1)
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
        
    bs = BandStructure(H, [[0,0,0],[0.5,0,0],[1,0,0]], 400, ['$\Gamma$', '$X$','$\Gamma$'])
    
    d_det, gamma_inter = zak_dft(bs, sub=occ, gauge='R')

    return d_det, round(gamma_inter,10)



def get_zak_dict(H, method='from_top'):
    """
    Get the Zak phase list (by default, we mean intercellualr Zak phase)
    Format of list:
        {index: (bands_list, determinant, Zak_phase)}
    -method:
        'from_top': occupied bands counted from top (fermi) to bottom
        'from_bottom': occupied bands counted from bottom to top (fermi)
    """

    occ = np.where(H.eigh()<0)[0]

    zak_dict = {}
    
    for i in range(len(occ)):
        if method == 'from_top':
            sub = occ[-i-1:]
        elif method == 'from_bottom':
            sub = occ[:i+1]
        d, z = zak_band_dft(H, sub)
        zak_dict.update({i:(sub,d,z)})
    return zak_dict



def list_str(l):
    """
    Convert the list to a string and shorten it if it's too long
    """
    if len(l) <= 4:
        lnew = str(l)
    else:
        lnew = '[' + ','.join((f"{l[0]},{l[1]}", '...', f"{l[-2]},{l[-1]}")) + ']'
        
    return lnew




def plot_zak_polar(zdict):
    """
    Put Zak phase in a complex plane, which is a polar plot here
    """
    from matplotlib import cm

    plt.figure(figsize=(10,10))
    for i, v in zdict.items():
        bwr = cm.get_cmap('bwr')
        rho, r = np.angle(v[1]), np.abs(v[1])
        plt.polar([0,rho], [0,r], marker='o',
                  color=bwr(-r*np.cos(rho)+1),
                  label=list_str(v[0])+': '+str(round(r*np.cos(rho),2)),
                  alpha=r*0.8+0.05)    
        #plt.text(rho, r, str(i), color="red", fontsize=10)
        plt.legend(bbox_to_anchor=[1.2,0.9])





