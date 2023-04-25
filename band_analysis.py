from scipy.spatial import ConvexHull, Delaunay
import itertools
import os
import re
import regex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import xarray
from sisl import BandStructure, Hamiltonian, MonkhorstPack, Grid
from sisl.physics import electron, gaussian
from sisl.io import get_sile

import functools
from scipy.sparse import lil_matrix
from sisl.messages import deprecate
from .geometry import *
from .tools import *
from glob import glob
from matplotlib.patches import Patch
from typing import List, Tuple, Union, Dict

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'


kpoints_dict = {
    "G": ('$\Gamma$', [0, 0, 0]),
    "X": ('X', [0.5, 0., 0.]),
    "M": ('M', [0.5, 0.5, 0]),
    "K": ('K', [2.0 / 3, 1.0 / 3, 0]),
}


def read_dushin_out(file_path="./dushin.out"):
    """
    Calculate Huang-Rhys factors from dushin output file
    By default, the first state is ground state, the second one is excited state
    """
    results = {}
    with open(file_path, encoding='ISO-8859–1') as fout:
        line = fout.readline()
        while "Displacement" not in line:
            line = fout.readline()
        # Jump out from while loop after finding the word Displacement
        line = fout.readline()
        line = fout.readline()
        G_freq = []
        G_Q = []
        G_lam = []
        E_freq = []
        E_Q = []
        E_lam = []
        while True:
            l = line.strip().split()
            if len(l) == 0:
                break
            G_freq.append(l[3])
            G_Q.append(l[5])
            G_lam.append(l[7])
            E_freq.append(l[9])
            E_Q.append(l[11])
            E_lam.append(l[13])
            line = fout.readline()
        # Jump out from while loop if find blank line
        G_freq = np.array(G_freq)
        G_Q = np.array(G_Q)
        G_lam = np.array(G_lam)
        E_freq = np.array(E_freq)
        E_Q = np.array(E_Q)
        E_lam = np.array(E_lam)
        G_hrf = G_lam * 1.9863e-23 / (G_freq * 2.99792458e10 * 6.6260696e-34)
        E_hrf = E_lam * 1.9863e-23 / (E_freq * 2.99792458e10 * 6.6260696e-34)
        results["G_freq"] = G_freq
        results["G_Q"] = G_Q
        results["G_lam"] = G_lam
        results["E_freq"] = E_freq
        results["E_Q"] = E_Q
        results["E_lam"] = E_lam
        results["G_hrf"] = G_hrf
        results["E_hrf"] = E_hrf
        while True:
            line = fout.readline()
            if " Summs projected onto state" in line:
                # shouldn't be more than two blanks between works in the label
                line = list(filter(None, line.strip().split("   ")))
                info = "Ground state: {}, Excited state: {}".format(
                    line[-2], line[-1])
                results["info"] = info
            if "total reorg energies in eV" in line:
                line = list(filter(None, line.strip().split()))
                results[
                    "tot_reorg"
                ] = "Ground state: {} eV, Excited state: {} eV".format(
                    line[-2], line[-1]
                )
            if "Huang-Rhys factor" in line:
                line = list(filter(None, line.strip().split()))
                results["tot_hrf"] = "Ground state: {}, Excited state: {}".format(
                    line[-2], line[-1]
                )
                break  # This is the last line to read
    print(info)
    return results


def get_Hamiltonian_with_spin(H):
    """
    EXtract the spin up and down part of the Hamiltonian
    """
    csr_afm = [H.tocsr(dim=i) for i in range(H.dim)]
    S_afm = csr_afm.pop()
    H1 = Hamiltonian.fromsp(H, csr_afm[0], S_afm)
    H2 = Hamiltonian.fromsp(H, csr_afm[1], S_afm)
    return H1, H2


def num_occ(H, k=[0, 0, 0]):
    e = H.eigh(k)
    return e[e < 0].shape[0]


@timer
def band_structure(
    H,
    name=None,
    path="./opt",
    Erange=[-3, 3],
    index=False,
    figsize=(6, 4),
    tick_labels="XGX",
    fermi_energy=0.0,
    knpts=200,
    spin_polarized=False,
    legend_position=[1.1, 0.9],
    ticks_fontsize=14,
    label_fontsize=14,
    border_linewidth=1.5,
    linewidth=1,
    save=False,
    save_name='tmp',
    save_path='./',
    save_format='png,svg',
    **kwargs,
):
    """
    Arguments:
        name, path: if given, calcualte band structure from the DFT, otherwise,
            calculate from tight-binding model
        spin_polarized: spin polarized mode or not
        save: save the figure or not
        save_name: seedname of the saved figure
        save_path: path to save the figure
        save_format: format of the saved figure, by default save both png and svg
            parse the argument by separating with comma.
    """
    if name:
        tb = False
    else:
        tb = True
    tkls = list(tick_labels)
    # Position of ticks in Brillouin zone
    tks = []
    for i, v in enumerate(tick_labels):
        tkls[i] = kpoints_dict[v][0]
        tks.append(kpoints_dict[v][1])

    bsar = []  # BandStructure array object list
    if spin_polarized:
        # Define figure properties
        label = ['spin up', 'spin down']
        linestyle = ['-', '--']
        color = ['r', 'b']
        # Extract Hamiltonian matrix
        H1, H2 = get_Hamiltonian_with_spin(H)
        name_up = name+'_up'
        name_dn = name+'_dn'
        # Get band structure data from Hamiltonian
        bs1 = BandStructure(H1, tks, knpts, tkls)
        bsar1 = bs1.apply.array
        bsar.append(bsar1)
        bs2 = BandStructure(H2, tks, knpts, tkls)
        bsar2 = bs2.apply.array
        bsar.append(bsar2)
        # linear k mesh, k ticks, k tick labels
        lk, kt, kl = bs1.lineark(True)
        eigfile = [os.path.join(path, f'{n}.eig.{tick_labels}{knpts}.txt') for
                   n in [name_up, name_dn]]
    else:
        label = [None]
        linestyle = ['-']
        color = [kwargs['color']] if 'color' in kwargs.keys() else [None]
        bs = BandStructure(H, tks, knpts, tkls)
        bsar.append(bs.apply.array)
        # linear k mesh, k ticks, k tick labels
        lk, kt, kl = bs.lineark(True)
        eigfile = [os.path.join(path, f"{name}.eig.{tick_labels}{knpts}.txt")]
    # try to read eigenvalues from file, if not exist then create one
    eigh = []

    # If it's in tight binding mode, then calculate the bands
    if tb:
        eigh.append(bsar[0].eigh())
    else:
        for i in range(len(bsar)):
            try:
                with open(eigfile[i]) as f:
                    eig = np.loadtxt(f)
                assert len(eig) != 0
            except:
                print(
                    f"eig file {eigfile[i]} not found or empty. Now calculate new eig")
                eig = bsar[i].eigh()
                # Usually the eigenvalues are shifted to Fermi energy by sisl already
                eig -= fermi_energy
                with open(eigfile[i], "w") as f:
                    np.savetxt(f, eig)
            eigh.append(eig)

    plt.figure(figsize=figsize)
    plt.xticks(kt, kl)
    plt.ylim(Erange[0], Erange[-1])
    plt.xlim(0, lk[-1])
    plt.ylabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsize)
    plt.tick_params(axis='x', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', labelsize=ticks_fontsize)

    # iterate spin
    for i, e in enumerate(eigh):
        # iterate bands
        for j, ek in enumerate(e.T):
            lb = label[i] if j == 0 else None

            # plot band only if the band has at least one value in the energy range
            if not np.any((Erange[0] < ek) & (ek < Erange[1])):
                continue
            plt.plot(lk, ek, linestyle=linestyle[i], color=color[i],
                     label=lb, linewidth=linewidth)
            # mark the band index
            if index:
                if Erange[0] < ek[-1] < Erange[1]:
                    plt.annotate(j + 1, (lk[-1], ek[0]), color=color[i])
    if spin_polarized:
        lgd = plt.legend(bbox_to_anchor=legend_position)
        lgd.get_frame().set_alpha(0)

    for border in ["left", "bottom", "top", "right"]:
        plt.gca().spines[border].set_linewidth(border_linewidth)

    if save:
        if tb:
            model = 'TB'
        else:
            model = 'DFT'
        for f in save_format.split(','):
            filename = os.path.join(
                save_path, 
                f'{save_name}.BandStructure.{model}.{tick_labels}.E{Erange[0]}to{Erange[-1]}.{f}')
            plt.savefig(filename, bbox_inches='tight',
                        dpi=300, transparent=True)
    plt.show()


def read_bands(
    name, path="./opt", as_dataarray=True, squeeze=True
) -> xarray.DataArray:
    """
    Read band structure from name.bands file
    """

    bands_path = os.path.join(path, f"{name}.bands")
    # read data using methods in sisl
    bandsile = get_sile(bands_path)
    bands = bandsile.read_data(as_dataarray=as_dataarray)
    # remove redundant dimension of data
    if squeeze:
        bands = bands.squeeze()
    try:
        if bands.ticklabels[0] == "Gamma":
            bands.ticklabels[0] = "$\Gamma$"
        if bands.ticklabels[1] == "X":
            bands.ticklabels[1] = "$X$"
        bands.k.data[:] *= 1.8897259886
        bands.ticks[:] = np.array(bands.ticks)*1.8897259886
    except:
        pass
    return bands


@timer
def plot_bands(name, path,
               Erange=[-3, 3],
               figsize=[6, 4],
               index=False,
               ticklabels=None,
               ticks_fontsize=14,
               label_fontsize=14,
               border_linewidth=1.5,
               linewidth=1,
               save=False,
               save_path='./',
               save_name='tmp',
               save_format='png,svg',
               color='k',
               **kwargs
               ):
    """
    Plot the band structure from .bands file
    """
    bands = read_bands(name, path, squeeze=False)
    if ticklabels:
        bands.ticklabels[:] = ticklabels

    for i in range(len(bands.ticklabels)):
        if bands.ticklabels[i] == 'Gamma':
            bands.ticklabels[i] = '$\Gamma$'

    ks = bands.k.data
    plt.figure(figsize=figsize)
    # set y axis limit if Energy range is given
    if Erange:
        emin, emax = Erange
        plt.ylim(emin, emax)
    else:
        emin, emax = -1e2, 1e6
    plt.ylabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsize)
    plt.xticks(bands.ticks, bands.ticklabels)
    plt.xlim([min(bands.ticks), max(bands.ticks)])
    plt.tick_params(axis='x', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', labelsize=ticks_fontsize)
    # plot the data
    for i in range(bands.shape[2]):  # iterate bands
        band = bands[:, :, i]
        # spin unpolarized
        if bands.shape[1] == 1:  # spin index
            # select the bands that are in the given energy window
            if np.any(np.logical_and(band > emin, band < emax)):
                plt.plot(ks, band[:, 0], color=color,
                         linewidth=linewidth)
            # mark the band index
            if index & (Erange[0] < band[-1, 0] < Erange[1]):
                plt.annotate(i + 1, (band.ticks[-1], band[-1, 0]))
        # spin polarized
        elif bands.shape[1] == 2:
            # select the bands that are in the given energy window
            if np.any(np.logical_and(band > emin, band < emax)):
                plt.plot(ks, band[:, 0], color='r', linestyle='-',
                         linewidth=linewidth)
                plt.plot(ks, band[:, 1], color='b', linestyle='--',
                         linewidth=linewidth)
                # mark the band index
                if index & (Erange[0] < band[-1, 0] < Erange[1]):
                    plt.annotate(
                        i + 1, (band.ticks[-1], band[-1, 0]), color='r')
                if index & (Erange[0] < band[-1, 1] < Erange[1]):
                    plt.annotate(
                        i + 1, (band.ticks[-1], band[-1, 1]), color='b')
    for border in ["left", "bottom", "top", "right"]:
        plt.gca().spines[border].set_linewidth(border_linewidth)

    if bands.shape[1] == 2:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='r', linestyle='-'),
                        Line2D([0], [0], color='b', linestyle='--')]
        lgd = plt.legend(custom_lines, ['spin up', 'spin down'],
                   bbox_to_anchor=[1.1, 0.9])
        lgd.get_frame().set_alpha(0)

    if save:
        tkl = bands.ticklabels
        tkl = ''.join([s.strip('$\\')[0] for s in tkl])
        # nkpts = bands.k.shape[0]
        for f in save_format.split(','):
            filename = os.path.join(
                save_path, f'{save_name}.BandStructure.DFT.{tkl}.E{Erange[0]}to{Erange[-1]}.{f}')
            plt.savefig(filename, bbox_inches='tight',
                        dpi=300, transparent=True)
    plt.show()


@timer
def interpolated_bs(
    name,
    path_int,
    H_int=None,
    path_pr=None,
    H_pr=None,
    Erange=[-5, 0],
    figsize=(6, 4),
    tick_labels="XGX",
    overlap=False,
    knpts_int=30,
    knpts_pr=200,
    ticks_fontsize=14,
    label_fontsize=14,
    border_linewidth=1.5,
    linewidth=1,
    marker_size=30,
    marker_color="g",
    marker="o",
    facecolors="none",
    save=False,
    save_path='./',
    save_name='tmp',
    save_format='png,svg',
    **kwargs,
):
    """
    PLot the interpolated band structure from Wannier90 output file
    Arguments:
        name: name of the system
        path_int: path to the system
        H_int: interpolated Hamiltonian
        path_pr: path to the pristine system
        H_pr: pristine Hamiltonian
        Erange: energy range to plot
        figsize: figure size
        tick_labels: labels of the k-points
        overlap: whether to overlap the pristine and interpolated band structures
        knpts_int: number of k-points for the interpolated band structure   
        knpts_pr: number of k-points for the pristine band structure
        ticks_fontsize: font size of the ticks
        label_fontsize: font size of the labels
        border_linewidth: linewidth of the border
        linewidth: linewidth of the band structure
        marker_size: size of the marker
        marker_color: color of the marker
        marker: marker type
        facecolors: facecolor of the marker
        save: whether to save the figure
        save_path: path to save the figure
        save_name: name of the figure
        save_format: format of the figure
    """
    if H_int:
        ham_int = H_int
    else:
        win_path = os.path.join(path_int, name + ".win")
        fwin = get_sile(win_path)
        ham_int = fwin.read_hamiltonian()

    tkls = list(tick_labels)
    tks = []
    for i, v in enumerate(tick_labels):
        tkls[i] = kpoints_dict[v][0]
        tks.append(kpoints_dict[v][1])

    if not overlap:
        knpts_int = knpts_pr
    else:
        if (not H_pr) or (not path_pr):
            raise ValueError(
                "Please provide the pristine Hamiltonian H_pr and path_pr")
        bs_pr = BandStructure(H_pr, tks, knpts_pr, tkls)

        eigfile = os.path.join(
            path_pr, f"{name}.eig.{tick_labels}{knpts_pr}.txt")
        # try to read eigenvalues from file, if not exist then create one
        try:
            with open(eigfile) as f:
                eig_pr = np.loadtxt(f)
            assert len(eig_pr) != 0
        except:
            print(
                f"eig file {eigfile} not found or empty. Now calculate new eig")
            bsar_pr = bs_pr.apply.array
            eig_pr = bsar_pr.eigh()
            with open(eigfile, "w") as f:
                np.savetxt(f, eig_pr)
        lk_pr = bs_pr.lineark(ticks=False)

    bs_int = BandStructure(ham_int, tks, knpts_int, tkls)
    bsar_int = bs_int.apply.array
    eig_int = bsar_int.eigh()
    fe = read_final_energy(name=name, path=path_int, which="fermi")
    eig_int -= fe

    lk_int, kt, kl = bs_int.lineark(ticks=True)
    plt.figure(figsize=figsize)
    plt.xticks(kt, kl, fontsize=label_fontsize)
    plt.ylim(Erange[0], Erange[-1])
    plt.ylabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsize)

    if overlap:
        for i, ek_pr in enumerate(eig_pr.T):
            plt.plot(lk_pr, ek_pr, color="k", linewidth=linewidth, **kwargs)
        for j, ek_int in enumerate(eig_int.T):
            plt.scatter(
                lk_int,
                ek_int,
                s=marker_size,
                marker=marker,
                facecolors=facecolors,
                edgecolors=marker_color,
            )
        plt.xlim(0, lk_pr[-1])
    else:
        for i, ek_int in enumerate(eig_int.T):
            plt.plot(lk_int, ek_int, **kwargs)
        plt.xlim(0, lk_int[-1])
    plt.tick_params(axis='x', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', labelsize=ticks_fontsize)
    # set the linewidth of the border
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(border_linewidth)

    if save:
        for fmt in save_format.split(','):
            plt.savefig(os.path.join(save_path, 
                f'{save_name}.InterpolatedBandStructure.{tick_labels}.E{Erange[0]}to{Erange[-1]}.{fmt}'),
                dpi=300,
                transparent=True,
                bbox_inches='tight')
    plt.show()



@timer
def unfold_band(
    H,
    lat_vec=1.0,
    Erange=None,
    kmesh=500,
    ky=0,
    marker_size: float = None,
    marker_size_range=[2, 10],
    cmap="Reds",
    fermi_energy=0.0,
    ring=False,
    **kwargs,
):
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
    if ring:
        xyz = xyz - np.mean(xyz, axis=0)
        # calculate the circumferential length of the arc and convert it into a
        # straight vector
        angle = np.arctan(xyz[:, 1] / xyz[:, 0])
        r = np.linalg.norm(xyz[0, :])
        xyz = np.vstack(
            (r * angle, np.zeros(len(angle)), np.zeros(len(angle)))).T
    # band lines scale
    bdlscale = np.pi / lat_vec
    G = H.rcell[0, 0]
    for red_k in np.linspace(-1, 1, kmesh):
        k = red_k * bdlscale
        k_vec = np.array([k, ky, 0])
        red_K = k / G
        if red_K > 0:
            red_K = red_K - np.floor(red_K)
        else:
            red_K = red_K - np.ceil(red_K)
        red_K_vec = np.array([red_K, 0, 0])
        # k vectors use reduced one here
        eigh = H.eigh(k=red_K_vec)
        eigenstate = H.eigenstate(k=red_K_vec).state
        phase_left = np.exp(xyz.dot(k_vec) * 1j)
        phase_right = np.exp(xyz.dot(k_vec) * (-1j))
        phase = np.outer(phase_left, phase_right)
        weight = (1 / N) * np.conj(eigenstate).dot(phase).dot(eigenstate.T)
        weight = np.abs(weight).diagonal()
        weight = weight / weight.max()
        kpts = np.repeat(k, N)
        if not marker_size:
            smin, smax = marker_size_range
            msize = weight * (smax - smin) + smin
        else:
            msize = marker_size
        plt.scatter(kpts, eigh - fermi_energy, s=msize,
                    c=weight, cmap=cmap, **kwargs)
        plt.ylabel("$E-E_F (eV)$")
        plt.xlabel("Wavenumber ($1 /\AA$)")
        if Erange:
            plt.ylim(Erange[0], Erange[1])
        plt.xlim(-bdlscale, bdlscale)


@timer
def band_gap(H, name=None, path="./opt", tb=True):

    if name:
        tb = False

    rlv = np.int0(~(np.array(H.nsc) == 1))
    bs = BandStructure(
        H, [[0, 0, 0], list(0.5 * rlv), list(rlv)
            ], 200, ["$\Gamma$", "X", "$\Gamma$"]
    )
    bsar = bs.apply.array
    # try to read eigenvalues from file
    if tb:
        eig = bsar.eigh()
    else:
        try:
            files = glob(os.path.join(path, f"{name}.eig*"))
            assert len(files) != 0
            # make sure the k path of eig file contains at least from Gamma to X
            for file in files:
                if ("GX" in file) or ("XG" in file):
                    eigfile = file
                    break
            # if eigfile doesn't exist it will call Name Error
            with open(eigfile) as f:
                eig = np.loadtxt(f)
            assert len(eig) != 0
        except:
            # maybe NotImplementedError, FileNotFoundError, AssertionError, or NameError
            print(f"eig file(s) not found or empty. Now calculate new eig")
            eigfile = os.path.join(path, f"{name}.eig.GXG200.txt")
            eig = bsar.eigh()

    bg = functools.reduce(
        lambda x, y: x if x <= y else y,
        (ek[ek > 0].min() - ek[ek < 0].max() for ek in eig),
    )
    return bg


def chain_hamiltonian(Huc, nc: int, rotate_geom=True, **kwargs):
    """
    Contruct chain hamiltonian from unit cell hamiltonian.
    This method build hamiltonian matrix by calculating sparse matrix, different from construct_hamiltonian method. It is to deal with the interpolated Wannier function tight binding basis hamiltonian that is read from _hr.dat file.
    Arguments:
        Huc: Hamiltonian of the unit cell
        nc: number of repeated cells
    """
    geom = Huc.geometry
    if rotate_geom:
        geom = rotate_gnr(geom)
    nsc = geom.nsc
    no = geom.no
    nsc0 = nsc[0]  # we focus on one-dimensional system
    if not nsc0 % 2:
        raise ValueError("nsc must be odd number")
    cutoff = kwargs.get("cutoff", 0.00001)
    huc = copy_hamiltonian(Huc)
    huc = np.squeeze(huc)

    # temporary zeros array
    zeros = np.zeros((nc, nc))
    # final Hamiltonian
    Ham = 0

    for i in range(nsc0):
        # index of super cell
        isc = int(i - (nsc0 - 1) / 2)
        # index of isc in unit cell hamiltonain
        sc_idx = geom.sc_index([isc, 0, 0])
        ham = huc[:, no * sc_idx: no * sc_idx + no]

        if isc == 0:
            onem = np.eye(nc)
        elif isc < 0:
            onem = zeros.copy()
            if nc >= -isc + 1:
                onem[-isc:, :isc] = np.eye(nc - abs(isc))
        elif isc > 0:
            onem = zeros.copy()
            if nc >= isc + 1:
                onem[:-isc, isc:] = np.eye(nc - abs(isc))
        Ham += np.kron(onem, ham)

    del zeros

    no_tot = no * nc
    spm = lil_matrix((no_tot, no_tot))
    for i in range(no_tot):
        for j in range(no_tot):
            if abs(Ham[i, j]) > cutoff:
                spm[i, j] = Ham[i, j]
    chain = geom.tile(nc, 0)
    chain.cell[0, 0] += 10
    chain = move_to_xycenter(chain, plot_geom=False)
    chain.set_nsc([1, 1, 1])
    H = Hamiltonian.fromsp(chain, spm)
    return H


@timer
def energy_levels(
    H,
    Erange=[-3, 3],
    figsize=(1, 5),
    index=False,
    color="k",
    fermi_energy=0.0,
    show=True,
    **kwargs,
):
    """
    should be a single molecule, nsc=[1,1,1]
    """
    eig = H.eigh()
    plt.figure(figsize=figsize)
    # use Hermitian solver, read values
    plt.hlines(eig - fermi_energy, 0, 1, color=color, **kwargs)
    plt.ylim(Erange[0], Erange[-1])
    plt.ylabel(r'$\rm E-E_F$ (eV)')
    plt.xticks([])

    if index:  # label the index of energy levels
        which = np.where(np.logical_and(eig < Erange[-1], eig > Erange[0]))[0]
        for i in which:
            plt.text(1.2, eig[i], str(i))
    if show:
        plt.show()


@timer
def dos(
    H,
    name=None,
    path="./opt",
    tb=True,
    Erange=[-3, 3],
    figsize=(2, 4),
    dE=0.01,
    ret=False,
    color="k",
    mpgrid=[50, 1, 1],
    gaussian_broadening=0.02,
    linewidth=1,
    border_linewidth=1.5,
    ticks_fontsize=14,
    label_fontsize=14,
    save=False,
    save_path='.',
    save_format='png,svg',
    save_name='tmp',
    **kwargs,
):

    if name:
        tb = False

    # Estimate the whole energy range by energy at Gamma point
    # Calculate DOS for whole energy range, then select to show certain range
    eg = H.eigh()
    E0, E1 = Erange
    emin = min(-30, E0)
    emax = max(30, E1)
    E = np.arange(emin, emax, dE)
    def num2str(x): return "m" + str(x)[1:] if x < 0 else str(x)
    erangestr = "{}to{}".format(num2str(emin), num2str(emax))
    # Firstly try to read dos from file
    # If the band structure undergoes big change, remember to remove the dos file
    # because the following code doesn't detect the change in Hamiltonian
    mp1, mp2, mp3 = mpgrid
    dosfile = os.path.join(
        path,
        f"{name}.dos.{erangestr}.dE{dE}.broaden{gaussian_broadening}.mpgrid{mp1}_{mp2}_{mp3}.txt",
    )
    mp = MonkhorstPack(H, mpgrid)
    dis = functools.partial(gaussian, sigma=gaussian_broadening)
    mpav = mp.apply.average

    def wrap_DOS(eigenstate):
        # Calculate the DOS for the eigenstates
        DOS = eigenstate.DOS(E, distribution=dis)
        return DOS

    if tb:
        dos = mpav.eigenstate(wrap=wrap_DOS, eta=True)
    else:
        try:
            with open(dosfile) as f:
                dos = np.loadtxt(f)
            assert len(dos) != 0
        except:
            print(
                f"dos file {dosfile} not found or empty. Now calculate new dos")
            dos = mpav.eigenstate(wrap=wrap_DOS, eta=True)
            # write to file
            with open(dosfile, "w") as f:
                np.savetxt(f, dos)

    plt.figure(figsize=figsize)
    select_range = np.logical_and(E >= Erange[0], E <= Erange[-1])
    sel_dos = dos[select_range]
    sel_E = E[select_range]
    plt.ylim(Erange[0], Erange[-1])
    plt.xlim(0, sel_dos.max()*1.1)
    plt.xticks([])
    plt.plot(sel_dos, sel_E, color=color, linewidth=linewidth, **kwargs)
    plt.ylabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsize)
    plt.xlabel("DOS (arb. u.)", fontsize=label_fontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    # set border linewidth
    for spine in plt.gca().spines.values():
        spine.set_linewidth(border_linewidth)
    
    if save:
        for fmt in save_format.split(','):
            filename = os.path.join(save_path, 
                f'{save_name}.DOS.{erangestr}.dE{dE}.broaden{gaussian_broadening}.mpgrid{mp1}_{mp2}_{mp3}.{fmt}')
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

    if ret:
        return sel_dos


@timer
def pdos(
    H,
    name,
    path="./opt",
    Erange=[-5, 5],
    figsize=(4, 6),
    dE=0.01,
    gaussian_broadening=0.02,
    mpgrid=[50, 1, 1],
    projected_atoms="all",
    projected_orbitals=["s", "pxy", "pz"],
    specify_atoms_and_orbitals=None,
    legend_position=[1.1, 0.9],
    colors:list=None,
    linewidth=1,
    border_linewidth=1.5,
    ticks_fontsize=14,
    label_fontsize=14,
    save=False,
    save_name='tmp',
    save_path='.',
    save_format='png,svg',
    max_pdos=None,
    **kwargs
):
    """
    Projected density of states
    by default plot selected projected orbitals for all the atom specie
    Arguments:
        H: Hamiltonian
        name: name of the system
        path: path to the directory where the pdos file is stored
        Erange: Energy range to plot.
        figsize: figure size
        dE: Energy mesh size.
        gaussian_broadening: gaussian broadening parameter
        mpgrid: Monkhorst-Pack grid
        project_atoms: atoms to be projected, by default all atoms.
        project_orbitals: orbitals to be projected.
        specify_atoms_and_orbitals: should follow the following format:
            'C: pz; N: pxy'. If the specify_atoms_and_orbitals argument is
            not None, then it will overwrite the projected_atoms and
            projected_orbitals arguments. If not, the projected orbitals will
            be all the orbitals in projected_orbitals of each atoms in
            projected_atoms.
        legend_position: position of the legend
        colors: colors of the projected orbitals
        linewidth: linewidth of the pdos
        border_linewidth: linewidth of the border
        ticks_fontsize: font size of the ticks
        label_fontsize: font size of the labels
        save: whether to save the figure
        save_name: name of the figure
        save_path: path to save the figure
        save_format: format of the figure
        max_pdos: maximum value of the pdos, to be used to set the xlim. This is
            useful when the pdos x range is to be consistent with other pdos.
    """
    # Firstly try to read pdos from file
    # The following part is similar to dos
    geom = H.geometry
    E0, E1 = Erange
    emin = min(-10, E0)
    emax = max(10, E1)
    E = np.arange(emin, emax, dE)
    def num2str(x): return "m" + str(x)[1:] if x < 0 else str(x)
    erangestr = "{}to{}".format(num2str(emin), num2str(emax))
    mp1, mp2, mp3 = mpgrid
    pdosfile = os.path.join(
        path,
        f"{name}.pdos.{erangestr}.dE{dE}.broaden{gaussian_broadening}.mpgrid{mp1}_{mp2}_{mp3}.npz",
    )
    pdos_dict = {}  # final data
    pdos_dict_temp = {}  # temporary data used for averaging in the wrap function
    try:
        # pdos dictionary with compacted keys
        pdos_dict_comp = np.load(pdosfile)
        assert len(pdos_dict_comp.files) != 0
        # recover the pdos dictionary with two layers of keys
        for key in pdos_dict_comp.files:
            a, orb = key.split(":")
            if not a in pdos_dict.keys():
                pdos_dict[a] = {}
            pdos_dict[a][orb] = pdos_dict_comp[key]
    except:
        print(f"pdos file {pdosfile} not found or empty. Calculating new pdos")
        mp = MonkhorstPack(H, mpgrid)
        mpav = mp.apply.average
        # index of all the orbitals of each atom species
        orb_idx_dict = get_orb_list(geom)

        def wrap(eigenstate):
            PDOS = eigenstate.PDOS(E, distribution=dis).squeeze()
            nonlocal pdos_dict_temp
            for a, all_idx in orb_idx_dict.items():
                pd_a = {}
                # calculation all the orbitals
                for orb in ["s", "pxy", "pz", "d", "f"]:
                    if all_idx[orb].size != 0:  # if it's not empty
                        pd_o = PDOS[all_idx[orb], :].sum(0)
                        pd_a.update({orb: pd_o})
                # the wrap function iterates over all the k points
                # so pdos_dict_temp will be overwrite in each iteration
                pdos_dict_temp.update({a: pd_a})
            return np.stack([v for vs in pdos_dict_temp.values() for v in vs.values()])

        E = np.arange(emin, emax, dE)
        # calculate the pdos, however, this pdos as single array is not as
        # useful as the dictionary version. So it won't be used.
        dis = functools.partial(gaussian, sigma=gaussian_broadening)
        pDOS = mpav.eigenstate(wrap=wrap, eta=True)
        # Convert the final array pDOS into dictionary format
        pdos_dict = pdos_dict_temp.copy()  # copy the structure then change the content
        i = 0
        for a, pd_a in pdos_dict.items():
            for orb, pd_o in pd_a.items():
                pd_a[orb] = pDOS[i, :]
                i += 1
        del i
        # convert the dictionary to compacted version
        pdos_dict_comp = {}
        for a, pd_a in pdos_dict.items():
            for orb, pd_o in pd_a.items():
                newkey = a + ":" + orb
                pdos_dict_comp[newkey] = pd_o
        # save to file
        np.savez(pdosfile, **pdos_dict_comp)

    # Plot pdos
    # define list of linestyle, to distinguish atoms
    linestyle_list = [
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
        (0, (1, 7)),  # loosely dotted
        (0, (5, 7)),  # loosely dashed
        (0, (3, 7, 1, 7)),  # loosely dashdot
    ]
    # define list of colors, to distinguish orbitals, they correspond to
    # s, pxy, pz, d, f respectively.
    color_list = ["C0", "C1", "C2", "C3", "C4"]

    orb_idx_dict = get_orb_list(geom)
    if not specify_atoms_and_orbitals:
        if projected_atoms == "all":
            proj_atoms = list(orb_idx_dict.keys())
        elif isinstance(projected_atoms, (list, tuple)):
            proj_atoms = projected_atoms
        if projected_orbitals == "all":
            proj_orbs = ["s", "pxy", "pz", "d", "f"]
        elif isinstance(projected_orbitals, (list, tuple)):
            proj_orbs = projected_orbitals
        proj_ats_orbs = dict(zip(proj_atoms, [proj_orbs] * len(proj_atoms)))
    else:
        proj_ats_orbs = convert_formated_str_to_dict(
            specify_atoms_and_orbitals)

    plt.figure(figsize=figsize)
    plt.ylabel("$E - E_F$ (eV)", fontsize=label_fontsize)
    plt.xlabel("PDOS (arb. u.)", fontsize=label_fontsize)
    ia = 0  # index of atoms, for line styles
    pdosmax = 0  # max of pdos
    iline = 0  # index of line styles

    if colors:
        color_gen = itertools.cycle(colors)

    for a, pd_a in pdos_dict.items():
        io = 0  # index of orbitals, for colors
        # choose atoms
        if a not in proj_ats_orbs.keys():
            # skip this atom
            ia += 1
            continue
        for orb, pd_o in pd_a.items():
            # choose orbitals
            if orb in proj_ats_orbs[a]:
                if orb == "pxy":
                    label = f"{a}: $p_x+p_y$"
                elif orb == "pz":
                    label = f"{a}: $p_z$"
                else:
                    label = f"{a}: ${orb}$"
                pd = pdos_dict[a][orb]
                # select the given range to plot
                select_range = np.logical_and(E >= Erange[0], E <= Erange[-1])
                sel_pd = pd[select_range]
                sel_E = E[select_range]
                if colors:
                    c = next(color_gen)
                else:
                    c = color_list[io]
                plt.plot(
                    sel_pd,
                    sel_E,
                    color=c,
                    label=label,
                    linestyle=linestyle_list[iline],
                    linewidth=linewidth,
                    **kwargs
                )
                pdmax = sel_pd.max()
                pdosmax = pdmax if pdmax > pdosmax else pdosmax
            io += 1
        iline += 1
        ia += 1
    plt.ylim(Erange[0], Erange[-1])
    plt.xlim(0, pdosmax*1.02)
    if max_pdos:
        plt.xlim(0, max_pdos*1.02)
    plt.xticks([])
    lgd = plt.legend(bbox_to_anchor=legend_position)
    lgd.get_frame().set_alpha(0)
    plt.tick_params(axis='x', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', labelsize=ticks_fontsize)
    for border in ["left", "bottom", "top", "right"]:
        plt.gca().spines[border].set_linewidth(border_linewidth)

    if save:
        for fmt in save_format.split(','):
            projs_dict = convert_dict_to_formatted_str(proj_ats_orbs)
            projs = projs_dict.replace(',','-').replace(':','-')
            plt.savefig(
                os.path.join(save_path, 
                f'{save_name}.PDOS.{projs}.E{Erange[0]}to{Erange[1]}.{fmt}'),
                bbox_inches='tight',
                dpi=300, transparent=True)
    plt.show()
    return pdosmax


@timer
def pzdos(
    H, Erange=[-10, 20], mpgrid=[50, 1, 1], gaussian_broadening=0.05, plot_pzdos=True
):

    mp = MonkhorstPack(H, mpgrid)
    mpav = mp.apply.average

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
    dis = functools.partial(gaussian, sigma=gaussian_broadening)
    pDOS = mpav.PDOS(E, wrap=wrap, distribution=dis)
    if plot_pzdos:
        for i, label in enumerate(all_pz):
            plt.plot(E, pDOS[i, :], label=label)
        plt.xlim(E[0], E[-1])
        plt.ylim(0, None)
        plt.xlabel(r"$\rm E - E_F$ (eV)")
        plt.ylabel(r"pz_DOS [1/eV]")
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


@timer
def fat_bands(
    H,
    name,
    path="./opt",
    Erange=(-10, 10),
    figsize=(6, 10),
    knpts=200,
    tick_labels="XGX",
    split_view=False,
    projected_atoms="all",
    projected_orbitals=["s", "pxy", "pz"],
    specify_atoms_and_orbitals=None,
    index=False,
    legend_position=[1.2, 0.9],
    alpha=1.0,
    fat_scale = 1,
    tick_fontsize=14,
    label_fontsizesize=14,
    legend_fontsize=12,
    linewidth=1,
    border_linewidth=1.5,
    save=False,
    save_name=None,
    save_format='png,svg',
    save_path='.',
    show=True,
    colors:list=None
):
    """
    Plot the fat bands, showing the weight of each kinds of orbital of every band.
    Arguments:
        H: Hamiltonian object.
        name: name of the system.
        path: path to the directory where the fat bands are saved.
        Erange: energy range to plot.
        figsize: figure size.
        knpts: number of kpoints to plot.
        tick_labels: By default the fat bands for different atoms are plot in
            one figure and the ticks are from X to G to X. For split view the
            ticks are from G to X for each subplot by default.
        split_view: show fat bands for different atomic species in differetn
            subplots.
        projected_atoms: the atoms that you want to project on. If it is
            'all', then the projected atoms will be all the atoms in the
            system.
        projected_orbitals: the orbitals that you want to project on. If it
            is 'all', then the projected orbitals will be all the orbitals
            in the system.
        specify_atoms_and_orbitals: should follow the following format:
            'C: pz; N: pxy'. If the specify_atoms_and_orbitals argument is
            not None, then it will overwrite the projected_atoms and
            projected_orbitals arguments. If not, the projected orbitals will
            be all the orbitals in projected_orbitals of each atoms in
            projected_atoms.
        index: whether to show the index of the bands.
        legend_position: the position of the legend.
        alpha: the transparency of the fat bands.
        fat_scale: defines how fat the bands are.
        tick_fontsize: the fontsize of the ticks.
        label_fontsizesize: the fontsize of the labels.
        legend_fontsize: the fontsize of the legend.
        save: whether to save the figure.
        save_name: the name of the saved figure.
        save_format: the format of the saved figure.
        save_path: the path to save the figure.
        show: whether to show the figure.
        colors: the colors of the fat bands.
    """

    # Position of ticks in Brillouin zone
    tks = []
    tkls = list(tick_labels)
    for i, v in enumerate(tick_labels):
        tkls[i] = kpoints_dict[v][0]
        tks.append(kpoints_dict[v][1])
    bs = BandStructure(H, tks, knpts, tkls)
    geom = H.geometry
    orb_idx_dict = get_orb_list(geom)
    # Generate the atoms and corresponding orbitals that you want to project on
    if not specify_atoms_and_orbitals:
        if projected_atoms == "all":
            proj_atoms = list(orb_idx_dict.keys())
        elif isinstance(projected_atoms, (list, tuple)):
            proj_atoms = projected_atoms
        if projected_orbitals == "all":
            proj_orbs = ["s", "pxy", "pz", "d", "f"]
        elif isinstance(projected_orbitals, (list, tuple)):
            proj_orbs = projected_orbitals
        proj_ats_orbs = dict(zip(proj_atoms, [proj_orbs] * len(proj_atoms)))
    else:
        proj_ats_orbs = convert_formated_str_to_dict(
            specify_atoms_and_orbitals)

    # Try to read fatbands from file
    fbwtfile = os.path.join(
        path, f"{name}.fatbands.weight.{tick_labels}{knpts}.npz")
    fbeigfile = os.path.join(
        path, f"{name}.fatbands.eig.{tick_labels}{knpts}.txt")
    wt_dict = {}
    try:
        wt_dict_comp = np.load(fbwtfile)
        assert len(wt_dict_comp.files) != 0
        for key in wt_dict_comp.files:
            a, orb = key.split(":")
            if a not in wt_dict.keys():
                wt_dict[a] = {}
            wt_dict[a][orb] = wt_dict_comp[key]
        eig = np.loadtxt(fbeigfile)
        assert len(eig) != 0
    except:
        print(f"fatbands files not found or empty. Now calculate new fatbands")
        # initialize the weight dictionary
        for a, orbs in orb_idx_dict.items():
            wt_dict[a] = {}
            for orb in orbs.keys():
                wt_dict[a][orb] = []
        bsar = bs.apply.array

        def wrap_fat_bands(eigenstate):
            """
            <psi_{i,v}|S(k)|psi_i>
            return the eigenvalue for a specify eigenstat and calculate
            the weight for each orbitals.
            """
            nonlocal wt_dict
            norm2 = eigenstate.norm2(sum=False)
            # calculate the weight for every kind of atom and orbital
            for a, orbs in orb_idx_dict.items():
                for orb, indices in orbs.items():
                    if len(indices) != 0:
                        wt_k = norm2[:, indices].sum(-1)
                        wt_dict[a][orb].append(wt_k)
            return eigenstate.eig

        eig = bsar.eigenstate(wrap=wrap_fat_bands, eta=True)
        # convert the items in wt_dict to numpy array
        for a, wt_a in wt_dict.items():
            for orb in wt_a.keys():
                wt_a[orb] = np.array(wt_a[orb])
        # save to file
        wt_dict_comp = {}
        for a, wt_a in wt_dict.items():
            for orb, wt_o in wt_a.items():
                newkey = a + ":" + orb
                wt_dict_comp[newkey] = wt_o
        np.savez(fbwtfile, **wt_dict_comp)
        np.savetxt(fbeigfile, eig)
    # Prepare for plotting
    linear_k, k_tick, k_label = bs.lineark(True)
    # define colors, row for orbital, column for atom
    color_list = [
        ["dodgerblue", "cyan", "navy", "steelblue", "teal", "blue"],  # s
        ["orange", "yellow", "goldenrod", "darkorange", "gold", "peru"],  # pxy
        [
            "limegreen",
            "palegreen",
            "darkgreen",
            "lime",
            "darkseagreen",
            "aquamarine",
        ],  # pz
        ["red", "lightcoral", "darkred", "darksalmon",
            "mistyrose", "rosybrown"],  # d
        ["purple", "thistle", "darkmagenta", "magenta", "violet", "indigo"],  # f
    ]
    Emin, Emax = Erange
    dE = (Emax - Emin) / (figsize[1] * 5) * fat_scale
    if split_view:
        na = len(proj_ats_orbs.keys())
        figsize_new = (na * figsize[0], figsize[1])
        fig, axes = plt.subplots(
            1,
            na,
            sharex=False,
            sharey=True,
            figsize=figsize_new,
            gridspec_kw={"wspace": 0},
        )
        axes[0].set_ylabel(r"$\rm E-E_F$ (eV)", fontsize=label_fontsizesize)
        axes[0].set_ylim(Emin, Emax)
        axes[0].set_xticks(k_tick)
        axes[0].set_xticklabels(k_label, fontsize=label_fontsizesize)
        for i in range(na - 1):
            axes[i + 1].set_xticks([])
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.set_ylabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsizesize)
        ax.set_xlim(linear_k[0], linear_k[-1])
        ax.set_xticks(k_tick)
        ax.set_xticklabels(k_label, fontsize=label_fontsizesize)
        ax.set_ylim(Emin, Emax)

    legend_dict = {}
    # ib is the band index
    if colors:
        # prepare a generator for colors
        color_gen = itertools.cycle(colors)
    for ib, e in enumerate(eig.T):
        # if the band has any segments that are in the selected energy range
        if np.any(np.logical_and(e < Emax, e > Emin)):
            filled_range = np.array([e, e])
            # plot the bands themselves first
            if split_view:
                for iax in range(na):
                    if not np.any(np.logical_and(e < Emax, e > Emin)):
                        continue
                    axes[iax].plot(linear_k, e, color="k", linewidth=linewidth)
                    if index:
                        ax[-1].annotate(ib + 1, (linear_k[-1], e[-1]))
            else:
                # plot the band only if it has at least one segment in the selected energy range
                if not np.any(np.logical_and(e < Emax, e > Emin)):
                    continue
                ax.plot(linear_k, e, color="k", linewidth=linewidth)
                if index:
                    ax.annotate(ib + 1, (linear_k[-1], e[-1]))
            # always iterate all the atoms and all the orbitals. Change their transparency
            # based on whether you want to see it or not.
            # ia: index of atom, ifig: index of figure. They are different because some
            # atoms might not be chosen to plot
            ia = 0
            ifig = 0
            for a, wt_a in wt_dict.items():
                io = 0  # index of orbital
                if not a in proj_ats_orbs.keys():
                    # if the atom is skipped then change ia only but not ifig
                    ia += 1
                    continue
                for orb, wt_o in wt_a.items():
                    # wt_o is a list
                    if len(wt_o) != 0:
                        if orb in proj_ats_orbs[a]:
                            alp = alpha
                            # define the legend patch
                            if orb == "pxy":
                                label = f"{a}: $p_x+p_y$"
                            elif orb == "pz":
                                label = f"{a}: $p_z$"
                            else:
                                label = f"{a}: ${orb}$"
                            legend_dict.update({label: (ia, io)})

                            # the wt_o array changes k along column, while changes band
                            # index along row.
                            weight = np.abs(wt_o[:, ib] * dE)
                            if colors:
                                c = next(color_gen)
                            else:
                                c = color_list[io][ia]
                            if split_view:
                                ax = axes[ifig]
                                ax.set_ylim(Emin, Emax)
                                ax.set_title(a)
                                ax.set_xlim(linear_k[0], linear_k[-1])
                            ax.fill_between(
                                linear_k,
                                filled_range[0] - weight,
                                filled_range[0],
                                color=c,
                                alpha=alp,
                            )
                            ax.fill_between(
                                linear_k,
                                filled_range[1],
                                filled_range[1] + weight,
                                color=c,
                                alpha=alp,
                            )
                            # update the "already filled range"
                            filled_range = filled_range + \
                                np.array([-weight, weight])
                    io += 1
                ifig += 1
                ia += 1

    # restart the color generator
    if colors:
        color_gen = itertools.cycle(colors)
        legend_elements = [
        Patch(facecolor=next(color_gen), label=label)
        for label, idx in legend_dict.items()
    ]
    else:
        legend_elements = [
            Patch(facecolor=color_list[idx[1]][idx[0]], label=label)
            for label, idx in legend_dict.items()
        ]
    lgd = plt.legend(handles=legend_elements, bbox_to_anchor=legend_position,
               fontsize=legend_fontsize)
    lgd.get_frame().set_alpha(0)
    plt.tick_params(axis='x', labelsize=tick_fontsize)
    plt.tick_params(axis='y', labelsize=tick_fontsize)
    # set border line width
    for axis in ['top', 'bottom', 'left', 'right']:
        plt.gca().spines[axis].set_linewidth(border_linewidth)
    if save:
        for fmt in save_format.split(','):
            projs_dict = convert_dict_to_formatted_str(proj_ats_orbs)
            projs = projs_dict.replace(',','-').replace(':','-')
            plt.savefig(
                os.path.join(save_path, 
                f'{save_name}.FatBands.{projs}.{tick_labels}.E{Emin}to{Emax}.{fmt}'),
                bbox_inches='tight',
                dpi=300, transparent=True)
    if show:
        plt.show()



@timer
def plot_eigst_band(
    H,
    offset: int = 0,
    k=None,
    figsize=None,
    dotsize=1000,
    phase=True,
    fermi_energy=0.0,
    save=False,
    save_name='tmp',
    save_path='.',
    save_format='png,svg',
):
    """
    Plot the eigenstate of a band, by default the topmost valence band
    - offset: offset from the HOMO
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    geom = H.geometry
    if not figsize:
        figsize = guess_figsize(geom)
    if k is None:
        _k = [0, 0, 0]
    else:
        _k = k

    es = H.eigenstate(k=_k)
    eig = H.eigh(k=_k)
    num_occ = len(eig[eig < fermi_energy])
    # num_occ = len(H)//2

    band = num_occ-1+offset
    # print the index of the plotted energy level relative to HOMO and LUMO
    if offset == 0:
        label = 'HOMO'
    elif offset < 0:
        label = f'HOMO{offset}'
    elif offset == 1:
        label = 'LUMO'
    else:
        label = f'LUMO+{offset-1}'
    print("Energy level: {} ({:.4f} eV)".format(label,
                                                H.eigh(k=_k)[band]-fermi_energy))

    fig, ax = plt.subplots(figsize=figsize)
    # plot the bonds first
    for i, atom1 in enumerate(geom.xyz):
        for j, atom2 in enumerate(geom.xyz):
            if i < j:
                distance = np.linalg.norm(atom1 - atom2)
                if 1.3 < distance < 1.7:
                    xs = [atom1[0], atom2[0]]
                    ys = [atom1[1], atom2[1]]
                    ax.plot(xs, ys, color='k', linewidth=1, zorder=1)

    # plot the eigenstate
    if not phase:
        esnorm = es.sub(band).norm2(sum=False).sum(0)
        ax.scatter(H.xyz[:, 0], H.xyz[:, 1], dotsize * esnorm, zorder=2)
    else:
        esstate = es.sub(band).state.squeeze()
        # plot the read part
        esstate = np.real(esstate)
        colorbar_max = np.abs(esstate).max()
        colorbar_min = -colorbar_max
        norm_esstate = (esstate-colorbar_min)/(colorbar_max-colorbar_min)
        color = cm.bwr(norm_esstate)
        ax.scatter(H.xyz[:, 0], H.xyz[:, 1], dotsize * np.abs(esstate), c=color,
                   zorder=2)
        # add horizontal color bar
        norm = mcolors.Normalize(vmin=colorbar_min, vmax=colorbar_max)
        # Create the colorbar using the bwr_r colormap and the normalization instance
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax,
                          orientation='horizontal', label='Amplitude',
                          fraction=0.1, pad=0.05, aspect=10)

    # Calculate the current axis range and the desired additional margin
    x_margin = 3
    y_margin = 3
    x_min, x_max = geom.xyz[:, 0].min(), geom.xyz[:, 0].max()
    y_min, y_max = geom.xyz[:, 1].min(), geom.xyz[:, 1].max()

    # Update xlim and ylim with the new margins
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Set the aspect ratio to 'equal' after updating xlim and ylim
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    if save:
        for f in save_format.split(','):
            # convert [0.1,0,0] to a string, remove brackets and remove extra spaces,
            # replace comma with dash
            kstr = str(_k).replace('[', '').replace(
                ']', '').replace(' ', '').replace(',', '-')
            filename = os.path.join(
                save_path, f'{save_name}.Eigenstate.{label}.k{kstr}.{f}')
            plt.savefig(filename, bbox_inches='tight',
                        dpi=300, transparent=True)

    plt.show()


@timer
def plot_eigst_energy(
    H,
    E=0.0,
    Ewidth=0.1,
    k=None,
    figsize=None,
    dotsize=10,
    mpgrid=[30, 1, 1],
    gaussian_broadening=0.01,
    dE=0.01,
    save=False,
    save_name='tmp',
    save_path='.',
    save_format='png,svg',
):
    """
    Plot the eigenstates whose eigenvalues are in a specific range, by default around fermi level
    Note that this method sums all the orbitals of one atom and plot it as a circle,
    therefore the orbital information is hidden here. To visualize the orbitals,
    use ldos_map method instead.
    pdos adds all the atoms for an orbital, while plot_eigst adds
    all the orbitals for an atom.
    """

    geom = H.geometry
    if not figsize:
        figsize = guess_figsize(geom)
    mp = MonkhorstPack(H, mpgrid)
    mpav = mp.apply.average

    Emin = E - Ewidth / 2
    Emax = E + Ewidth / 2
    mesh_pts = int(Ewidth / dE)
    Emesh = np.linspace(Emin, Emax, mesh_pts)
    lpdos = np.zeros((geom.na, mesh_pts))

    dis = functools.partial(gaussian, sigma=gaussian_broadening)

    def wrap(eigenstate):
        # local projected dos
        # sum all the orbitals for each atom
        PDOS = eigenstate.PDOS(Emesh, distribution=dis).squeeze()
        for io in range(PDOS.shape[0]):
            ia = geom.o2a(io)
            lpdos[ia, :] += PDOS[io, :]
        return lpdos

    lpdos = mpav.eigenstate(wrap=wrap, eta=True)
    lpdos = lpdos.sum(-1)

    fig, ax = plt.subplots(figsize=figsize)
    # plot the bonds first
    for i, atom1 in enumerate(geom.xyz):
        for j, atom2 in enumerate(geom.xyz):
            if i < j:
                distance = np.linalg.norm(atom1 - atom2)
                if 1.3 < distance < 1.7:
                    xs = [atom1[0], atom2[0]]
                    ys = [atom1[1], atom2[1]]
                    ax.plot(xs, ys, color='k', linewidth=1, zorder=1)

    # plot the eigenstate
    ax.scatter(geom.xyz[:, 0], geom.xyz[:, 1], dotsize * lpdos)

    # Calculate the current axis range and the desired additional margin
    x_margin = 3
    y_margin = 3
    x_min, x_max = geom.xyz[:, 0].min(), geom.xyz[:, 0].max()
    y_min, y_max = geom.xyz[:, 1].min(), geom.xyz[:, 1].max()

    # Update xlim and ylim with the new margins
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Set the aspect ratio to 'equal' after updating xlim and ylim
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    if save:
        for f in save_format.split(','):
            filename = os.path.join(
                save_path, f'{save_name}.LDOSmap(no orbital).E{E}width{Ewidth}.{f}')
            plt.savefig(filename, bbox_inches='tight',
                        dpi=300, transparent=True)
    plt.show()




def ldos(H, 
         location, 
         Erange=[-3, 3], 
         figsize=(6,4), 
         gaussian_broadening=0.05,
         plot=True,
         ret=False,
         linewidth=1,
         border_linewidth=1.5,
         save=False,
         save_name='tmp',
         save_path='.',
         save_format='png,svg',
         **kwargs):
    """
    Plot the local density of states
    Args:
        H: hamiltonian
        location: list of indices of the atoms
        Erange: energy range
        figsize: figure size
        gaussian_broadening: gaussian broadening
        plot: plot or not
        ret: return the data or not
        linewidth: linewidth of the plot
        border_linewidth: linewidth of the border of the plot
        save: save the plot or not
        save_name: name of the saved file
        save_path: path of the saved file
        save_format: format of the saved file
        **kwargs: other arguments for plt.plot
    Returns:
        X: energy
        Y: local density of states
    """
    es = H.eigenstate()
    Emin, Emax = Erange
    eig = H.eigh()
    sub = np.where(np.logical_and(eig > Emin, eig < Emax))
    es_sub = es.sub(sub)
    eig_sub = eig[sub]

    X = np.linspace(Emin, Emax, 1000)
    Y = np.zeros((X.shape[0], len(location)))

    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    for i, loc in enumerate(location):
        es_sub_loc = es_sub.state[:, loc]
        ldos = np.multiply(es_sub_loc, np.conj(es_sub_loc))
        ldos = ldos / ldos.max()  # from 0 to 1

        x = np.linspace(Emin, Emax, 1000)
        y = np.zeros_like(x)

        for j in range(ldos.shape[0]):
            y += ldos[j] * gaussian(x, eig_sub[j], gaussian_broadening)

        y = y / np.max(y)
        Y[:, i] = y

    if plot:
        fig, ax = plt.subplots(figsize=figsize)

        for i, loc in enumerate(location):
            ax.plot(X, Y[:, i], label='Atom {}'.format(loc), 
                    linewidth=linewidth, **kwargs)

        ax.set_xlabel(r'$\rm E-E_F$ (eV)')
        ax.set_ylabel('LDOS (arb. u.)')
        ax.set_yticks([])
        ax.set_xlim(Emin, Emax)
        ax.legend()

        # set the borader linewidth
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(border_linewidth)

        if save:
            for fmt in save_format.split(','):
                location_str = '-'.join([str(loc) for loc in location])
                filename = '{}.LDOS.Atom{}.E{}to{}.{}'.format(save_name, 
                            location_str, Emin, Emax, fmt)
                fig.savefig(os.path.join(save_path, filename), dpi=300,
                            bbox_inches='tight', transparent=True)
        plt.show()

    if ret:
        return X, Y


@timer
def cell_ldos(g:Geometry=None, 
              repetitions:int=None, 
              H:Hamiltonian=None,
              end_cell_indices:list=None,
              bulk_cell_indices:list=None,
              fermi_energy=None,
              Erange=[-3, 3], 
              figsize=[6, 4],
              gaussian_broadening=0.05, 
              linewidth=1, 
              border_linewidth=1.5,
              ticks_fontsize=14,
              label_fontsize=14,
              plot_geom=True, 
              save=False, 
              save_name='tmp',
              save_path='.', 
              save_format='png,svg',
              vertical_plot=False,
              **kwargs):
    """
    Calculate the local density of states of the end cell and the middle cell,
    provide either the geometry with repetitions or the Hamiltonian.
    Args:
        g: geometry
        repetitions: number of repetitions
        Erange: energy range
        figsize: figure size of LDOS plot
        gaussian_broadening: gaussian broadening
        linewidth: linewidth of the plot
        border_linewidth: linewidth of the border of the plot
        ticks_fontsize: fontsize of the ticks
        label_fontsize: fontsize of the label
        plot_geom: plot the geometry or not
        save: save the plot or not
        save_name: name of the saved file
        save_path: path of the saved file
        save_format: format of the saved file
        **kwargs: other arguments for plt.plot
    """
    if g is not None:
        G = g.tile(repetitions,0)
        G = rotate_gnr(G)
        G.set_nsc([1,1,1])
        G.cell[0,0] += 20
        G = move_to_center(G, plot_geom=False)
        H = construct_hamiltonian(G)
        fermi_energy = 0.0

        # Find the indices of the end cell
        end_cell_indices = np.arange(0,g.na)
        bulk_index = repetitions//2
        bulk_cell_indices = np.arange((bulk_index-1)*g.na, bulk_index*g.na)
    else:
        # assert H is not None, "Either geometry or Hamiltonian must be provided"
        try:
            G = H.geometry
            assert end_cell_indices is not None
            assert bulk_cell_indices is not None
            assert fermi_energy is not None
        except:
            raise ValueError("without providing geometry and repetitions, \
                             Hamiltonian and the indices of end and bulk cell \
                             must be provided. Fermi energy also need to be provided.")

    if plot_geom:
        # Plot the bonds in geometry G, with end cell in red and bulk cell in magenta, the rest 
        # in teal, scale the figsize with the geometry size
        molecule_size = get_molecule_size(G)
        fig_with, fig_height = molecule_size[:2] * 0.5
        fig1, ax1 = plt.subplots(figsize=(fig_with, fig_height))
        # plot the bonds in G. Neighbours closer than 1.6 Angstrom are considered as bonded
        # Plot the bonds in end cell in teal, bulk cell in magenta, and the rest in black

        for i in range(G.na):
            for j in range(i+1, G.na):
                if G.rij(i, j) < 1.6:
                    if i in end_cell_indices and j in end_cell_indices:
                        ax1.plot([G.xyz[i, 0], G.xyz[j, 0]], [G.xyz[i, 1], G.xyz[j, 1]],
                                color='teal', linewidth=3)
                    elif i in bulk_cell_indices and j in bulk_cell_indices:
                        ax1.plot([G.xyz[i, 0], G.xyz[j, 0]], [G.xyz[i, 1], G.xyz[j, 1]],
                                color='magenta', linewidth=3)
                    else:
                        ax1.plot([G.xyz[i, 0], G.xyz[j, 0]], [G.xyz[i, 1], G.xyz[j, 1]],
                                color='black', linewidth=3)

        # set equal aspect ratio
        ax1.set_aspect('equal')
        # remove axes
        ax1.set_axis_off()
        # show the plot
        plt.show()

    # Calculate the LDOS of the end cell and the bulk cell
    x_end, y_end = ldos(H, end_cell_indices, Erange=np.array(Erange)+fermi_energy, 
                        gaussian_broadening=gaussian_broadening,
                        plot=False, ret=True)
    x_bulk, y_bulk = ldos(H, bulk_cell_indices, Erange=np.array(Erange)+fermi_energy,
                            gaussian_broadening=gaussian_broadening,
                            plot=False, ret=True)
    x_end -= fermi_energy
    x_bulk -= fermi_energy    
    
    # Plot the LDOS of the end cell and the bulk cell
    if not vertical_plot:
        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.plot(x_end, y_end.sum(axis=1), label='End cell', color='teal',
                 linewidth=linewidth, **kwargs)
        ax2.plot(x_bulk, y_bulk.sum(axis=1), label='Bulk cell', color='magenta',
                 linewidth=linewidth, **kwargs)
        ax2.set_xlabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsize)
        ax2.set_xlim(Erange[0], Erange[1])
        ax2.set_ylabel('LDOS (arb. u.)', fontsize=label_fontsize)
        ax2.set_ylim(0, None)
        ax2.set_yticks([])
    else:
        fig2, ax2 = plt.subplots(figsize=figsize[::-1])
        ax2.plot(y_end.sum(axis=1), x_end, label='End cell', color='teal',
                 linewidth=linewidth, **kwargs)
        ax2.plot(y_bulk.sum(axis=1), x_bulk, label='Bulk cell', color='magenta',
                 linewidth=linewidth, **kwargs)
        ax2.set_ylabel(r'$\rm E-E_F$ (eV)', fontsize=label_fontsize)
        ax2.set_ylim(Erange[0], Erange[1])
        ax2.set_xlabel('LDOS (arb. u.)', fontsize=label_fontsize)
        ax2.set_xlim(0, None)
        ax2.set_xticks([])

    legend = plt.legend()
    legend.get_frame().set_alpha(0)

    # set the borader linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(border_linewidth)
    plt.tick_params(axis='x', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', labelsize=ticks_fontsize)

    if save:
        for fmt in save_format.split(','):
            if plot_geom:
                geom_file = '{}.{}'.format(save_name, fmt)
                fig1.savefig(os.path.join(save_path, geom_file), dpi=300,
                             bbox_inches='tight', transparent=True)
            ldos_file = '{}.LDOS.Cell.E{}to{}.{}'.format(save_name, Erange[0],
                                                         Erange[1], fmt)
            fig2.savefig(os.path.join(save_path, ldos_file), dpi=300,
                         bbox_inches='tight', transparent=True)
    plt.show()
   


@timer
def ldos_map(
    H,
    E=0.0,
    Ewidth=0.1,
    k=[0, 0, 0],
    height=3.0,
    mesh=0.1,
    save=False,
    save_name='tmp',
    save_path='./',
    save_format='png,svg',
    figsize=None,
):
    """
    Localized Density of States
    - E: the median of the energy range that you want to investigate
    - Ewidth: width of the energy range
    - height: height above the plane of the ribbon, default 1.0 angstrom
    - mesh: mesh size of the grid, default 0.1 angstrom
    """
    xyz = H.xyz
    minxyz = np.amin(xyz, 0)
    maxxyz = np.amax(xyz, 0)
    length = maxxyz - minxyz
    # Calculate the length-to-width ratio and adjust the figsize
    ratio = length[0] / length[1]
    if figsize is None:
        figsize = (4 * ratio, 4)

    attach_pz(H.geometry)

    Emin = E - Ewidth / 2
    Emax = E + Ewidth / 2

    es = H.eigenstate(k=k)
    eig = H.eigh(k=k)
    sub = np.where(np.logical_and(eig > Emin, eig < Emax))[0]

    zmean = H.geometry.xyz[:, 2].mean()
    dos = 0
    for b in sub:
        grid = Grid(mesh, sc=H.sc)
        index = grid.index([0, 0, zmean + height])
        es_sub = es.sub(b)
        es_sub.wavefunction(grid)  # add the wavefunction to grid
        dos += grid.grid[:, :, index[2]].T ** 2
    # normalize
    dos = dos/dos.max()

    plt.figure(figsize=figsize)
    plt.imshow(dos, cmap="hot", origin='lower')
    plt.axis('off')
    plt.colorbar(orientation='horizontal', fraction=0.1, pad=0.05, aspect=10,
                 ticks=[0, 1], label='LDOS')
    if save:
        for f in save_format.split(','):
            filename = os.path.join(
                save_path, f'{save_name}.LDOSmap.E{E}width{Ewidth}.{f}')
            plt.savefig(filename, bbox_inches='tight',
                        dpi=300, transparent=True)
    plt.show()
    return dos


@timer
def zak_phase(H):

    bs = BandStructure(
        H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
            "$\Gamma$", "$X$", "$\Gamma$"]
    )

    occ = [i for i in range(len(H.eig()) // 2)]
    gamma = electron.berry_phase(bs, sub=occ, method="zak")

    return gamma


def zak(contour, sub=None, gauge="R"):

    from sisl import Hamiltonian
    import sisl._array as _a
    from sisl.linalg import det_destroy
    import numpy as np
    from numpy import dot, angle

    if not isinstance(contour.parent, Hamiltonian):
        raise TypeError(
            "Requires the Brillouine zone object to contain a Hamiltonian")

    if not contour.parent.orthogonal:
        raise TypeError("Requires the Hamiltonian to use orthogonal basis")

    if not sub:
        raise ValueError("Calculate only the occupied bands!")

    def _zak(eigenstates):
        first = next(eigenstates).sub(sub)
        if gauge == "r":
            first.change_gauge("r")
        prev = first
        prd = 1
        for second in eigenstates:
            second = second.sub(sub)
            if gauge == "r":
                second.change_gauge("r")
            prd = dot(prd, prev.inner(second, diag=False))
            prev = second
        if gauge == "r":
            g = contour.parent.geometry
            axis = contour.k[1] - contour.k[0]
            axis /= axis.dot(axis) ** 0.5
            phase = dot(g.xyz[g.o2a(_a.arangei(g.no)), :], dot(axis, g.rcell)).reshape(
                1, -1
            )
            prev.state *= np.exp(-1j * phase)
        prd = dot(prd, prev.inner(first, diag=False))
        return prd

    d = _zak(contour.apply.iter.eigenstate())
    ddet = det_destroy(d)
    result = -angle(ddet)

    return result


@timer
def inter_zak(H, offset: int = 0, fermi_energy=0.0):

    bs = BandStructure(
        H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 200, [
            "$\Gamma$", "$X$", "$\Gamma$"]
    )

    bsar = bs.apply.array
    lk = bs.lineark(ticks=False)
    eigh = bsar.eigh()  # row-k, column-energy
    e0 = eigh[0, :]  # at Gamma point
    occ0 = len(e0[e0 < fermi_energy])  # at Gamma point
    # make sure it is insulator, band lines don't pass fermi level
    for i in range(eigh.shape[0]):
        ei = eigh[i, :]
        occi = len(ei[ei < fermi_energy])
        if occi != occ0:
            ki = [lk[i], 0, 0]
            redki = bs.toreduced(ki)
            raise RuntimeError(
                "Band line pass the fermi level at k point \
                [{:.2f} {:.2f} {:.2f}] (Reduced: [{:.2f} {:.2f} {:.2f}].\
                occ0: {}, occi: {}".format(
                    *ki, *redki, occ0, occi
                )
            )
    occ = [i for i in range(occ0)]
    print('Number of occupied bands: {}'.format(len(occ)))
    if offset == 0:
        occ = occ
    elif offset > 0:
        for i in range(offset):
            occ.append(occ[-1] + 1)
    elif offset < 0:
        for i in range(-offset):
            occ.pop()
    print("Bands taken into account: {} to {}".format(occ[0] + 1, occ[-1] + 1))
    gamma_inter = zak(bs, sub=occ, gauge="R")

    return round(gamma_inter, 5)


@timer
def ssh(H, occ):
    """
    Please provide the offset from homo as occ
    """
    bs = BandStructure(
        H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
            "$\Gamma$", "$X$", "$\Gamma$"]
    )

    occn = len(H) // 2
    bands = []
    occ.sort()
    for i in occ:
        bands.append(occn-1 + i)
    print('Number of occupied bands: {}'.format(occn))
    gamma_inter = zak(bs, sub=bands, gauge="R")
    bandsIdx = np.array(bands) + 1
    print("Bands that are taken into account: ", bandsIdx)

    return round(gamma_inter, 10)


@timer
def zak_band(H, occ):

    bs = BandStructure(
        H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
            "$\Gamma$", "$X$", "$\Gamma$"]
    )

    gamma_inter = zak(bs, sub=occ, gauge="R")

    return round(gamma_inter, 10)


def zak_dft(contour, sub=None, gauge="R"):
    """
    Deprecated method
    """

    from sisl import Hamiltonian, Overlap
    import sisl._array as _a
    from sisl.linalg import det_destroy
    import numpy as np
    from numpy import dot, angle, multiply, conj

    if not isinstance(contour.parent, Hamiltonian):
        raise TypeError(
            "Requires the Brillouine zone object to contain a Hamiltonian")

    if sub is None:
        raise ValueError("Calculate only the occupied bands!")

    def _zak(eigenstates):

        H = contour.parent
        k = contour.k
        dk = k[1] - k[0]

        first = next(eigenstates).sub(sub)
        if gauge == "r":
            first.change_gauge("r")
        prev = first

        prd = 1
        for second in eigenstates:
            if gauge == "r":
                second.change_gauge("r")
            second = second.sub(sub)

            k_sec = second.info["k"]
            ovlpm = H.Sk(k=k_sec - dk / 2, gauge="R", format="array")
            prev_state = prev.state
            second_state = second.state
            inner_prd = dot(dot(conj(prev_state), ovlpm), second_state.T)
            prd = dot(prd, inner_prd)
            prev = second

        if gauge == "r":
            g = contour.parent.geometry
            axis = contour.k[1] - contour.k[0]
            axis /= axis.dot(axis) ** 0.5
            coords = g.xyz[g.o2a(_a.arangei(g.no)), :]
            phase = dot(coords, dot(axis, g.rcell)).reshape(1, -1)
            prev.state *= np.exp(-1j * phase)

        # in case the last state and first state are not equal
        ovlpm_last = H.Sk(k=[0] * 3, gauge="R", format="array")
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

    bs = BandStructure(
        H, [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], 400, [
            "$\Gamma$", "$X$", "$\Gamma$"]
    )

    d_det, gamma_inter = zak_dft(bs, sub=occ, gauge="R")

    return d_det, round(gamma_inter, 10)


def get_zak_dict(H, method="from_top"):
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
        if method == "from_top":
            sub = occ[-i - 1:]
        elif method == "from_bottom":
            sub = occ[: i + 1]
        d, z = zak_band_dft(H, sub)
        zak_dict.update({i: (sub, d, z)})
    return zak_dict


def plot_zak_polar(zdict):
    """
    Put Zak phase in a complex plane, which is a polar plot here
    """
    from matplotlib import cm

    plt.figure(figsize=(10, 10))
    for i, v in zdict.items():
        bwr = cm.get_cmap("bwr")
        rho, r = np.angle(v[1]), np.abs(v[1])
        plt.polar(
            [0, rho],
            [0, r],
            marker="o",
            color=bwr(-r * np.cos(rho) + 1),
            label=list_str(v[0]) + ": " + str(round(r * np.cos(rho), 2)),
            alpha=r * 0.8 + 0.05,
        )
        # plt.text(rho, r, str(i), color="red", fontsize=10)
        plt.legend(bbox_to_anchor=[1.2, 0.9])


@timer
def Z2(H, num_bands=None):
    "Sometimes not correct, not sure why."
    # reverse idx order list
    geom = H.geometry
    idx_rev = []
    center = geom.center()
    for ia in geom:
        xyzia = geom.xyz[ia]
        xyzia_rev = center*2 - xyzia
        idx = geom.close(xyzia_rev, R=0.1)
        idx_rev.append(idx[0])

    if not num_bands:
        e0 = H.eigh()
        occ = e0[e0 < 0].size
    k = [[0, 0, 0], [0.5, 0, 0]]
    res = 1
    for _k in k:
        bands = [i for i in range(occ)]
        states = H.eigenstate(k=_k, gauge='R').sub(bands).state
        states_rev = states[:, idx_rev]
        # np.inner(a, a) = a times a.T
        par_mat = np.inner(np.conj(states), states_rev)
        det = np.linalg.det(par_mat)
        res *= det
    res = np.around(res)
    if res == -1:
        _Z2 = 1
    elif res == 1:
        _Z2 = 0
    return _Z2


@timer
def plot_wannier_centers(
    geom,
    name,
    path: str = None,
    figsize=(6, 4),
    sc=False,
    marker="*",
    marker_size=200,
    marker_color="green",
    plot_sum=True,
    index=False
):
    """
    Plot the wannier centres. 
    Translate to home cell, only translate y and z components.
    The x component will be translated manually.
    Args:
        geom: geometry object
        name: name of geometry
        path: path to wannier centres
        figsize: figure size
        sc: whether to plot supercell
        marker: marker type
        marker_size: marker size
        marker_color: marker color
        plot_sum: whether to plot the sum of wannier centres
        index: whether to plot the index of wannier centres
    """
    if not path:
        path = "./s2w/"
    cell = geom.cell
    gcenter = geom.center(what='xyz')
    ccenter = geom.center(what='cell')
    abc = cell.diagonal()
    file_path = os.path.join(path, name + "_centres.xyz")
    with open(file_path) as f:
        contents = f.readlines()
        # wannier centres in raw coordinates strings
        wc_raw = []
        for line in contents:
            if line.startswith("X"):
                wc_raw.append(line)
    # convert it to an array
    wc = np.array(
        list(
            map(lambda x: list(map(float, x.strip().split()[1:])), wc_raw))
    )
    # Translate wannier centres to home cell
    yzcell = cell.copy()
    yzcell[0] = np.array([0, 0, 0])
    wc_hc = wc - np.floor((wc - (gcenter - abc / 2)) / abc).dot(yzcell)
    # sum of wannier centers, relative to geomtry center
    wc_hc -= gcenter
    wcc_abs = wc_hc.sum(0)
    wcc_frac = wcc_abs.dot(np.linalg.inv(cell))
    wcc_frac = np.around(wcc_frac, 4)
    print("Sum of Wannier centres (Relative to geometry center, absolute):\n\t", wcc_abs)
    # print("Sum of Wannier centres (Relative):\n\t", wcc_rel)
    print("Sum of Wannier centres (Relative to geometry center, fractional):\n\t", wcc_frac)

    # translate back to geometry center for plotting
    wc_hc += gcenter
    plt.figure(figsize=figsize)
    plot(geom, supercell=sc)
    plt.scatter(wc_hc[:, 0], wc_hc[:, 1], s=marker_size,
                c=marker_color, marker=marker)
    if index:
        for i, wc in enumerate(wc_hc):
            # add the text with no shift
            plt.text(wc[0], wc[1], str(i+1), color="red", fontsize=10,
                     horizontalalignment='center', verticalalignment='center')
    if plot_sum:
        wcc_abs += gcenter
        plt.scatter(wcc_abs[0], wcc_abs[1], s=marker_size*2,
                    c="k", marker=marker)
    plt.axis("equal")
    plt.axis('off')
    plt.show()



def view_wannier_centers(path, 
                         plot_sum=True, 
                         index=False, 
                         bond_cutoff=1.6,
                         expand_length=0.71,
                         scaling_factor = 0.5,
                         save=True, 
                         save_format='png,svg'):
    """Plot the Wannier centers for a given path.
    Args:
        path (str): path to the folder containing the .XV and _centres.xyz files
        plot_sum (bool): whether to plot the sum of the Wannier centers
        index (bool): whether to plot the index of the Wannier centers
        bond_cutoff (float): the cutoff distance to determine whether two atoms are bonded
        expand_length (float): the length to expand the box containing geometry
        scaling_factor (float): the scaling factor to scale the geometry
        save (bool): whether to save the plot
        save_format (str): the format to save the plot
    """
    from glob import glob

    xyz_file = glob(os.path.join(path, '*_centres.xyz'))[0]
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    atoms = []
    atom_elements = []
    wanniers = []
    for line in lines:
        if line.startswith('X'):
            wanniers.append(list(map(float, line.split()[1:])))
        elif line.startswith('C') or line.startswith('H'):
            elements = line.split()[0]
            atom_elements.append(elements)
            atoms.append(list(map(float, line.split()[1:])))
    atoms = np.array(atoms)
    wanniers = np.array(wanniers)

    # Read the unit cell from .XV file
    xv_file = glob(os.path.join(path, '*.XV'))[0]
    g = get_sile(xv_file).read_geometry()
    cell = g.cell
    # translate to home cell if they are shifted by multiples of cell vectors
    gcenter = g.center(what='xyz')
    ccenter = g.center(what='cell')
    abc = cell.diagonal()

    # Calculate the real size of the geometry
    min_coords = np.min(atoms, axis=0)
    max_coords = np.max(atoms, axis=0)
    geometry_size = max_coords - min_coords

    # Adjust the figure size based on the scaling factor
    fig_width, fig_height = geometry_size[:2] * scaling_factor
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Find the convex hull of the geometry (atoms)
    def expand_convex_hull(points, hull, expand_length):
        expanded_hull_points = points.copy()
        for i in range(len(hull.simplices)):
            normal = hull.equations[i, :-1]
            simplex = hull.simplices[i]
            centroid = np.mean(points[simplex], axis=0)
            displacement = (expand_length / np.linalg.norm(normal)) * normal
            expanded_hull_points[simplex] = points[simplex] + displacement
        return expanded_hull_points

    hull = ConvexHull(atoms[:, :2])
    expanded_hull_points = expand_convex_hull(
        atoms[:, :2], hull, expand_length)

    # Create a Delaunay triangulation of the expanded hull points
    delaunay = Delaunay(expanded_hull_points)

    # Check if a point is inside the expanded convex hull
    def is_inside_expanded_hull(point):
        return delaunay.find_simplex(point) >= 0

    # Translate Wannier centers back to the area defined by the expanded convex hull
    for i, wannier in enumerate(wanniers):
        for a, b, c in itertools.product(range(-1, 2), repeat=3):
            translation = a * cell[0] + b * cell[1] + c * cell[2]
            translated_wannier = wannier + translation
            if is_inside_expanded_hull(translated_wannier[:2]) and translated_wannier[2] >= atoms[:, 2].min() - expand_length and translated_wannier[2] <= atoms[:, 2].max() + expand_length:
                wanniers[i] = translated_wannier
                break

    def draw_bond_segment(ax, x1, y1, x2, y2, linewidth, color, round_end):
        line = plt.Line2D([x1, x2], [y1, y2], linewidth=linewidth,
                          color=color, zorder=1,
                          solid_capstyle='round' if round_end else 'butt')
        ax.add_line(line)

    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            d = np.linalg.norm(atoms[i] - atoms[j])
            if d < bond_cutoff:
                x1, y1 = atoms[i, :2]
                x2, y2 = atoms[j, :2]

                if atom_elements[i] == atom_elements[j]:  # C-C bond
                    draw_bond_segment(ax, x1, y1, x2, y2, linewidth=5, color=[
                                      0.36]*3, round_end=True)
                else:  # C-H bond
                    mid_point = (atoms[i] + atoms[j]) / 2
                    color1 = [0.36]*3 if atom_elements[i] == 'C' else [0.8]*3
                    color2 = [0.36]*3 if atom_elements[j] == 'C' else [0.8]*3
                    round_end1 = atom_elements[i] == 'H'
                    round_end2 = atom_elements[j] == 'H'
                    draw_bond_segment(
                        ax, mid_point[0], mid_point[1], x2, y2, linewidth=5, color=color2, round_end=round_end2)
                    draw_bond_segment(
                        ax, x1, y1, mid_point[0], mid_point[1], linewidth=5, color=color1, round_end=False)

    ax.scatter(wanniers[:, 0], wanniers[:, 1], color='red', s=100, zorder=2)
    if index:
        for i in range(len(wanniers)):
            x, y = wanniers[i, :2]
            ax.text(x, y, str(i+1), fontsize=20, color='black',
                    ha='center', va='center', zorder=3)

    # Calculate the average position of the Wannier centers
    wannier_sum = (wanniers - gcenter).sum(axis=0)
    wannier_sum_frac = np.dot(wannier_sum, np.linalg.inv(cell))
    print("Sum of Wannier centres (Relative to geometry center, absolute):\n\t",
          [round(x, 4) for x in wannier_sum])
    print("Sum of Wannier centres (Relative to geometry center, fractional):\n\t",
          [round(x, 4) for x in wannier_sum_frac])

    if plot_sum:
        # Plot the average position with a blue diamond marker
        # convert relative to geometry center to absolute
        wannier_sum += gcenter
        ax.scatter(wannier_sum[0], wannier_sum[1], color='blue',
                   marker='D', s=150, zorder=3, edgecolors='none', linewidths=1)

    ax.set_xlim([atoms[:, 0].min()-2, atoms[:, 0].max()+2])
    ax.set_ylim([atoms[:, 1].min()-0.5, atoms[:, 1].max()+0.5])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis('off')
    if save:
        # save all the formats defined in save_format
        for fmt in save_format.split(','):
            filepath = xyz_file.split('.')[0] + '.' + fmt
            plt.savefig(filepath, transparent=True, bbox_inches='tight', dpi=300)

    plt.show()




def chiral_phase_index(H, knpts=200, plot_phase=False,
                       Aidx=None, Bidx=None):
    """
    Calculate the chiral phase index defined in
        'Jingwei Jiang and Steven Louie, Nano Lett. 2021, 21, 1, 197–202'
    Args:
        knpts: number of k points in BZ
        plot_phase: plot the evolution of phase
        Aidx, Bidx: provide indices of A and B sublattices because the 
            find_sublattice method sometimes doesn't work if the lattice is
            not regular
    """
    from numpy.linalg import det
    from numpy import angle

    phases = []
    # phases2 = []
    g = H.geometry
    if not (Aidx and Bidx):
        Aidx, Bidx = find_sublattice(g)
    N = len(Aidx)

    def off_diagonalize(M: np.ndarray):
        # Bring a matrix to off-diagonalized form
        Mnew = np.zeros((2*N, 2*N), dtype=np.complex128)
        for i in range(N):
            for j in range(N):
                Mnew[i, j] = M[Aidx[i], Aidx[j]]
                Mnew[i, N+j] = M[Aidx[i], Bidx[j]]
                Mnew[N+i, j] = M[Bidx[i], Aidx[j]]
                Mnew[N+i, N+j] = M[Bidx[i], Bidx[j]]
        return Mnew

    k = np.linspace(0, 1, knpts)
    kiter = (i for i in k)
    k_prev = next(kiter)
    k_first = k_prev
    # modified eigenvalue matrix
    eig_mod = np.kron(np.array([[-1, 0], [0, 1]]), np.eye(N))
    states_prev = H.eigenstate(k=[k_prev, 0, 0]).state
    Qk_prev = states_prev.T.dot(eig_mod).dot(states_prev.conj())
    Uk_prev = off_diagonalize(Qk_prev)[:N, N:]
    for i in range(knpts-1):
        k_second = next(kiter)
        states_second = H.eigenstate(k=[k_second, 0, 0]).state
        Qk_second = states_second.T.dot(eig_mod).dot(states_second.conj())
        Uk_second = off_diagonalize(Qk_second)[:N, N:]
        detUk = det(Uk_prev.conj().T.dot(Uk_second))
        ph = angle(detUk)
        phases.append(ph)

        # Hk_prev = H.Hk(k=[k_prev,0,0], format='array')
        # Uk_prev = off_diagonalize(Hk_prev)[:N,N:]
        # Hk_second = H.Hk(k=[k_second,0,0], format='array')
        # Uk_second = off_diagonalize(Hk_second)[:N,N:]
        # detUk = det(Uk_prev.conj().T.dot(Uk_second))
        # eigUk = np.linalg.eig(Uk_prev)
        # ph = angle(detUk)

        k_prev = k_second
        Uk_prev = Uk_second
    k_second = k_first
    states_second = H.eigenstate(k=[k_second, 0, 0]).state
    Qk_second = states_second.T.dot(eig_mod).dot(states_second.conj())
    Uk_second = off_diagonalize(Qk_second)[:N, N:]
    detUk = det(Uk_prev.conj().T.dot(Uk_second))
    ph = angle(detUk)
    phases.append(ph)
    phases = np.array(phases)

    sumPhases = phases.sum()
    Z = np.around(sumPhases/2/np.pi, 5)

    if plot_phase:
        cumPhases = phases.cumsum()
        rads = np.linspace(1, abs(Z)*1.2, cumPhases.shape[0])
        plt.plot(np.multiply(rads, np.cos(cumPhases)),
                 np.multiply(rads, np.sin(cumPhases)))
        # plt.scatter(np.linspace(0,1,knpts),phases, 50, marker='*')
        # plt.xticks([0,0.25,0.5,0.75,1])
        plt.axis('equal')

    return Z



def bs_under_electric_field(H, field, plot=True, **kwargs):
    h = H.copy()
    g = h.geometry
    for i in range(len(h)):
        ymean = g.xyz.mean(axis=0)[1]
        y = g.xyz[i, 1] - ymean
        h[i, i] = field*y
    if plot:
        band_structure(h, **kwargs)
    return h
