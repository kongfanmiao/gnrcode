from stringprep import map_table_b2
from sisl import Geometry
import numpy as np
import matplotlib.pyplot as plt
import os
from .geometry import *
from .band_analysis import *
from .tools import *
from datetime import datetime

import socket, getpass
userName = getpass.getuser()
hostName = socket.gethostname()
User = userName + '@' + hostName


kpoints_dict = {
    "G": ("\Gamma", [0., 0., 0.]),
    "X": ("X", [0.5, 0., 0.]),
    "M": ("M", [0.5, 0.5, 0.]),
    "K": ("K", [2.0 / 3, 1.0 / 3, 0.]),
}

def get_datetime():
    # dd/mm/YY H:M:S
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


def write_siesta_runfile(
        geom: Geometry, name: str, path="./opt",
        pseudo_path=None,
        xc_functional='GGA',
        xc_authors='PBE',
        basis_size='DZP',
        pao_energy_shift=100,
        mesh_cutoff=400,
        mpgrid=[21, 1, 1],
        electronic_temperature=300,
        variable_cell=True,
        md_steps=1000,
        max_disp_len=0.05,
        max_force_tol=0.01,
        diag_algorithm="Divide-and-Conquer",
        num_eigenstates=None,
        mixer_method='Pulay',
        mixer_weight=0.25,
        mixer_history=6,
        scf_H_tol=1e-3,
        max_scf_iter=500,
        cdf=True,
        spin_polarized=False,
        spin_afm=True,
        spin_orbit=False,
        soc_strength=1.0,
        compute_bands=True,
        bandlines_kpath='XGX',
        bandlines_nkpts=200,
        denchar=False,
        wfs_write_for_kpts=False,
        wfs_write_for_bands=False,
        wfs_kpts: list = None,
        wfs_bands_range: list = None,
        E_field: list = None,
        slab_dipole_correction=False,
        optical_calc=False,
        wannier90=False,
        num_bands_to_wannier=None,
        num_bands_to_wannier_up=None,
        num_bands_to_wannier_down=None,
        s2w_mesh=2,
        others:dict=None
):
    """
    Write Siesta input file
    Args:
        pseudo_path: pseudopotential file path
        xc_functional: Exchange correlation functional (default GGA)
        xc_authors: functional flavor (default PBE)
        basis_size: SZ, SZP, DZ, DZP (default), TZ, TZP, ...
        pao_energy_shift: PAO.EnergyShift, default 100 meV
        mesh_cutoff: Plane wave cutoff, in unit of Ry
        mpgrid: MonkHorst Pack grid (default 21x1x1)
        electronic_temperature: electronic temperature (default 300K)
        variable_cell: Fix the cell during MD relaxation or not
        md_steps: Maximum MD steps
        max_disp_len: Max atomic displacement in optimization move, in Ang
        max_force_tol: Max Force tolerance in coordinate optimization, in unit of eV/Ang
        diag_algorithm: By default it's Divide-and-Conquer. If job is to send to
            ARC, then use "expert" instead
        num_eigenstates: number of eigenstates to calculate. Could be efficient
            to reduce computing time. Only effective for MRRR, ELPA and Expert.
        mixer_method: SCF mixer method, Pulay (default), or Broyden, Linear
        mixer_weight: SCF mixing weight
        mixer_history: SCF mixer history
        scf_H_tol: maximum absolute tolerance of Hamiltonian matrix elements
        max_scf_iter: Max number of SCF iterations
        cdf: Use NetCDF utility or not
        spin_polarized: Spin polarized or unpolarized calculation
        spin_afm: Antiferromagnetic configuration for spin initialization, if False,
            then it means ferromagnetic configuration
        spin_orbit: Includes spin-orbit coupling in calculation
        soc_strength: Spin-orbit coupling strength
        compute_bands: compute band structure or not
        bandlines_kpath: k path to calculation band structure
        bandlines_nkpts: Number of k points for band line from 0 to 2*pi/a
        denchar: Use DENCHAR to calculate the density of charge. Required for 
            wavefunctions calculation
        wfs_write_for_kpts: Calculate and plot wavefunctions or not
        wfs_write_for_bands: Write wavefunctions for given bands
        wfs_kpts: At which k points that the wavefunctions are calculated.
            Note that by default the k points is scaled by pi/a. MUST use float number!
        wfs_bands_range: Specify for which bands that the wavefunctions are calculated,
            and this calculates the wavefunctions at all k points for specified bands
        E_field: Apply electric field to the system, by default should be a list
        slab_dipole_correction: Use slab dipole correction during electric field calculation
        optical_calc: Including the optical calculation module or not
        wannier90: Interface with Wannier90 or not
        num_bands_to_wannier: Specify the number of bands to be wannierized. If 
            not given, Siesta will take all occupied bands
        num_bands_to_wannier_up: Number of spin up bands to be wannierized. If 
            not given, Siesta will use same value as bands_to_wannier
        num_bands_to_wannier_down: Similar to num_bands_to_wannier_up, but for spin down
        s2w_mesh: Mesh grid size per Angstrom during Siesta2Wannier90 calculation. The default
            is 2, which means 2 mesh points for each Angstrom
    """
    # Some other default parameters:
    #   PAO.BasisType       split
    #   SolutionMethod      diagon
    #   SaveRho             F
    #   WriteVoronoiPop     F
    #   NetCharge           0
    #   WriteBands          false # write eigenvalues to .out file

    run_file = name + '_RUN.fdf'
    struct_file = name + '_STRUCT.fdf'

    # prepare pseudopotential files
    species = set()
    for a in geom.atoms:
        species.add(a.tag)
    species = ','.join(species)
    copy_psf_files(path_from=pseudo_path, path_to=path, 
        elements=species, functional=xc_functional)

    def check_directory(calc_type):
        # for some calculations, create new directory for them. This method will
        # raise error if we are still working in the ./opt directory
        # if path == "./opt":
        #     raise ValueError(f"Dont't Work in directory ./opt for {calc_type}")
        pass

    mp1, mp2, mp3 = mpgrid
    with open(os.path.join(path, run_file), 'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
%include {struct_file}
SystemName              {name}
SystemLabel             {name}

############################################
#   Parameters
############################################
XC.functional           {xc_functional}
XC.authors              {xc_authors}
PAO.BasisSize           {basis_size}
PAO.EnergyShift         {pao_energy_shift} meV
MeshCutoff              {mesh_cutoff} Ry
%block kgrid.MonkhorstPack
    {mp1}   0   0   0.0 
    0   {mp2}   0   0.0
    0   0   {mp3}   0.0
%endblock kgrid.MonkhorstPack
ElectronicTemperature   {electronic_temperature} K

############################################
#   Molecular Dynamics
############################################
MD.TypeOfRun            CG 
MD.Steps                {md_steps}
MD.MaxDispl             {max_disp_len}  Ang
MD.MaxForceTol          {max_force_tol} eV/Ang
MD.VariableCell         {variable_cell}
MD.UseSaveXV            T
MD.UseSaveCG            T

############################################
#   SCF
############################################
Diag.Algorithm          {diag_algorithm}
""")
        if num_eigenstates:
            # Only use this argument when diagonalization algorithm is
            # MRRR, ELPA, or Expert
            f.write("""NumberOfEigenStates     {}
""".format(num_eigenstates))
        f.write(f"""DM.UseSaveDM            T
DM.History.Depth        6
MaxSCFIterations        {max_scf_iter}
SCF.Mixer.Method        {mixer_method}
SCF.Mixer.Weight        {mixer_weight}
SCF.Mixer.History       {mixer_history}
SCF.H.Tolerance         {scf_H_tol} eV

############################################
#   Output Settings
############################################
COOP.write              F   # Write .fullBZ.WFSX and .HSX file
WriteMullikenPop        1   # Write atomic and orbital charges
SaveHS                  T   # Write .HSX file
WriteCoorXmol           T   # Write optimized structure coordinates in .xyz file
WriteCoorStep           T   # Write coordinate in every MD step to .XV file
WriteMDXmol             F   # Write .ANI file readable by XMoL for animation of MD
WriteForces             T   # Write forces of each MD step to output file
TS.HS.Save              T
WFS.Energy.Min          -20 eV
WFS.Energy.Max          20 eV""")
        if cdf:
            f.write("""
CDF.Save                T
CDF.Compress            3
""")
#------------------------------------------------------------------------------#
        # write spin settings, by default spin unpolarized
        spin_mode = 'non-polarized'
        if spin_polarized:
            spin_mode = 'polarized'
        if spin_orbit:
            spin_mode = 'spin-orbit'
        f.write(f"""
############################################
#   Spin Settings
############################################
Spin                    {spin_mode}""")
        # this can only change the initial spin state, if read from .DM,
        # this parameter is useless. So, delete .DM file before change this
        # parameter
        if spin_mode != 'non-polarized':
            f.write(f"""
DM.InitSpin.AF          {spin_afm}
""")
        if spin_mode == 'spin-orbit':
            f.write(f"""Spin.OrbitStrength          {soc_strength}
""")
#------------------------------------------------------------------------------#
        # calculate band structure
        if compute_bands:
            f.write("""
############################################
#   Band Structures
############################################
# BandLinesScale  pi/a 
%block BandLines""")
            for i, bdk in enumerate(bandlines_kpath):
                tmp = kpoints_dict[bdk]
                ktmp = 2*np.array(tmp[1]) # remember bandlines scale is pi/a
                nkpt = 1 if i == 0 else int(bandlines_nkpts*np.linalg.norm(
                    np.array(ktmp)/2-np.array(kpoints_dict[bandlines_kpath[i-1]][1])))
                f.write("\n{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
                    nkpt, *ktmp, tmp[0]))
            f.write("""
%endblock BandLines""")
#------------------------------------------------------------------------------#
        # use DENCHAR program to plot wavefunction
        if denchar:
            denchar_file_name = name + ".denchar.fdf"
            f.write(f"""
############################################
#   Use Utility Program DENCHAR
############################################
Write.Denchar            T   # write .PLD and .DIM file to be used by Denchar
%include {denchar_file_name}
""")
#------------------------------------------------------------------------------#
        # write wavefunction for selected k points and selected bands
        # generate SystemLabel.selected.WFSX file
        if wfs_write_for_kpts:
            f.write("""
############################################
#   Write Wavefunctions for k-Points
############################################
WriteWaveFunctions      T
# the k points are in the scale of pi/a by default
%block WaveFuncKPoints""")
            # The WaveFuncKPointsScale is pi/a by default
            for kpt in wfs_kpts:
                if not isinstance(kpt, list):
                    raise TypeError(
                        "The k points list should be a list of lists")
                f.write("\n {:5f} {:5f} {:5f} from {} to {}".format(
                    *kpt, *wfs_bands_range))
            f.write("""
%endblock WaveFuncKPoints
""")
#------------------------------------------------------------------------------#
        # write wavefunction for selected bands and all k points
        # generate SystemLabel.bands.WFSX file
        if wfs_write_for_bands:
            f.write("""
############################################
#   Write Wavefunctions for Bands
############################################
WriteWaveFunctions      T
WFS.Write.For.Bands     T
WFS.Band.Min            {}
WFS.Band.Max            {}
""".format(*wfs_bands_range))
#------------------------------------------------------------------------------#
        # Electric field
        if E_field:
            check_directory("calculation with electric field")
            f.write("""
############################################
#   Electric Field
############################################
%block ExternalElectricField
    {:.5f}  {:.5f}  {:.5f}  V/Ang
%endblock ExternalElectricField
Slab.DipoleCorrection    {}
""".format(*E_field, slab_dipole_correction))
#------------------------------------------------------------------------------#
        # optical calculation
        if optical_calc:
            check_directory("optical calculation")
            optcalc_file_name = name + '.optical_calc.fdf'
            f.write(f"""
############################################
#   Optical Calculation
############################################
%include {optcalc_file_name}
""")
#------------------------------------------------------------------------------#
        # Interface with Wannier90
        if wannier90:
            check_directory("interfacing with Wannier90")
            s2w_grid = np.rint(geom.cell.diagonal()*s2w_mesh).astype(np.int0)
            f.write("""
############################################
#   Interface with Wannier90
############################################
Siesta2Wannier90.WriteMmn       T
Siesta2Wannier90.WriteAmn       T
Siesta2Wannier90.WriteEig       T
Siesta2Wannier90.WriteUnk       T
Siesta2Wannier90.UnkGridBinary  T
Siesta2Wannier90.UnkGrid1       {:d}
Siesta2Wannier90.UnkGrid2       {:d}
Siesta2Wannier90.UnkGrid3       {:d}""".format(*s2w_grid))
            if spin_mode == "non-polarized":
                # NumberOfBands is by default all occupied bands
                if num_bands_to_wannier:
                    f.write(f"""
Siesta2Wannier90.NumberOfBands  {num_bands_to_wannier}""")
            else:
                # NumberOfBandsUp/Down are by default the same as NumberOfBands
                f.write(f"""
Siesta2Wannier90.NumberOfBandsUp  {num_bands_to_wannier_up}
Siesta2Wannier90.NumberOfBandsDown  {num_bands_to_wannier_down}
""")
#------------------------------------------------------------------------------#
        # Other arguments
        if others:
            f.write("""
############################################
#   Others
############################################
""")
            for key, value in others.items():
                f.write('{}\t\t{}\n'.format(key, value))


def write_struct_fdf(
        geom: Geometry, name: str, path="./opt",
        lattice_constant=None,
        unit='Ang',
        fmt='.8f'
):
    """
    Write name_STRUCT.fdf file for Siesta calculation
    Args:
        geom: sisl Geometry object
        lattice_constant: by default it's the cell[0]
    """
    struct_file = name + '_STRUCT.fdf'
    if not lattice_constant:
        lattice_constant = np.linalg.norm(geom.cell[0])
    lat_con = round(lattice_constant, 8)
    cell_raw = geom.cell
    # nomalized cell
    cell = cell_raw/lat_con
    na = geom.na
    xyz = geom.xyz
    num_sp = len(geom.atoms.atom)

    with open(os.path.join(path, struct_file), 'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
LatticeConstant     {lat_con} {unit}
%block LatticeVectors""")
        for v in cell:
            f.write(str('\n'+f' {{:{fmt}}}'*3).format(
                *v))
        f.write(f"""
%endblock LatticeVectors

NumberOfAtoms   {na}
AtomicCoordinatesFormat {unit}
%block AtomicCoordinatesAndAtomicSpecies""")
        fmt_str = '\n' + f' {{:{fmt}}}'*3 + ' {} #  {}: {}'
        for ia, a, isp in geom.iter_species():
            f.write(fmt_str.format(*xyz[ia, :], isp+1, ia+1, a.tag))
        f.write(f"""
%endblock AtomicCoordinatesAndAtomicSpecies

NumberOfSpecies  {num_sp}
%block ChemicalSpeciesLabel""")
        for i, a in enumerate(geom.atoms.atom):
            f.write('\n {} {} {}'.format(i + 1, a.Z, a.tag))
        f.write(f"""
%endblock ChemicalSpeciesLabel
""")


def write_optical_calc_fdf(
        name, path='./opt',
        energy_range=None,
        broaden=0,
        scissor=0,
        num_bands=None,
        kmesh=[20, 1, 1],
        polarization_type='unpolarized',
        polarize_vector=[1, 0, 0]
):
    """
    Write optical calculation file
    """
    filename = name + '.optical_calc.fdf'
    filepath = os.path.join(path, filename)
#     constant = 0.0734986176 # eV to Ry
    emin, emax = energy_range
#     emin *= constant
#     emax *= constant
#     broaden *= constant
#     scissor *= constant
    with open(filepath, 'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
OpticalCalculation      True
Optical.Energy.Minumum  {emin} eV
Optical.Energy.Maximum  {emax} eV
Optical.Broaden         {broaden} eV
Optical.Scissor         {scissor} eV""")
        if num_bands:
            f.write(f"""
Optical.NumberOfBands   {num_bands}
""")
        f.write("""
%block Optical.Mesh
""")
        f.write(' {} {} {}'.format(*kmesh))
        f.write(f"""
%endblock Optical.Mesh
Optical.PolarizationType   {polarization_type}
%block Optical.Vector""")
        f.write('\n {:.1f} {:.1f} {:.1f}'.format(*polarize_vector))
        f.write("""
%endblock Optical.Vector""")


def write_denchar_file(
        geom: Geometry, name, path='./opt',
        type_of_run='3D',
        plot_charge=True,
        plot_wavefunctions=True,
        coor_units='Ang',
        num_unit_cells=2,
        mesh_grid=4,
        box_extension=[1,5,5]
):
    """
    Write SystemLabel.denchar.fdf file for density charge calculation, to be 
    consumed by denchar program
    """
    filename = name + '.denchar.fdf'
    filepath = os.path.join(path, filename)
    num_sp = len(geom.atoms.atom)
    gTmp = geom.tile(num_unit_cells,0)
    xyz = gTmp.xyz
    cell = gTmp.cell
    origin = gTmp.center()
    origin[0] = 0
    xmax, ymax, zmax = np.max(xyz, axis=0) - origin
    xmin, ymin, zmin = np.min(xyz, axis=0) - origin
    # denchar will multiple these numbers by 1.1
    x1, y1, z1 = box_extension
    xmax += x1
    xmin -= x1
    ymax += y1
    ymin -= y1
    zmin -= z1
    zmax += z1
    xnpts, ynpts, znpts = np.around(np.array(
        [xmax-xmin, ymax-ymin, zmax-zmin]
    )*mesh_grid).astype(int)
    xaxis = origin + np.array([5, 0, 0])

    with open(filepath, 'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
SystemLabel             {name}
NumberOfSpecies         {num_sp}
%block ChemicalSpeciesLabel""")

        for i, a in enumerate(geom.atoms.atom):
            f.write('\n {} {} {}'.format(i + 1, a.Z, a.tag))
        f.write(f"""
%endblock ChemicalSpeciesLabel

Denchar.TypeOfRun       {type_of_run}
Denchar.PlotCharge      {plot_charge}
Denchar.PlotWaveFunctions   {plot_wavefunctions}

Denchar.CoorUnits       {coor_units}""")
        f.write("""
%block Denchar.PlaneOrigin
 {:.8f} {:.8f} {:.8f}
%endblock Denchar.PlaneOrigin
""".format(*origin))
        f.write("""
%block Denchar.X-Axis
 {:.8f} {:.8f} {:.8f}
%endblock Denchar.X-Axis
""".format(*xaxis))
        f.write(f"""
Denchar.MinX            {xmin:.8f} Ang
Denchar.MaxX            {xmax:.8f} Ang
Denchar.MinY            {ymin:.8f} Ang
Denchar.MaxY            {ymax:.8f} Ang
Denchar.MinZ            {zmin:.8f} Ang
Denchar.MaxZ            {zmax:.8f} Ang
Denchar.NumberPointsX   {xnpts:1d}
Denchar.NumberPointsY   {ynpts:1d}
Denchar.NumberPointsZ   {znpts:1d}
""")

# make write_win_file more well organized, well aligned, can you do that?

def write_win_file(
        geom: Geometry, name, path="./s2w",
        restart = None,
        tot_num_bands=None,
        num_ex_bands=None,
        num_wann=None,
        proj_orbs="C:pz",
        select_proj=None,
        kmesh=[12, 1, 1],
        dis_win_max=None,
        dis_win_min=None,
        dis_froz_max=None,
        dis_froz_min=None,
        kpoints_path="GXG",
        guiding_centres=True,
        wa_plot_sc=[3, 1, 1],
        kmesh_tol=1e-6,
        search_shells=36,
        fermi_energy=None,
        wannier_plot_mode='crystal'
):
    """
    Write input file for Wannier90 calculation
    Args:
        restart: restart from 'default', 'wannierise', 'plot', or 'transport'
        tot_num_bands: Total number of bands
        num_ex_bands: Number of excluded bands (from 1 to num_ex_bands)
        num_wann: Number of bands to be wannierzed
        proj_orbs: projected orbitals
        select_proj: selected projections
        kmesh: k point mesh, CAVEAT: currectly only works for 1D system
        dis_win_max: Maximum of disentangle energy window
        dis_win_min: Minimum of disentangle energy window
        dis_froz_max: Maximum of frozen energy window
        dis_froz_min: Minimum of frozen energy window
        kpoints_path: k points path
        guiding_centres: Use guiding centres or not
        wa_plot_sc: paramater for wannier_plot_supercell
        kmesh_tol: k mesh tolerance
        search_shells: search shells
        fermi_energy: fermi energy. If nothing is provide, read from .out file
            in currect directory
        wannier_plot_mode: crystal (default) or molecule
    """

    # by default we exclude all the s bands, this already gives very good result
    # the default values only work for DZP basis set!!!
    C_atoms = []
    for i, at in enumerate(geom.atoms):
        if at.Z == 6:
            C_atoms.append(i)
    if not num_wann:
        # this only works for DZP basis set
        num_wann = int(len(C_atoms)/2)
    num_bands = tot_num_bands - num_ex_bands if num_ex_bands else tot_num_bands

    if select_proj:
        proj_orb_idx = select_proj
    else:
        proj_orb_idx = f'1-{num_wann}'
    
    if not fermi_energy:
        # read fermi energy from siesta output
        fe = read_final_energy(name=name, path=path, which='fermi')
    else:
        fe = fermi_energy

    with open(os.path.join(path, f"{name}.win"), 'w') as f:
        f.write(f"! {User} created at {get_datetime()}\n")
        if restart:
            f.write(f'restart = {restart}\n')
        f.write(f"""
num_bands   =   {num_bands}
num_wann    =   {num_wann}""")
        if num_ex_bands:
            if isinstance(num_ex_bands, int):
                f.write(f"\nexclude_bands =  1-{num_ex_bands}")
            elif isinstance(num_ex_bands, str):
                f.write(f"\nexclude_bands = {num_ex_bands}")
        if dis_win_min != None:
            f.write(f"\ndis_win_min =  {dis_win_min+fe}")
        if dis_win_max != None:
            f.write(f"\ndis_win_max =  {dis_win_max+fe}")
        if dis_froz_min != None:
            f.write(f"\ndis_froz_min =  {dis_froz_min+fe}")
        if dis_froz_max != None:
            f.write(f"\ndis_froz_max =  {dis_froz_max+fe}")
        f.write(f"""
select_projections: {proj_orb_idx}

begin projections""")
        for orbs in proj_orbs.strip().split():
            f.write(f"\n{orbs}")
        wp1, wp2, wp3 = wa_plot_sc
        f.write(f"""
end projections

search_shells = {search_shells}
num_iter	=	200
write_hr	=	true
write_tb	=	true
write_xyz   =   true
translate_home_cell =   true
guiding_centres =  {guiding_centres}
iprint : 3
!trial_step  =   1.0

!bands_plot      =   true
wannier_plot    =   true
wannier_plot_supercell  =  {wp1}, {wp2}, {wp3}
wannier_plot_mode = {wannier_plot_mode}
kmesh_tol = {kmesh_tol}

begin unit_cell_cart
Ang""".format(*wa_plot_sc))
        # write unit cell
        for i in range(len(geom.cell)):
            f.write("\n  {:.10f}\t{:.10f}\t{:.10f}".format(*geom.cell[i]))
        f.write("""
end unit_cell_cart

begin kpoint_path""")
        # write k points path
        # in fractional units w.r.t. 2*pi/a
        for i in range(len(kpoints_path)-1):
            tmp_str = "\n"+(" {}" + " {:.5f}"*3)*2
            k0, k1 = kpoints_path[i:i+2]
            _K0 = np.array(kpoints_dict[k0][1])
            _K1 = np.array(kpoints_dict[k1][1])
            f.write(tmp_str.format(
                k0, *_K0, k1, *_K1))
        f.write("""
end kpoint_path

mp_grid: {} {} {}
begin kpoints""".format(*kmesh))
        # write k points
        for i in range(kmesh[0]):
            f.write("\n\t{:.8f}\t{:.8f}\t{:.8f}".format(i/kmesh[0], 0, 0))
        f.write("""
end kpoints

begin atoms_cart
Ang""")
        # write atom coordinates
        for i in range(len(geom.xyz)):
            f.write("\n{}\t{:.10f}\t{:.10f}\t{:.10f}".format(
                geom.atoms[i].symbol, *geom.xyz[i]))
        f.write("""
end atoms_cart""")




def write_wannier90insiesta_runfile(name: str):
    """
    This is only a sample. values are not parameterized.
    """

    run_file = name + 'RUN.fdf'
    struct_file = name + 'STRUCT.fdf'

    with open(f'./wins/{run_file}', 'a') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
############################################
# Interface with Wannier90
############################################

NumberOfBandManifoldsForWannier   1

%block WannierManifolds
  1                         # Sequential index of the manifold, from 1 to NumberOfBandManifoldsForWannier
  13    26                  # Indices of the initial and final band of the manifold
  6                         # Number of bands for Wannier transformation
  4  17  30  43  56  69     # Indices of the orbitals that will be used as localized trial orbitals
  num_iter 500              # Number of iterations for the minimization of \Omega
  wannier_plot 3            # Plot the Wannier function 
  fermi_surface_plot False  # Plot the Fermi surface
  write_hr                  # Write the Hamiltonian in the WF basis
  write_tb                  # Write the Hamiltonian in the WF basis
  -30.0     -0              # Bottom and top of the outer energy window for band disentanglement (in eV)
  -30.0     -20             # Bottom and top of the inner energy window for band disentanglement (in eV)
%endblock WannierManifolds

%block kMeshforWannier
   12  1  1  
%endblock kMeshforWannier

Siesta2Wannier90.UnkGrid1       30
Siesta2Wannier90.UnkGrid2       30
Siesta2Wannier90.UnkGrid3       30

Wannier90_in_SIESTA_compute_unk .true.
""")



# Way to calc phonon:
# siesta/Utils/Vibra/Src/fcbuild < name.fcbuild.fdf
# siesta/siesta < name.ifc.fdf > name.ifc.out
# siesta/Utils/Vibra/Src/vibrator < name.fcbuild.fdf

def write_fcbuild_file(
        geom: Geometry, name: str, path='./phonon',
        supercell=[1, 0, 0],
        bandlines_kpath='GX',
        bandlines_nkpts=200,
):
    """
    Create fcbuild file for utility program fcbuild
    Args:
        mpgrid: Monkhorst-Pack grid
        supercell: 0-1, 1-3, 2-5,..., so [1,0,0] actually means [3,1,1]
        meash_cutoff: mesh cutoff
        bandlines_kpath: band lines k path
        bandlines_nkpts: Number of k points from 0 to 2*pi/a
    """

    fcbuild_file = name + '.fcbuild.fdf'
    # commonly used atoms
    mass_list = {'H': 1.0079, 'C': 12.0107,
                 'N': 14.0067, 'O': 15.9994,
                 'Cl': 35.4530, 'V': 50.9415,
                 'Cu': 63.5460, 'Zn': 65.3900,
                 'Br': 79.9040}

    lc = np.linalg.norm(geom.cell[0])
    cellfrac = geom.cell/lc

    with open(os.path.join(path, fcbuild_file), 'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        sc1, sc2, sc3 = supercell
        f.write(f"""
SystemName           {name}
SystemLabel          {name}

NumberOfSpecies      {geom.atoms.nspecie}
NumberOfAtoms        {geom.na}

Eigenvectors         True
SuperCell_1          {sc1} 
SuperCell_2          {sc2}    
SuperCell_3          {sc3} 

BandLinesScale       pi/a
%block BandLines""")
        for i, bdk in enumerate(bandlines_kpath):
            tmp = kpoints_dict[bdk]
            ktmp = 2*np.array(tmp[1]) # becuase bandlines scale is pi/a
            kpt = 1 if i == 0 else int(bandlines_nkpts*np.linalg.norm(
                np.array(ktmp)/2-np.array(kpoints_dict[bandlines_kpath[i-1]][1])))
            f.write("\n{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
                kpt, *ktmp, tmp[0]))
        f.write("""
%endblock BandLines

%block ChemicalSpeciesLabel""")
        for i, a in enumerate(geom.atoms.atom):
            f.write(f"\n{i+1}\t{a.Z}\t{a.symbol}")
        f.write(f"""
%endblock ChemicalSpeciesLabel

LatticeConstant      {lc} Ang
%block LatticeVectors""")
        for i in range(len(geom.cell)):
            f.write("\n{:.8f}\t{:.8f}\t{:.8f}".format(*cellfrac[i]))
        f.write("""
%endblock LatticeVectors

AtomicCoordinatesFormat NotScaledCartesianAng
%block AtomicCoordinatesAndAtomicSpecies""")
        for ia, a, isp in geom.iter_species():
            f.write("\n{:.8f}\t{:.8f}\t{:.8f}\t{}  {}\t# {} {}".format(
                *geom.xyz[ia, :], isp+1, mass_list[a.tag], ia+1, a.tag))
        f.write(f"""
%endblock AtomicCoordinatesAndAtomicSpecies
""")


def write_ifc_file(
        geom: Geometry, name: str, path='./phonon',
        pseudo_path=None,
        xc_functional='GGA',
        xc_authors='PBE',
        basis_size='DZP',
        pao_energy_shift=100,
        mpgrid=[21, 1, 1],
        mesh_cutoff=400,
        max_scf_iter=500,
        mixer_method='Pulay',
        mixer_weight=0.25,
        mixer_history=6,
        scf_H_tol=1e-3,
):
    """
    Create ifc file for Siesta phonon calculation
    Args:
        pseudo_path: pseudopotential file path
        xc_functional: default GGA
        xc_authors: functional flavor, default PBE
        basis_size: default DZP
        pao_energy_shift: PAO.EnergyShift, default 100 meV
        mpgrid: Monkhorst-Pack grid, default 21x1x1
        mesh_cutoff: mesh cutoff
        max_scf_iter: maximum number of SCF iterations
        mixer_method: Pulay (default) or Broyden, Linear
        mixer_weight: SCF mixing weight, default 0.25
        mixer_history: SCF mixing history, default 6
        scf_H_tol: SCF Hafmiltonian tolerance
    """
    # prepare pseudopotential files
    species = set()
    for a in geom.atoms:
        species.add(a.tag)
    species = ','.join(species)
    copy_psf_files(path_from=pseudo_path, path_to=path, 
        elements=species, functional=xc_functional)

    ifc_file = name + '.ifc.fdf'
    mp1, mp2, mp3 = mpgrid
    with open(os.path.join(path, ifc_file), 'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
SystemName           {name}
SystemLabel          {name}

NumberOfSpecies      {geom.atoms.nspecie}
NumberOfAtoms        < FC.fdf

XC.functional        {xc_functional}
XC.authors           {xc_authors}
PAO.BasisSizes       {basis_size}
PAO.EnergyShift      {pao_energy_shift} meV
MeshCutoff           {mesh_cutoff} Ry
%block kgrid.MonkhorstPack
    {mp1}  0   0   0.0 
    0   {mp2}  0   0.0
    0   0   {mp3}  0.0
%endblock kgrid.MonkhorstPack

MaxSCFIterations     {max_scf_iter}
SCF.Mixer.Method     {mixer_method}
SCF.Mixer.Weight     {mixer_weight}
SCF.Mixer.History    {mixer_history}
SCF.H.Tolerance      {scf_H_tol} eV

%block ChemicalSpeciesLabel""")
        for i, a in enumerate(geom.atoms.atom):
            f.write(f"\n{i+1}\t{a.Z}\t{a.symbol}")
        f.write("""
%endblock ChemicalSpeciesLabel

LatticeConstant      < FC.fdf
LatticeVectors       < FC.fdf
AtomicCoordinatesFormat             < FC.fdf     
AtomicCoordinatesAndAtomicSpecies   < FC.fdf

MD.TypeOfRun    < FC.fdf   # Compute the interatomic force constants matrix 
MD.FCfirst      < FC.fdf   # Index of first atom to displace
MD.FClast       < FC.fdf   # Index of the last atom to displace
MD.FCdispl      < FC.fdf   # Displacement to use for the computation of the
                           # interatomic force constant matrix
                           # (Remember that the second derivative of the 
                           # energy with respect the displacement of two
                           # atoms is computed by means of a 
                           # finite difference derivative of the forces)
""")


# def write_xcrysden_shell_script(
#         path,
#         keyword,
#         file_format="cube",
#         bash_file="export_xcrysden.sh",
#         xcrysden_state_file="state_real.xcrysden"):
#     """
#     Write shell script to run XCrySDen automatically.
#     The xcrysden state file should be named as state.xcrysden
#     """

#     file_path = os.path.join(path, bash_file)
#     with open(file_path, 'w') as f:
#         f.write(f"""
# for input in `ls {keyword}`; do
#     cp {xcrysden_state_file} tmp.xcrysden
#     filename="${{input%.*}}.png"
#     echo "
# scripting::printToFile $filename windowdump
# exit 0" >> tmp.xcrysden;
#     xcrysden --{file_format} $input --script tmp.xcrysden;
#     rm -f tmp.xcrysden;
# done
# """)


def write_sbatch_file(name, path, program='siesta', cluster='htc',
    time=10, num_nodes=1, num_tasks_per_node=48,
    job_name=None, memory=None, phonon=False):
    """
    Prepare the bash file for sbatch

    Args:
        program: 'siesta' or 'orca'
        cluster: 'htc' or 'arc' or 'all'
        time: in hours
        num_nodes: number of nodes
        num_tasks_per_node: number of tasks per node
        job_name: job name
        memory: by default 8000M per task
        phonon: Run siesta phonon calculation or not
    """

    if not memory:
        memory = num_tasks_per_node*8000
    if not job_name:
        p = path.split('/')[-1]
        job_name = name+p
    if program == 'siesta':
        load_module = 'module load Siesta/4.1.5-foss-2020a'
        run_cmd = 'mpirun siesta < $name\_RUN.fdf > $name.out'
        if phonon:
            run_cmd = 'mpirun siesta < $name\.ifc.fdf > $name.ifc.out'
    elif program == 'orca':
        load_module = 'module load ORCA/5.0.3-gompi-2021b'
        run_cmd = '$EBROOTORCA/orca $name.inp > $name.out'
    
    if time <= 12:
        partition = 'short'
    elif time <= 48:
        partition = 'medium'
    else:
        partition = 'long'
    with open(os.path.join(path, f'run_{program}.sh'),'w') as sh:
        sh.write(f"""#!/bin/bash

#SBATCH --clusters={cluster}
#SBATCH --nodes={num_nodes}
#SBATCH --time={time}:00:00
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --partition={partition}
#SBATCH --mem={memory}
#SBATCH --job-name={job_name}
#SBATCH --mail-user=fanmiao.kong@materials.ox.ac.uk
#SBATCH --mail-type=ALL

{load_module}

name="{name}"
{run_cmd}""")




def write_submit_all_file(path, dirs:list, program='siesta'):
    """
    Write bash file to submit all jobs
    Args:
        path: path to write submit_all.sh file
        dirs: directories that contains run_{program} files
        program: 'siesta' or 'orca'
    """
    if not isinstance(dirs, str):
        dirs = ' '.join(dirs)
    with open(os.path.join(path, 'submit_all.sh'), 'w') as sh:
        sh.write(f"""
for name in {dirs}; do
    cd $name
    sbatch run_{program}.sh
    cd ..
done
""")



def copy_psf_files(
    path_from=None, 
    path_to=None, 
    elements=None, 
    functional=None):
    """
    Copy pseudopotential files
    Args:
        path: destination
        elements: string of elements, eg. 'C,H'
        functional: 'GGA' or 'LDA'
    """
    from shutil import copyfile
    if not elements:
        raise ValueError('Please provide elements')
    if not functional:
        raise ValueError('Please provide functional')
    if not path_to:
        raise ValueError('Please provide path_to')
    if not path_from:
        if hostName == 'OUMS-OXBOX':
            path_from = f'/mnt/d/kfm/Computation/dft/pseudopotentials/{functional}'
        else:
            raise ValueError(f'Please provide pseudo_path that stores pseudopotential \
            files')
    elements = elements.strip().split(',')
    for e in elements:
        e = e.strip()
        psf_src = os.path.join(path_from, f'{e}.psf')
        psf_dst = os.path.join(path_to, f'{e}.psf')
        copyfile(psf_src, psf_dst)


def write_orca_input_file(name, path, spin_multiplicity:int, charge=0,
    functional='B3LYP', basis_set='6-31G**', optimize_control='TightOPT',
    SCF_control='TightSCF', num_processor=48, plot_spin_density=True):
    """
    Write Orca input file
    You should provide the name.xyz file
    """
    inp_file = f'{name}.inp'
    multiplets_name = {1:'singlet', 2:'doublet', 3:'triplet', 4:'quartet',
        5:'quintet', 6:'sextet', 7:'septet', 8:'octet'}
    
    with open(os.path.join(path,inp_file),'w') as f:
        f.write(f"# {User} created at {get_datetime()}\n")
        f.write(f"""
!UKS {functional} {basis_set} {optimize_control} {SCF_control} RIJCOSX
!UNO

%base "{name}"

%pal 
nprocs {num_processor}
end

*xyzfile {charge} {spin_multiplicity} {name}.xyz
""")
        if plot_spin_density:
            f.write(f"""
%plots
dim1 120
dim2 120
dim3 120
Format Gaussian_cube
SpinDens("{name}_{multiplets_name[spin_multiplicity]}.cube");
end""")
