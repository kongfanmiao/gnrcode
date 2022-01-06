from sisl import Geometry
import numpy as np
import matplotlib.pyplot as plt
import os
from .geometry import *
from .tools import *


kpoints_dict = {
    "G": ("$\Gamma$", [0., 0., 0.]),
    "X": ("$X$", [0.5, 0., 0.]),
    "M": ("$M$", [0.5, 0.5, 0.]),
    "K": ("$K$", [2.0 / 3, 1.0 / 3, 0.]),
}


def create_siesta_runfile(
        geom: Geometry, name: str, path="./opt",
        mpgrid=[21, 1, 1],
        bandlines_kpath='GXG',
        bandlines_nkpts=200,
        spin_polarized=False,
        spin_afm=True,
        spin_orbit=False,
        soc_strength=1.0,
        denchar=False,
        wfs_write_for_kpts=False,
        wavefunc_kpts: list = None,
        wfs_write_for_bands=False,
        wfs_bands_range: list = None,
        E_field: list = None,
        slab_dipole_correction=False,
        optical_calc=False,
        wannier90=False,
        num_bands_to_wannier=None,
        num_bands_to_wannier_up=None,
        num_bands_to_wannier_down=None,
        s2w_grid=[30, 30, 30],
        diag_algorithm="Divide-and-Conquer",
        cdf=True,
        mixer_weight=0.25,
        mixer_history=6,
        variable_cell=True
):
    """
    Write Siesta input file
    Args:
        mpgrid: MonkHorst Pack grid
        bandlines_kpath: k path to calculation band structure
        bandlines_nkpts: Number of k points for band line from 0 to 2*pi/a
        spin_polarized: Spin polarized or unpolarized calculation
        spin_afm: Antiferromagnetic configuration for spin initialization, if False,
            then it means ferromagnetic configuration
        spin_orbit: Includes spin-orbit coupling in calculation
        soc_strength: Spin-orbit coupling strength
        denchar: Use DENCHAR to calculate the density of charge. Required for 
            wavefunctions calculation
        wfs_write_for_kpts: Calculate and plot wavefunctions or not
        wavefunc_kpts: At which k points that the wavefunctions are calculated.
            Note that by default the k points is scaled by pi/a. MUST use float number!
        wfs_write_for_bands: Write wavefunctions for given bands
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
        s2w_grid: Mesh grid points along three lattice vector directions to plot
            wavefunctions during Siesta2Wannier90 calculation
        diag_algorithm: By default it's Divide-and-Conquer. If job is to send to
            ARC, then use "expert" instead
        cdf: Use NetCDF utility or not
        mixer_weight: SCF mixing weight
        mixer_history: SCF mixer history
        variable_cell: Fix the cell during MD relaxation or not
    """
    # Some other default parameters:
    #   PAO.BasisSize       DZP
    #   PAO.BasisType       split
    #   PAO.EnergyShift     0.02 Ry
    #   SCF.DM.Tolerance    1e-4
    #   SolutionMethod      diagon
    #   SaveRho             F
    #   WriteVoronoiPop     F
    #   NetCharge           0

    run_file = name + '_RUN.fdf'
    struct_file = name + '_STRUCT.fdf'

    def check_directory(calc_type):
        # for some calculations, create new directory for them. This method will
        # raise error if we are still working in the ./opt directory
        if path == "./opt":
            raise ValueError(f"Dont't Work in directory ./opt for {calc_type}")

    with open(os.path.join(path, run_file), 'w') as f:
        f.write("""%include {}
SystemName              {}
SystemLabel             {}

############################################
#   Parameters
############################################
XC.functional           GGA
XC.authors              PBE
MeshCutoff              400 Ry
%block kgrid_Monkhorst_Pack
    {}  0   0   0.0 
    0   {}  0   0.0
    0   0   {}  0.0
%endblock kgrid_Monkhorst_Pack

############################################
#   Molecular Dynamics
############################################
MD.TypeOfRun            CG  # coordinate optimization by conjugation gradient
MD.Steps                1000
MD.MaxDispl             0.05  Ang
MD.MaxForceTol          0.01 eV/Ang
MD.VariableCell         {}
MD.UseSaveXV            T
MD.UseSaveCG            T

############################################
#   SCF
############################################
Diag.Algorithm          {}
DM.UseSaveDM            T
DM.History.Depth        6
MaxSCFIterations        500
SCF.Mixer.Weight        {}
SCF.Mixer.History       {}

############################################
#   Band Structures
############################################
# BandLines_kpathScale  pi/a
%block BandLines_kpath""".format(
            struct_file, name, name, *mpgrid, variable_cell,
            diag_algorithm, mixer_weight, mixer_history
        ))
        for i, bdk in enumerate(bandlines_kpath):
            tmp = kpoints_dict[bdk]
            kpt = 1 if i == 0 else int(bandlines_nkpts*np.linalg.norm(
                np.array(tmp[1])-np.array(kpoints_dict[bandlines_kpath[i-1]][1])))
            f.write("\n{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
                kpt, *tmp[1], tmp[0]))
        f.write("""
%endblock BandLines_kpath
WriteBands              T   # Write .bands file

############################################
#   Output Settings
############################################
COOP.write              F   # Crystal-Orbital Overlap, write 
                            # SystemLabel.fullBZ.WFSX and SystemLabel.HSX file
WriteMullikenPop        1   # Write atomic and orbital charges
WriteEigenvalues        T   # Write eigenvalues for sampling k points
SaveHS                  T   # Write Hamiltonian and overlap matrices, in .HSX file
WriteCoorXmol           T   # Write optimized structure coordinates in .xyz file
WriteCoorStep           T   # Write coordinate in every MD step to .XV file
WriteMDXmol             F   # Write .ANI file readable by XMoL for animation of MD""")
        if cdf:
            f.write("""
CDF.Save                T
CDF.Compress            3
WFS.Energy.Min          -30 eV
WFS.Energy.Max          30 eV
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
DM.InitSpin.AF          {spin_afm}""")
        if spin_mode == 'spin-orbit':
            f.write(f"""
Spin.OrbitStrength          {soc_strength}
""")
#------------------------------------------------------------------------------#
        # use DENCHAR program to plot wavefunction
        if denchar:
            denchar_file_name = name + ".denchar.fdf"
            f.write(f"""
############################################
#   Use Utility Program DENCHAR
############################################
WriteDenchar            T   # write .PLD and .DIM file to be used by Denchar
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
            for kpt in wavefunc_kpts:
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
SlabDipoleCorrection    {}
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
            f.write("""
############################################
#   Interface with Wannier90
############################################
Siesta2Wannier90.WriteMmn       true
Siesta2Wannier90.WriteAmn       true
Siesta2Wannier90.WriteEig       true
Siesta2Wannier90.WriteUnk       true
Siesta2Wannier90.UnkGridBinary  true
Siesta2Wannier90.UnkGrid1       {}
Siesta2Wannier90.UnkGrid2       {}
Siesta2Wannier90.UnkGrid3       {}""".format(*s2w_grid))
            if spin_mode == "non-polarized":
                # NumberOfBands is by default all occupied bands
                if num_bands_to_wannier:
                    f.write(f"""
Siesta2Wannier90.NumberOfBands  {num_bands_to_wannier}""")
            else:
                # NumberOfBandsUp/Down are by default the same as NumberOfBands
                try:
                    f.write(f"""
Siesta2Wannier90.NumberOfBandsUp  {int(num_bands_to_wannier_up)}
Siesta2Wannier90.NumberOfBandsDown  {int(num_bands_to_wannier_down)}
""")
                except:
                    pass


def write_struct_fdf(
        geom: Geometry, name: str, path="./opt",
        lattice_constant=1.0,
        unit='Ang',
        fmt='.8f'
):
    """
    Write name_STRUCT.fdf file for Siesta calculation
    Args:
        geom: sisl Geometry object
        lattice_constant: by default it's 1.0 Angstrom
    """
    struct_file = name + '_STRUCT.fdf'
    lat_con = round(lattice_constant, 8)
    cell_raw = geom.cell
    # nomalized cell
    cell = cell_raw/lat_con
    na = geom.na
    xyz = geom.xyz
    num_sp = len(geom.atoms.atom)

    with open(os.path.join(path, struct_file), 'w') as f:
        f.write(f"""LatticeConstant     {lat_con} {unit}
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
        mesh_grid=4
):
    """
    Write SystemLabel.denchar.fdf file for density charge calculation, to be 
    consumed by denchar program
    """
    filename = name + '.denchar.fdf'
    filepath = os.path.join(path, filename)
    num_sp = len(geom.atoms.atom)
    xyz = geom.xyz
    cell = geom.cell
    geom_size = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    ymin, zmin = -geom_size[1:]/2
    ymax, zmax = geom_size[1:]/2
    xmin = -cell[0, 0]/2
    xmax = 3*cell[0, 0]/2
    xnpts, ynpts, znpts = np.around(np.array(
        [xmax-xmin, ymax-ymin, zmax-zmin]
    )*mesh_grid).astype(int)
    center = geom.center()
    xaxis = center + np.array([5, 0, 0])

    with open(filepath, 'w') as f:
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
""".format(*center))
        f.write("""
%block Denchar.X-Axis
 {:.8f} {:.8f} {:.8f}
%endblock Denchar.X-Axis
""".format(*xaxis))
        f.write(f"""
Denchar.MaxX            {xmax:.8f} Ang
Denchar.MaxY            {ymax:.8f} Ang
Denchar.MaxZ            {zmax:.8f} Ang
Denchar.MinX            {xmin:.8f} Ang
Denchar.MinY            {ymin:.8f} Ang
Denchar.MinZ            {zmin:.8f} Ang
Denchar.NumberPointsX   {xnpts:1d}
Denchar.NumberPointsY   {ynpts:1d}
Denchar.NumberPointsZ   {znpts:1d}
""")


def create_win_file(
        geom: Geometry, name, path="./s2w",
        tot_num_bands=None,
        num_ex_bands=None,
        num_wann=None,
        proj_orbs="C:pz",
        kmesh=[12, 1, 1],
        dis_win_max=None,
        dis_win_min=None,
        dis_froz_max=None,
        dis_froz_min=None,
        kpoints_path="GXG",
        guiding_centres=False,
        wa_plot_sc=[3, 1, 1],
        kmesh_tol=0.0001
):
    """
    Write input file for Wannier90 calculation
    Args:
        tot_num_bands: Total number of bands
        num_ex_bands: Number of excluded bands (from 1 to num_ex_bands)
        num_wann: Number of bands to be wannierzed
        proj_orbs: projected orbitals
        kmesh: k point mesh, CAVEAT: currectly only works for 1D system
        dis_win_max: Maximum of disentangle energy window
        dis_win_min: Minimum of disentangle energy window
        dis_froz_max: Maximum of frozen energy window
        dis_froz_min: Minimum of frozen energy window
        kpoints_path: k points path
        guiding_centres: Use guiding centres or not
        wa_plot_sc: paramater for wannier_plot_supercell
        kmesh_tol: k mesh tolerance
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

    # proj_orb = ''
    # for i in range(num_wann):
    #     proj_orb += f'{2*i+1} '
    proj_orb_idx = f'1-{num_wann}'

    if not dis_win_min:
        dis_win_min = -30
    if not dis_win_max:
        dis_win_max = 0

    with open(os.path.join(path, f"{name}.win"), 'w') as f:
        f.write(f"""
num_bands   =   {num_bands}
num_wann    =   {num_wann}""")
        if num_ex_bands:
            f.write(f"\nexclude_bands =  1-{num_ex_bands}")
        if dis_win_min:
            f.write(f"\ndis_win_min =  {dis_win_min}")
        if dis_win_max:
            f.write(f"\ndis_win_max =  {dis_win_max}")
        if dis_froz_min:
            f.write(f"\ndis_froz_min =  {dis_froz_min}")
        if dis_froz_max is not None:
            f.write(f"\ndis_froz_max =  {dis_froz_max}")
        f.write(f"""
select_projections: {proj_orb_idx}

begin projections""")
        for orbs in proj_orbs.strip().split():
            f.write(f"\n{orbs}")
        f.write("""
end projections

num_iter	=	500
write_hr	=	true
write_tb	=	true
write_xyz   =   true
translate_home_cell =   true
guiding_centres =  {}
iprint : 3
!trial_step  =   1.0

!bands_plot      =   true
wannier_plot    =   true
wannier_plot_supercell  =  {}, {}, {}
!wannier_plot_mode = molecule
kmesh_tol = {}

begin unit_cell_cart
Ang""".format(guiding_centres, *wa_plot_sc, kmesh_tol))
        # write unit cell
        for i in range(len(geom.cell)):
            f.write("\n  {:.10f}\t{:.10f}\t{:.10f}".format(*geom.cell[i]))
        f.write("""
end unit_cell_cart

begin kpoint_path""")
        # write k points path
        for i in range(len(kpoints_path)-1):
            tmp_str = "\n"+(" {}" + " {:.5f}"*3)*2
            k0, k1 = kpoints_path[i:i+2]
            f.write(tmp_str.format(
                k0, *kpoints_dict[k0][1], k1, *kpoints_dict[k1][1]))
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
            f.write("{}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(
                geom.atoms[i].symbol, *geom.xyz[i]))
        f.write("""
end atoms_cart""")


def create_wannier90insiesta_runfile(name: str):
    """
    This is only a sample. values are not parameterized.
    """

    run_file = name + 'RUN.fdf'
    struct_file = name + 'STRUCT.fdf'

    with open(f'./wins/{run_file}', 'a') as f:
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


def create_fcbuild_file(
        geom: Geometry, name: str, path='./phonon',
        mpgrid=[21, 1, 1],
        supercell=[1, 0, 0],
        mesh_cutoff=400,
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
        f.write("""
SystemName           {}
SystemLabel          {}

NumberOfSpecies      {}
NumberOfAtoms        {}

PAO.BasisSizes       DZP
Eigenvectors         .true.
%block kgrid_Monkhorst_Pack
    {}  0   0   0.0 
    0   {}  0   0.0
    0   0   {}  0.0
%endblock kgrid_Monkhorst_Pack
MeshCutoff           {} Ry

SuperCell_1          {} 
SuperCell_2          {}    
SuperCell_3          {} 

BandLines_kpathScale       pi/a
%block BandLines_kpath""".format(
            name, name, geom.atoms.nspecie, geom.na, *mpgrid, mesh_cutoff, *supercell))
        for i, bdk in enumerate(bandlines_kpath):
            tmp = kpoints_dict[bdk]
            kpt = 1 if i == 0 else int(bandlines_nkpts*np.linalg.norm(
                np.array(tmp[1])-np.array(kpoints_dict[bandlines_kpath[i-1]][1])))
            f.write("\n{}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(
                kpt, *tmp[1], tmp[0]))
        f.write("""
%endblock BandLines_kpath

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


def create_ifc_file(
        geom: Geometry, name: str, path='./phonon',
        mpgrid=[21, 1, 1],
        mesh_cutoff=400
):
    """
    Create ifc file for Siesta phonon calculation
    """

    ifc_file = name + '.ifc.fdf'
    with open(os.path.join(path, {ifc_file}), 'w') as f:
        f.write("""
SystemName           {}
SystemLabel          {}

NumberOfSpecies      {}
NumberOfAtoms        < FC.fdf
PAO.BasisSizes       DZP
MeshCutoff           {} Ry

%block kgrid_Monkhorst_Pack
    {}  0   0   0.0 
    0   {}  0   0.0
    0   0   {}  0.0
%endblock kgrid_Monkhorst_Pack

%block ChemicalSpeciesLabel""".format(
            name, name, geom.atoms.nspecie, *mpgrid, mesh_cutoff
        ))
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


def write_xcrysden_shell_script(
        path,
        keyword,
        file_format="cube",
        bash_file="export_xcrysden.sh",
        xcrysden_state_file="state_real.xcrysden"):
    """
    Write shell script to run XCrySDen automatically.
    The xcrysden state file should be named as state.xcrysden
    """

    file_path = os.path.join(path, bash_file)
    with open(file_path, 'w') as f:
        f.write(f"""
for input in `ls {keyword}`; do
    cp {xcrysden_state_file} tmp.xcrysden
    filename="${{input%.*}}.png"
    echo "
scripting::printToFile $filename windowdump
exit 0" >> tmp.xcrysden;
    xcrysden --{file_format} $input --script tmp.xcrysden;
    rm -f tmp.xcrysden;
done
""")
