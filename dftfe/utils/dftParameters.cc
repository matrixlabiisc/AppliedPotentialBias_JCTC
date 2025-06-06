// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Phani Motamarri, Sambit Das
//
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <dftParameters.h>
#include <fstream>
#include <iostream>



namespace dftfe
{
  namespace internalDftParameters
  {
    void
    declare_parameters(dealii::ParameterHandler &prm)
    {
      prm.declare_entry(
        "WRITE STRUCTURE ENERGY FORCES DATA POST PROCESS",
        "false",
        dealii::Patterns::Bool(),
        R"([Standard] Write ground-state atomistics data to a file (structureEnergyForcesGSData*.txt) with the suffix number in the file-name denoting the geometry relaxation step number. Order: number of atoms, lattice vectors (see format for DOMAIN BOUNDING VECTORS), structure, electronic free energy, internal energy, ionic forces and finally the cell stress. Structure format is four columns with the first column being atomic number and the next three columns in fractional coordinates for periodic and semi-periodic systems and Cartesian coordinates with origin at the domain center for non-periodic systems. Ionic forces are negative of gradient of DFT free energy with respect to ionic positions with the first, second and third column in each row corresponding to the x,y and z components. Cell stress is negative of gradient of the DFT free energy with respect to affine strain components scaled by volume. Cell stress is printed as sigma\_{ij} with i denoting the row index and j denoting the column index of the stress tensor. Atomic units are used everywhere. Default: false.)");

      prm.declare_entry(
        "REPRODUCIBLE OUTPUT",
        "false",
        dealii::Patterns::Bool(),
        "[Developer] Limit output to what is reproducible, i.e. don't print timing or absolute paths. This parameter is only used for testing purposes.");

      prm.declare_entry(
        "KEEP SCRATCH FOLDER",
        "false",
        dealii::Patterns::Bool(),
        "[Advanced] If set to true this option does not delete the dftfeScratch folder when the dftfe object is destroyed. This is useful for debugging and code development. Default: false.");

      prm.declare_entry(
        "MEM OPT MODE",
        "false",
        dealii::Patterns::Bool(),
        "[Adavanced] Uses algorithms which have lower peak memory but with a marginal performance degradation. Default: true.",
        true);


      prm.enter_subsection("GPU");
      {
        prm.declare_entry(
          "AUTO GPU BLOCK SIZES",
          "true",
          dealii::Patterns::Bool(),
          "[Advanced] Automatically sets total number of kohn-sham wave functions and eigensolver optimal block sizes for running on GPUs. If manual tuning is desired set this parameter to false and set the block sizes using the input parameters for the block sizes. Default: true.");

        prm.declare_entry(
          "FINE GRAINED GPU TIMINGS",
          "false",
          dealii::Patterns::Bool(),
          "[Developer] Print more fine grained GPU timing results. Default: false.");


        prm.declare_entry(
          "SUBSPACE ROT FULL CPU MEM",
          "true",
          dealii::Patterns::Bool(),
          R"([Developer] Option to use full NxN memory on CPU in subspace rotation and when mixed precision optimization is not being used. This reduces the number of MPI\_Allreduce communication calls. Default: true.)");

        prm.declare_entry(
          "USE GPUDIRECT MPI ALL REDUCE",
          "false",
          dealii::Patterns::Bool(),
          R"([Adavanced] Use GPUDIRECT MPI\_Allreduce. This route will only work if DFT-FE is either compiled with NVIDIA NCCL library or withGPUAwareMPI=ON. Both these routes require GPU Aware MPI library to be available as well relevant hardware. If both NVIDIA NCCL library and withGPUAwareMPI modes are toggled on, the NCCL mode takes precedence. Also note that one MPI rank per GPU can be used when using this option. Default: false.)");

        prm.declare_entry(
          "USE DCCL",
          "false",
          dealii::Patterns::Bool(),
          R"([Adavanced] Use NCCL/RCCL for GPUDIRECT communications. Default: false.)");

        prm.declare_entry(
          "USE ELPA GPU KERNEL",
          "false",
          dealii::Patterns::Bool(),
          "[Advanced] If DFT-FE is linked to ELPA eigensolver library configured to run on GPUs, this parameter toggles the use of ELPA GPU kernels for dense symmetric matrix diagonalization calls in DFT-FE. ELPA version>=2020.11.001 is required for this feature. Default: false.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Post-processing Options");
      {
        prm.declare_entry(
          "WRITE WFC FE MESH",
          "false",
          dealii::Patterns::Bool(),
          R"([Standard] Writes DFT ground state wavefunction solution fields (FEM mesh nodal values) to wfcOutput.vtu file for visualization purposes. The wavefunction solution fields in wfcOutput.vtu are named wfc\_s\_k\_i in case of spin-polarized calculations and wfc\_k\_i otherwise, where s denotes the spin index (0 or 1), k denotes the k point index starting from 0, and i denotes the Kohn-Sham wavefunction index starting from 0. In the case of geometry optimization, the wavefunctions corresponding to the last ground-state solve are written.  Default: false.)");

        prm.declare_entry(
          "PRINT KINETIC ENERGY",
          "false",
          dealii::Patterns::Bool(),
          R"([Standard] Prints the Kinetic energy of the electrons.  Default: false.)");

        prm.declare_entry(
          "WRITE DENSITY FE MESH",
          "false",
          dealii::Patterns::Bool(),
          R"([Standard] Writes DFT ground state electron-density solution fields (FEM mesh nodal values) to densityOutput.vtu file for visualization purposes. The electron-density solution field in densityOutput.vtu is named density. In case of spin-polarized calculation, two additional solution fields- density\_0 and density\_1 are also written where 0 and 1 denote the spin indices. In the case of geometry optimization, the electron-density corresponding to the last ground-state solve is written. Default: false.)");

        prm.declare_entry(
          "WRITE TOTAL ELECTROSTATICS POTENTIAL",
          "false",
          dealii::Patterns::Bool(),
          R"([Standard] Writes DFT ground state \phi(x) + V_loc(x) solution fields (FEM mesh nodal values) to potentialOutput.vtu file for visualization purposes.  Default: false.)");


        prm.declare_entry(
          "WRITE DENSITY QUAD DATA",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Writes DFT ground state electron-density solution fields at generally non-uniform quadrature points to a .txt file for post-processing. There will be seven columns (in case of collinear spin polarization) and 6 columns in case of spin-restricted calculation. The first column is the quadrature point index. The next three columns are the quadrature point cartesian coordinates (non-uniform grid with origin at cell center), fifth column is the quadrature integration weight incorporating the determinant of FE cell jacobian, and the sixth and seventh columns are the spin-up and spin-down densities in case of collinear spin polarization. In case of spin-restricted calculation, the sixth column has the total density. Default: false.");

        prm.declare_entry(
          "WRITE DENSITY OF STATES",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Computes density of states using Gaussian smearing. Uses specified 'DOS SMEAR TEMPERATURE' as broadening parameter. Outputs a file name 'dosData.out' containing two columns with first column indicating the energy in eV (without shift wrt Fermi energy. Fermi energy can be obtained from the file'fermiEnergy.out' that is generated when 'SAVE RHO DATA = true' in 'GS' calculation) and second column indicating the density of states. In case of collinear spin polarization, the second and third columns indicate the spin-up and spin-down density of states.");

        prm.declare_entry(
          "WRITE LOCAL DENSITY OF STATES",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Computes local density of states on each atom using Lorentzians. Uses specified Temperature for SCF as the broadening parameter. Outputs a file name 'ldosData.out' containing NUMATOM+1 columns with first column indicating the energy in eV and all other NUMATOM columns indicating local density of states for each of the NUMATOM atoms.");

        prm.declare_entry(
          "WRITE PROJECTED DENSITY OF STATES",
          "false",
          dealii::Patterns::Bool(),
          R"([Standard] Computes projected density of states on each atomic orbital using Gaussian smearing. Uses specified 'DOS SMEAR TEMPERATURE' as the broadening parameter. Outputs files with name format 'pdosData_atom#{atom number}_wfc#{wfc number}({wfc name}).out'. For colinear, spin-unpolarized case, each of these file contain columns with format 'E sumPDOS  PDOS_0 .... PDOS_(2l)', where E: the energy is eV (without shift wrt Fermi energy. Fermi energy can be obtained from the file'fermiEnergy.out' that is generated when 'SAVE RHO DATA = true'in 'GS' calculation),l: azimuthal quantum number, sumPDOS: PDOS_0 + .. +PDOS_(2l).
          For colinear, spin-polarized case, the columns has format 'E sumPDOS_up sumPDOS_down PDOS_0_up PDOS_0_down .... PDOS_(2l)_up PDOS_(2l)_down', where 'up' and 'down' refer to the spin up and spin down case with all other terms having the same meaning as of the spin-unpolarized case.)");

        prm.declare_entry(
          "DOS SMEAR TEMPERATURE",
          "500",
          dealii::Patterns::Double(),
          "[standard] Gaussian smearing temperature (in K) for DOS, PDOS and LDOS calculation");

        prm.declare_entry(
          "DELTA ENERGY",
          "0.01",
          dealii::Patterns::Double(),
          "[standard] Interval size of energy spectrum (in eV), for DOS, PDOS and LDOS calculation");

        prm.declare_entry(
          "READ ATOMIC WFC PDOS FROM PSP FILE",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Read atomic wavefunctons from the pseudopotential file for computing projected density of states. When set to false atomic wavefunctions from the internal database are read, which correspond to sg15 ONCV pseudopotentials.");

        prm.declare_entry(
          "WRITE LOCALIZATION LENGTHS",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Computes localization lengths of all wavefunctions which is defined as the deviation around the mean position of a given wavefunction. Outputs a file name 'localizationLengths.out' containing 2 columns with first column indicating the wavefunction index and second column indicating localization length of the corresponding wavefunction.");

        prm.enter_subsection("XY-Planar Average post-processing");
        {
          prm.declare_entry(
            "XY-PLANAR AVERAGE OF ELECTRON DENSITY",
            "false",
            dealii::Patterns::Bool(),
            "[standard] Computes the planar average density $\rho(z)$");

          prm.declare_entry(
            "XY-PLANAR AVERAGE OF ELECTROSTATIC POTENTIAL",
            "false",
            dealii::Patterns::Bool(),
            "[standard] Computes the planar average density $\phi(z)$");

          prm.declare_entry(
            "XY-PLANAR AVERAGE OF BARE POTENTIAL",
            "false",
            dealii::Patterns::Bool(),
            "[standard] Computes the planar average density $V_{bare} = \phi + V_{pseudo}-V_{self}+u_{ext}$");


          prm.declare_entry(
            "NUMBER OF SAMPLING POINTS ALONG X",
            "501",
            dealii::Patterns::Integer(1),
            "[standard] Number of sampling points alon the X axis");

          prm.declare_entry(
            "NUMBER OF SAMPLING POINTS ALONG Y",
            "501",
            dealii::Patterns::Integer(1),
            "[standard] Number of sampling points alon the Y axis");

          prm.declare_entry(
            "NUMBER OF SAMPLING POINTS ALONG Z",
            "501",
            dealii::Patterns::Integer(1),
            "[standard] Number of sampling points alon the Z axis");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("FunctionalTest");
      {
        prm.declare_entry(
          "TEST NAME",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Name of the Functional test that needs to be run.");
      }
      prm.leave_subsection();
      prm.enter_subsection("Parallelization");
      {
        prm.declare_entry(
          "NPKPT",
          "1",
          dealii::Patterns::Integer(1),
          "[Standard] Number of groups of MPI tasks across which the work load of the irreducible k-points is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPKPT must be less than or equal to the number of irreducible k-points.");

        prm.declare_entry(
          "NPBAND",
          "1",
          dealii::Patterns::Integer(1),
          "[Standard] Number of groups of MPI tasks across which the work load of the bands is parallelised. NPKPT times NPBAND must be a divisor of total number of MPI tasks. Further, NPBAND must be less than or equal to NUMBER OF KOHN-SHAM WAVEFUNCTIONS.");

        prm.declare_entry(
          "MPI ALLREDUCE BLOCK SIZE",
          "100.0",
          dealii::Patterns::Double(0),
          R"([Advanced] Block message size in MB used to break a single MPI\_Allreduce call on wavefunction vectors data into multiple MPI\_Allreduce calls. This is useful on certain architectures which take advantage of High Bandwidth Memory to improve efficiency of MPI operations. This variable is relevant only if NPBAND>1. Default value is 100.0 MB.)");

        prm.declare_entry(
          "BAND PARAL OPT",
          "true",
          dealii::Patterns::Bool(),
          "[Standard] Uses a more optimal route for band parallelization but at the cost of extra wavefunctions memory.");
      }
      prm.leave_subsection();

      prm.enter_subsection("SCF Checkpointing and Restart");
      {
        prm.declare_entry(
          "SAVE QUAD DATA",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Saves the various variables involved in the SCF fixed point interation to restart file. Default value is false.");

        prm.declare_entry(
          "LOAD QUAD DATA",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Loads the various variables involved in the SCF fixed point iteration from restart file. Used for NSCF calculations where the quadrature density is required. Default value is false.");


        prm.declare_entry(
          "RESTART SP FROM NO SP",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Enables ground-state solve for SPIN POLARIZED case reading the SPIN UNPOLARIZED density from the checkpoint files, and use the TOTAL MAGNETIZATION to compute the spin up and spin down densities. This option is used in conjuction with LOAD QUAD DATA or LOAD RHO DATA. Default false.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Geometry");
      {
        prm.declare_entry(
          "ATOMIC COORDINATES FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Atomic-coordinates input file name. For a fully non-periodic domain, give Cartesian coordinates of the atoms (in a.u) with respect to the origin at the center of the domain. For periodic and semi-periodic domains, give fractional coordinates of atoms. File format (example for two atoms for a spin unpolarized calculation): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 c1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 c2 (row2), where c1 and c2 are optional parameters representing partial charges to be used to set the initial guess of charge density. Parameter c scales the valence charge as (scaledAtomicValenceCharge= c * atomicValenceCharge). File format (example for two atoms for a spin-polarized calculation): Atom1-atomic-charge Atom1-valence-charge x1 y1 z1 m1 c1 (row1), Atom2-atomic-charge Atom2-valence-charge x2 y2 z2 m2 c2 (row2), where m1 and m2 are the initial guess of magnetization (ranging from -1.0 to 1.0, with -1.0 representing all electrons of the atom having spin -0.5 and 1.0 representing all electrons of the atom having spin 0.5) to be used to set the initial guess of magnetization density. In case both m and c are used, please use the scaled valence charge to set the value appropriately (m*scaledAtomicValenceCharge=atomicMagnetization). The number of rows must be equal to NATOMS, and the number of unique atoms must be equal to NATOM TYPES.");
        prm.declare_entry(
          "ATOMIC DISP COORDINATES FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Atomic displacement coordinates input file name. The FEM mesh is deformed using Gaussian functions attached to the atoms. File format (example for two atoms): delx1 dely1 delz1 (row1), delx2 dely2 delz2 (row2). The number of rows must be equal to NATOMS. Units in a.u.");

        prm.declare_entry(
          "NATOMS",
          "0",
          dealii::Patterns::Integer(0),
          "[Standard] Total number of atoms. This parameter requires a mandatory non-zero input which is equal to the number of rows in the file passed to ATOMIC COORDINATES FILE.");

        prm.declare_entry(
          "NATOM TYPES",
          "0",
          dealii::Patterns::Integer(0),
          "[Standard] Total number of atom types. This parameter requires a mandatory non-zero input which is equal to the number of unique atom types in the file passed to ATOMIC COORDINATES FILE.");

        prm.declare_entry(
          "DOMAIN VECTORS FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Domain vectors input file name. Domain vectors are the vectors bounding the three edges of the 3D parallelepiped computational domain. File format: v1x v1y v1z (row1), v2y v2y v2z (row2), v3z v3y v3z (row3). Units: a.u. CAUTION: please ensure that the domain vectors form a right-handed coordinate system i.e. dotProduct(crossProduct(v1,v2),v3)>0. Domain vectors are the typical lattice vectors in a fully periodic calculation.");
        prm.enter_subsection("Optimization");
        {
          prm.declare_entry(
            "OPTIMIZATION MODE",
            "ION",
            dealii::Patterns::Selection("ION|CELL|IONCELL"),
            "[Standard] Specifies whether the ionic coordinates and/or the lattice vectors are relaxed.");

          prm.declare_entry(
            "ION FORCE",
            "false",
            dealii::Patterns::Bool(),
            "[Standard] Boolean parameter specifying if atomic forces are to be computed. Automatically set to true if ION OPT is true.");

          prm.declare_entry(
            "NON SELF CONSISTENT FORCE",
            "false",
            dealii::Patterns::Bool(),
            "[Developer] Boolean parameter specifying whether to include the force contributions arising out of non self-consistency in the Kohn-Sham ground-state calculation. Currently non self-consistent force computation is still in experimental phase. The default option is false.");

          prm.declare_entry(
            "ION OPT SOLVER",
            "LBFGS",
            dealii::Patterns::Selection("BFGS|LBFGS|CGPRP"),
            "[Standard] Method for Ion relaxation solver. LBFGS is the default");

          prm.declare_entry(
            "CELL OPT SOLVER",
            "LBFGS",
            dealii::Patterns::Selection("BFGS|LBFGS|CGPRP"),
            "[Standard] Method for Cell relaxation solver. LBFGS is the default");

          prm.declare_entry(
            "MAXIMUM OPTIMIZATION STEPS",
            "300",
            dealii::Patterns::Integer(1, 1000),
            "[Standard] Sets the maximum number of optimization steps to be performed.");

          prm.declare_entry(
            "MAXIMUM STAGGERED CYCLES",
            "300",
            dealii::Patterns::Integer(1, 1000),
            "[Standard] Sets the maximum number of staggered ion/cell optimization cycles to be performed.");

          prm.declare_entry(
            "MAXIMUM ION UPDATE STEP",
            "0.5",
            dealii::Patterns::Double(0, 5.0),
            "[Standard] Sets the maximum allowed step size (displacement in a.u.) during ion relaxation.");

          prm.declare_entry(
            "MAXIMUM CELL UPDATE STEP",
            "0.1",
            dealii::Patterns::Double(0, 5.0),
            "[Standard] Sets the maximum allowed step size (deformation) during cell relaxation.");

          prm.declare_entry(
            "MAX LINE SEARCH ITER",
            "5",
            dealii::Patterns::Integer(1, 100),
            "[Standard] Sets the maximum number of line search iterations in the case of CGPRP. Default is 5.");

          prm.declare_entry(
            "FORCE TOL",
            "1e-4",
            dealii::Patterns::Double(0, 1.0),
            "[Standard] Sets the tolerance on the maximum force (in a.u.) on an atom during atomic relaxation, when the atoms are considered to be relaxed.");

          prm.declare_entry(
            "ION RELAX FLAGS FILE",
            "",
            dealii::Patterns::Anything(),
            "[Standard] File specifying the permission flags (1-free to move, 0-fixed) and external forces for the 3-coordinate directions and for all atoms. File format (example for two atoms with atom 1 fixed and atom 2 free and 0.01 Ha/Bohr force acting on atom 2): 0 0 0 0.0 0.0 0.0(row1), 1 1 1 0.0 0.0 0.01(row2). External forces are optional.");

          prm.declare_entry(
            "CELL STRESS",
            "false",
            dealii::Patterns::Bool(),
            "[Standard] Boolean parameter specifying if cell stress needs to be computed. Automatically set to true if CELL OPT is true.");

          prm.declare_entry(
            "STRESS TOL",
            "1e-6",
            dealii::Patterns::Double(0, 1.0),
            "[Standard] Sets the tolerance of the cell stress (in a.u.) during cell-relaxation.");

          prm.declare_entry(
            "CELL CONSTRAINT TYPE",
            "12",
            dealii::Patterns::Integer(1, 13),
            "[Standard] Cell relaxation constraint type, 1 (isotropic shape-fixed volume optimization), 2 (volume-fixed shape optimization), 3 (relax along domain vector component v1x), 4 (relax along domain vector component v2y), 5 (relax along domain vector component v3z), 6 (relax along domain vector components v2y and v3z), 7 (relax along domain vector components v1x and v3z), 8 (relax along domain vector components v1x and v2y), 9 (relax along domain vector components v1x, v2y and v3z), 10 (2D - relax along x and y components), 11(2D- relax only x and y components with inplane area fixed), 12(relax all domain vector components), 13 automatically decides the constraints based on boundary conditions. CAUTION: A majority of these options only make sense in an orthorhombic cell geometry.");

          prm.declare_entry(
            "REUSE WFC",
            "true",
            dealii::Patterns::Bool(),
            "[Standard] Reuse previous ground-state wavefunctions during geometry optimization. Default setting is true.");

          prm.declare_entry(
            "REUSE DENSITY",
            "1",
            dealii::Patterns::Integer(0, 2),
            "[Standard] Parameter controlling the reuse of ground-state density during geometry optimization. The options are 0 (reinitialize density based on superposition of atomic densities), 1 (reuse ground-state density of previous relaxation step), and 2 (subtract superposition of atomic densities from the previous step's ground-state density and add superposition of atomic densities from the new atomic positions. Option 2 is not enabled for spin-polarized case. Default setting is 0.");

          prm.declare_entry(
            "BFGS STEP METHOD",
            "QN",
            dealii::Patterns::Selection("QN|RFO"),
            "[Standard] Method for computing update step in BFGS. Quasi-Newton step (default) or Rational Function Step as described in JPC 1985, 89:52-57.");

          prm.declare_entry(
            "USE PRECONDITIONER",
            "false",
            dealii::Patterns::Bool(),
            "[Standard] Boolean parameter specifying if the preconditioner described by JCP 144, 164109 (2016) is to be used.");

          prm.declare_entry(
            "LBFGS HISTORY",
            "5",
            dealii::Patterns::Integer(1, 20),
            "[Standard] Number of previous steps to considered for the LBFGS update.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Boundary conditions");
      {
        prm.declare_entry(
          "SELF POTENTIAL RADIUS",
          "0.0",
          dealii::Patterns::Double(0.0, 50),
          "[Advanced] The radius (in a.u) of the ball around an atom in which self-potential of the associated nuclear charge is solved. For the default value of 0.0, the radius value is automatically determined to accommodate the largest radius possible for the given finite element mesh. The default approach works for most problems.");

        prm.declare_entry(
          "PERIODIC1",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Periodicity along the first domain bounding vector.");

        prm.declare_entry(
          "PERIODIC2",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Periodicity along the second domain bounding vector.");

        prm.declare_entry(
          "PERIODIC3",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Periodicity along the third domain bounding vector.");

        prm.declare_entry(
          "POINT WISE DIRICHLET CONSTRAINT",
          "false",
          dealii::Patterns::Bool(),
          "[Developer] Flag to set point wise dirichlet constraints to eliminate null-space associated with the discretized Poisson operator subject to periodic BCs.");

        prm.declare_entry(
          "MULTIPOLE BOUNDARY CONDITIONS",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Flag to set point wise multipole boundary conditions (upto quadrupole term) for non-periodic systems.");

        prm.enter_subsection("External Electric Potential");
        {
          prm.declare_entry(
            "INCLUDE EXTERNAL ELECTROSTATIC POTENTIAL BIAS",
            "false",
            dealii::Patterns::Bool(),
            "[Standard] Enables the user to apply an external electrostatic potential bias to the system. Ensure PERIODIC3 = false. The default value is false.");

          prm.declare_entry(
            "TYPE OF EXTERNAL ELECTOSTATIC POTENTIAL BIAS",
            "CEF",
            dealii::Patterns::Selection("CPD-APD|CEF|CPD-PPD"),
            "[Standard] Select the type of external potential bias considered. CPD ensures a total electrostaticpotential difference between the locations provieded. CEF ensures a constant electric field along the non-periodic axis. The default is CEF.");

          prm.declare_entry(
            "SLOPE OF EXTERNAL POTENTIAL",
            "0.0",
            dealii::Patterns::Double(-50.0, 50),
            "[Standard] Slope (in Ha/bohr) of the external potential applied. Used for CEF type of external potential bias. The CEF setup can bse used to simulate slab models with only Neuman boundary conditions along non-periodic axis when value = 0.0. The default value is 0.0. ");

          prm.declare_entry(
            "CONSTRAINED POTENTIAL DIFFERENCE VALUE",
            "0.0",
            dealii::Patterns::Double(-50.0, 50),
            "[Standard] The value of the potential difference applied when APPLY_POTENTIAL_DIFFERENCE is used.");
          prm.declare_entry(
            "CONSTRAINED POTENTIAL DIFFERENCE LEFT SURFACE VALUE",
            "0.0",
            dealii::Patterns::Double(0.0, 50),
            "[Standard] The value of the potential applied of the left surface when APPLY_POTENTIAL_DIFFERENCE is used.");

          prm.declare_entry(
            "CONSTRAINED POTENTIAL LEFT SURFACE POSITION",
            "0.0",
            dealii::Patterns::Double(0.0, 1.0),
            "[Standard] Fractional coordinate at which the left potential value is set.");

          prm.declare_entry(
            "CONSTRAINED POTENTIAL RIGHT SURFACE POSITION",
            "1.0",
            dealii::Patterns::Double(0.0, 1.0),
            "[Standard] Fractional coordinate at which the right potential value is set.");

          prm.declare_entry(
            "CONSTRAINED POTENTIAL LOCALIZED WIDTH",
            "0.0",
            dealii::Patterns::Double(0.0, 500),
            "[Standard] Width in bohr of the constrained potential surface on the right");

          prm.declare_entry(
            "POSITION OF SAW TOOTH POTENTIAL MAX",
            "0.9",
            dealii::Patterns::Double(0.0, 1.0),
            "[Standard] Fractional coordinate at which the saw tooth potential attains maximum value.");
          prm.declare_entry(
            "REGION OF SAW TOOTH POTENTIAL DECREASE",
            "0.2",
            dealii::Patterns::Double(0.0, 1.0),
            "[Standard] Zone in the unit cell where the saw-like potential decreases. Similar to eopreg used in QE");
        }
        prm.leave_subsection();


        prm.declare_entry(
          "CONSTRAINTS PARALLEL CHECK",
          "false",
          dealii::Patterns::Bool(),
          "[Developer] Check for consistency of constraints in parallel.");

        prm.declare_entry(
          "CONSTRAINTS FROM SERIAL DOFHANDLER",
          "false",
          dealii::Patterns::Bool(),
          "[Developer] Create constraints from serial dofHandler.");

        prm.declare_entry(
          "SMEARED NUCLEAR CHARGES",
          "true",
          dealii::Patterns::Bool(),
          "[Developer] Nuclear charges are smeared for solving electrostatic fields. Default is true for pseudopotential calculations and false for all-electron calculations.");

        prm.declare_entry(
          "FLOATING NUCLEAR CHARGES",
          "true",
          dealii::Patterns::Bool(),
          "[Developer] Nuclear charges are allowed to float independent of the FEM mesh nodal positions. Only allowed for pseudopotential calculations. Internally set to false for all-electron calculations.");
      }
      prm.leave_subsection();


      prm.enter_subsection("Finite element mesh parameters");
      {
        prm.declare_entry(
          "POLYNOMIAL ORDER",
          "6",
          dealii::Patterns::Integer(1, 12),
          "[Standard] The degree of the finite-element interpolating polynomial in the Kohn-Sham Hamitonian except the electrostatics. Default value is 6 which is good choice for most pseudopotential calculations. POLYNOMIAL ORDER= 4 or 5 is usually a good choice for all-electron problems.");

        prm.declare_entry(
          "POLYNOMIAL ORDER ELECTROSTATICS",
          "0",
          dealii::Patterns::Integer(0, 24),
          "[Standard] The degree of the finite-element interpolating polynomial for the electrostatics part of the Kohn-Sham Hamiltonian. It is automatically set to POLYNOMIAL ORDER if POLYNOMIAL ORDER ELECTROSTATICS set to default value of zero.");
        prm.declare_entry(
          "POLYNOMIAL ORDER DENSITY NODAL",
          "0",
          dealii::Patterns::Integer(0, 24),
          "[Standard] The degree of the finite-element interpolating polynomial for interpolating electron density. It is automatically set to max of POLYNOMIAL ORDER+2 and POLYNOMIAL ORDER ELECTROSTATICS  if POLYNOMIAL ORDER DENSITY NODAL set to default value of zero.");
        prm.declare_entry(
          "DENSITY QUADRATURE RULE",
          "0",
          dealii::Patterns::Integer(0, 24),
          "[Standard] The quadrtaure rule used for computing the electron density. It is automatically set to 16 for MGGA exchange-correlation functional or max of POLYNOMIAL ORDER DENSITY NODAL +1 for other exchange-correlation functionals if DENSITY QUADTRATURE RULE set to default value of zero.");
        prm.enter_subsection("Auto mesh generation parameters");
        {
          prm.declare_entry(
            "BASE MESH SIZE",
            "0.0",
            dealii::Patterns::Double(0, 20),
            "[Advanced] Mesh size of the base mesh on which refinement is performed. For the default value of 0.0, a heuristically determined base mesh size is used, which is good enough for most cases. Standard users do not need to tune this parameter. Units: a.u.");

          prm.declare_entry(
            "ATOM BALL RADIUS",
            "0.0",
            dealii::Patterns::Double(0, 20),
            "[Standard] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AROUND ATOM and coarse-grained in the region outside the enclosing balls. For the default value of 0.0, a heuristically determined value is used, which is good enough for most cases but can be a bit conservative choice for fully non-periodic and semi-periodic problems as well as all-electron problems. To improve the computational efficiency user may experiment with values of ATOM BALL RADIUS ranging between 3.0 to 6.0 for pseudopotential problems, and ranging between 1.0 to 2.5 for all-electron problems.  Units: a.u.");

          prm.declare_entry(
            "INNER ATOM BALL RADIUS",
            "0.0",
            dealii::Patterns::Double(0, 20),
            "[Advanced] Radius of ball enclosing every atom, inside which the mesh size is set close to MESH SIZE AT ATOM. Standard users do not need to tune this parameter. Units: a.u.");


          prm.declare_entry(
            "MESH SIZE AROUND ATOM",
            "1.0",
            dealii::Patterns::Double(0.0001, 10),
            "[Standard] Mesh size in a ball of radius ATOM BALL RADIUS around every atom. For pseudopotential calculations, the value ranges between 0.8 to 2.5 depending on the cutoff energy for the pseudopotential. For all-electron calculations, a value of around 0.5 would be a good starting choice. In most cases, MESH SIZE AROUND ATOM is the only parameter to be tuned to achieve the desired accuracy in energy and forces with respect to the mesh refinement. Units: a.u.");

          prm.declare_entry(
            "MESH SIZE AT ATOM",
            "0.0",
            dealii::Patterns::Double(0.0, 10),
            "[Advanced] Mesh size of the finite elements in the immediate vicinity of the atom. For the default value of 0.0, a heuristically determined MESH SIZE AT ATOM is used for all-electron calculations. For pseudopotential calculations, the default value of 0.0, sets the MESH SIZE AT ATOM to be the same value as MESH SIZE AROUND ATOM. Standard users do not need to tune this parameter. Units: a.u.");

          prm.declare_entry(
            "MESH ADAPTION",
            "false",
            dealii::Patterns::Bool(),
            "[Developer] Generates adaptive mesh based on a-posteriori mesh adaption strategy using single atom wavefunctions before computing the ground-state. Default: false.");

          prm.declare_entry(
            "AUTO ADAPT BASE MESH SIZE",
            "true",
            dealii::Patterns::Bool(),
            "[Developer] Automatically adapt the BASE MESH SIZE such that subdivisions of that during refinement leads closest to the desired MESH SIZE AROUND ATOM. Default: true.");


          prm.declare_entry(
            "TOP FRAC",
            "0.1",
            dealii::Patterns::Double(0.0, 1),
            "[Developer] Top fraction of elements to be refined.");

          prm.declare_entry("NUM LEVELS",
                            "10",
                            dealii::Patterns::Integer(0, 30),
                            "[Developer] Number of times to be refined.");

          prm.declare_entry(
            "TOLERANCE FOR MESH ADAPTION",
            "1",
            dealii::Patterns::Double(0.0, 1),
            "[Developer] Tolerance criteria used for stopping the multi-level mesh adaption done apriori using single atom wavefunctions. This is used as Kinetic energy change between two successive iterations");

          prm.declare_entry(
            "ERROR ESTIMATE WAVEFUNCTIONS",
            "5",
            dealii::Patterns::Integer(0),
            "[Developer] Number of wavefunctions to be used for error estimation.");

          prm.declare_entry(
            "GAUSSIAN CONSTANT FORCE GENERATOR",
            "0.75",
            dealii::Patterns::Double(0.0),
            "[Developer] Force computation generator gaussian constant. Also used for mesh movement. Gamma(r)= exp(-(r/gaussianConstant);(gaussianOrder)).");

          prm.declare_entry(
            "GAUSSIAN ORDER FORCE GENERATOR",
            "4.0",
            dealii::Patterns::Double(0.0),
            "[Developer] Force computation generator gaussian order. Also used for mesh movement. Gamma(r)= exp(-(r/gaussianConstant);(gaussianOrder)).");

          prm.declare_entry(
            "GAUSSIAN ORDER MOVE MESH TO ATOMS",
            "4.0",
            dealii::Patterns::Double(0.0),
            "[Developer] Move mesh to atoms gaussian order. Gamma(r)= exp(-(r/gaussianConstant);(gaussianOrder)).");

          prm.declare_entry(
            "USE FLAT TOP GENERATOR",
            "false",
            dealii::Patterns::Bool(),
            "[Developer] Use a composite generator flat top and Gaussian generator for mesh movement and configurational force computation.");

          prm.declare_entry("MESH SIZES FILE",
                            "",
                            dealii::Patterns::Anything(),
                            "[Developer] Use mesh sizes from this file.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Brillouin zone k point sampling options");
      {
        prm.enter_subsection("Monkhorst-Pack (MP) grid generation");
        {
          prm.declare_entry(
            "SAMPLING POINTS 1",
            "1",
            dealii::Patterns::Integer(1, 1000),
            "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 1.");

          prm.declare_entry(
            "SAMPLING POINTS 2",
            "1",
            dealii::Patterns::Integer(1, 1000),
            "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 2.");

          prm.declare_entry(
            "SAMPLING POINTS 3",
            "1",
            dealii::Patterns::Integer(1, 1000),
            "[Standard] Number of Monkhorst-Pack grid points to be used along reciprocal lattice vector 3.");

          prm.declare_entry(
            "SAMPLING SHIFT 1",
            "0",
            dealii::Patterns::Integer(0, 1),
            "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 1.");

          prm.declare_entry(
            "SAMPLING SHIFT 2",
            "0",
            dealii::Patterns::Integer(0, 1),
            "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 2.");

          prm.declare_entry(
            "SAMPLING SHIFT 3",
            "0",
            dealii::Patterns::Integer(0, 1),
            "[Standard] If fractional shifting to be used (0 for no shift, 1 for shift) along reciprocal lattice vector 3.");
        }
        prm.leave_subsection();

        prm.declare_entry(
          "kPOINT RULE FILE",
          "",
          dealii::Patterns::Anything(),
          "[Developer] File providing list of k points on which eigen values are to be computed from converged KS Hamiltonian. The first three columns specify the crystal coordinates of the k points. The fourth column provides weights of the corresponding points, which is currently not used. The eigen values are written on an output file bands.out");

        prm.declare_entry(
          "USE GROUP SYMMETRY",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Flag to control the use of point group symmetries. Currently this feature cannot be used if ION FORCE or CELL STRESS input parameters are set to true.");

        prm.declare_entry(
          "USE TIME REVERSAL SYMMETRY",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Flag to control the use of time reversal symmetry.");
      }
      prm.leave_subsection();

      prm.enter_subsection("DFT functional parameters");
      {
        prm.enter_subsection("CONFINING POTENTIAL parameters");
        {
          prm.declare_entry(
            "APPLY CONFINING POTENTIAL",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Apply confining potential. Usually required for anionic charges."
            "The confining potential is applied between maxDist + r1 and maxDist + r2. Where maxDist "
            "is the maximum distance of atoms from the center. r1 and r2 is the INNER and OUTER radii respectively."
            "Between 0 and maxdist + r1: V(r) = 0;"
            "Between maxdist + r1 and maxdist + r2: V(r) = (C*exp(-W/(dist1)))/(dist2*dist2 + 1E-6);"
            "Beyond maxdist + r2: V(r) = (C*exp(-W/(dist1)))/(1E-6);"
            "where dist1 = r - (maxdist + r1) and dist2 = (maxdist + r2) - r");

          prm.declare_entry(
            "INNER RADIUS",
            "17.0",
            dealii::Patterns::Double(0, 100),
            "[Advanced] The inner radius (r1) for the confining potential.");

          prm.declare_entry(
            "OUTER RADIUS",
            "20.0",
            dealii::Patterns::Double(0, 100),
            "[Advanced] The outer radius (r2) for the confining potential.");

          prm.declare_entry(
            "W PARAM",
            "1.0",
            dealii::Patterns::Double(0, 100),
            "[Advanced] The W parameter for the confining potential.");

          prm.declare_entry(
            "C PARAM",
            "1.0",
            dealii::Patterns::Double(0, 100),
            "[Advanced] The C parameter for the confining potential.");
        }
        prm.leave_subsection();
        prm.declare_entry(
          "PSEUDOPOTENTIAL CALCULATION",
          "true",
          dealii::Patterns::Bool(),
          "[Standard] Boolean Parameter specifying whether pseudopotential DFT calculation needs to be performed. For all-electron DFT calculation set to false.");

        prm.declare_entry(
          "PSEUDO TESTS FLAG",
          "false",
          dealii::Patterns::Bool(),
          "[Developer] Boolean parameter specifying the explicit path of pseudopotential upf format files used for ctests");

        prm.declare_entry(
          "PSEUDOPOTENTIAL FILE NAMES LIST",
          "",
          dealii::Patterns::Anything(),
          R"([Standard] Pseudopotential file. This file contains the list of pseudopotential file names in UPF format corresponding to the atoms involved in the calculations. UPF version 2.0 or greater and norm-conserving pseudopotentials(ONCV and Troullier Martins) in UPF format are only accepted. File format (example for two atoms Mg(z=12), Al(z=13)): 12 filename1.upf(row1), 13 filename2.upf (row2). Important Note: ONCV pseudopotentials data base in UPF format can be downloaded from http://www.quantum-simulation.org/potentials/sg15\_oncv or http://www.pseudo-dojo.org/.  Troullier-Martins pseudopotentials in UPF format can be downloaded from http://www.quantum-espresso.org/pseudopotentials/fhi-pp-from-abinit-web-site.)");

        prm.declare_entry(
          "EXCHANGE CORRELATION TYPE", "GGA-PBE", dealii::Patterns::Selection("LDA-PZ|LDA-PW|LDA-VWN|GGA-PBE|GGA-RPBE|GGA-LBxPBEc|MLXC-NNLDA|MLXC-NNGGA|MLXC-NNLLMGGA|LDA-PZ+U|LDA-PW+U|LDA-VWN+U|GGA-PBE+U|GGA-RPBE+U|GGA-LBxPBEc+U|MLXC-NNLDA+U|MLXC-NNGGA+U|MLXC-NNLLMGGA+U|MGGA-SCAN|MGGA-R2SCAN"), R"([Standard] Parameter specifying the type of exchange-correlation to be used: LDA-PZ (Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)]), LDA-PW (Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)]), LDA-VWN (Vosko, Wilk \& Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)]), GGA-PBE (Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)]), GGA-RPBE (RPBE: B. Hammer, L. B. Hansen, and J. K. Nørskov, Phys. Rev. B 59, 7413 (1999)), GGA-LBxPBEc van Leeuwen \& Baerends exchange [Phys. Rev. A 49, 2421 (1994)] with  PBE correlation [Phys. Rev. Lett. 77, 3865 (1996)], MLXC-NNLDA (LDA-PW + NN-LDA), MLXC-NNGGA (GGA-PBE + NN-GGA), MLXC-NNLLMGGA (GGA-PBE + NN Laplacian level MGGA), MGGA-SCAN (Strongly Constrained and Appropriately Normed functional [Phys. Rev. Lett. 115, 03640 (2015)]), MGGA-R2SCAN (regularized-restored SCAN [J. Phys. Chem. Lett. 19, 8208-8215 (2020)]). Caution: MLXC options are experimental. Add +U to use hubbard correction)");

        prm.declare_entry(
          "MODEL XC INPUT FILE",
          "",
          dealii::Patterns::Anything(),
          "[Developer] File that contains both the pytorch MLXC NN model (.ptc file) and the tolerances. This is an experimental feature to test out any new XC functional developed using machine learning.");

        prm.declare_entry(
          "AUX BASIS TYPE",
          "FE",
          dealii::Patterns::Selection("FE|SLATER|PW"),
          "[Developer] Auxiliary basis for projecting the Kohn-Sham density or density matrix for XC evaluation. FE is the default option.");

        prm.declare_entry(
          "AUX BASIS DATA",
          "",
          dealii::Patterns::Anything(),
          "[Developer] File that contains additional information for the Auxiliary basis selected in AUX BASIS TYPE.");

        prm.declare_entry(
          "NET CHARGE",
          "0.0",
          dealii::Patterns::Double(),
          "[Standard] Net charge of the system in atomic units, positive quantity implies addition of electrons. In case of non-periodic boundary conditions, this capability is implemented using multipole Dirichlet inhomogeneous boundary conditions for the electrostatics. In case of periodic and semi-periodic conditions a uniform background charge is used to create a neutral system.");

        prm.declare_entry(
          "SPIN POLARIZATION",
          "0",
          dealii::Patterns::Integer(0, 1),
          "[Standard] Spin polarization: 0 for no spin polarization and 1 for collinear spin polarization calculation. Default option is 0.");



        prm.declare_entry(
          "TOTAL MAGNETIZATION",
          "0.0",
          dealii::Patterns::Double(-1.0, 1.0),
          "[Standard] Total magnetization to be used for constrained spin-polarized DFT calculations (must be between -1.0 and +1.0). Corresponding magnetization per simulation domain will be (TOTAL MAGNETIZATION x (Number of electrons+Net charge)) a.u. ");

        prm.declare_entry(
          "USE ATOMIC MAGNETIZATION GUESS FOR CONSTRAINT MAG",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Use atomic magnetization initial guess from coordinates.inp when using constrained magnetization solve for a set TOTAL MAGNETIZATION. The default value of false sets the initial guess of spin density to be a scaling of the total atomic density.");

        prm.declare_entry(
          "PSP CUTOFF IMAGE CHARGES",
          "15.0",
          dealii::Patterns::Double(),
          "[Standard] Distance from the domain till which periodic images will be considered for the local part of the pseudopotential. Units in a.u. ");
        prm.enter_subsection("Dispersion Correction");
        {
          prm.declare_entry(
            "DISPERSION CORRECTION TYPE",
            "0",
            dealii::Patterns::Integer(0, 2),
            "[Standard] The dispersion correction type to be included post scf convergence: 0 for none, 1 for DFT-D3[JCP 132, 154104 (2010)][JCC 32, 1456 (2011)], 2 for DFT-D4 [JCP 147, 034112 (2017)][JCP 150, 154122 (2019)][PCCP 22, 8499-8512 (2020)].");
          prm.declare_entry(
            "D3 DAMPING TYPE",
            "3",
            dealii::Patterns::Integer(0, 4),
            "[Standard] The damping used for DFTD3, 0 for zero damping, 1 for BJ damping, 2 for D3M variant, 3 for BJM variant (default) and 4 for the OP variant.");
          prm.declare_entry(
            "D3 ATM",
            "false",
            dealii::Patterns::Bool(),
            "[Standard] Boolean parameter specifying whether or not the triple dipole correction in DFTD3 is to be included (ignored if DAMPING PARAMETERS FILE is specified).");
          prm.declare_entry(
            "D4 MBD",
            "false",
            dealii::Patterns::Bool(),
            "[Standard] Boolean parameter specifying whether or not the MBD correction in DFTD4 is to be included (ignored if DAMPING PARAMETERS FILE is specified).");
          prm.declare_entry(
            "DAMPING PARAMETERS FILE",
            "",
            dealii::Patterns::Anything(),
            "[Advanced] Name of the file containing custom damping parameters, for ZERO damping 6 parameters are expected (s6, s8, s9, sr6, sr8, alpha), for BJ anf BJM damping 6 parameters are expected (s6, s8, s9, a1, a2, alpha), for ZEROM damping 7 parameters are expected (s6, s8, s9, sr6, sr8, alpha, beta) and for optimized power damping 7 parameters are expected (s6, s8, s9, a1, a2, alpha, beta).");
          prm.declare_entry(
            "TWO BODY CUTOFF",
            "94.8683298050514",
            dealii::Patterns::Double(0.0),
            "[Advanced] Cutoff in a.u. for computing 2 body interactions terms in D3 correction");
          prm.declare_entry(
            "THREE BODY CUTOFF",
            "40.0",
            dealii::Patterns::Double(0.0),
            "[Advanced] Cutoff in a.u. for computing 3 body interactions terms in D3 correction");
          prm.declare_entry(
            "CN CUTOFF",
            "40.0",
            dealii::Patterns::Double(0.0),
            "[Advanced] Cutoff in a.u. for computing coordination number in D3 correction");
        }
        prm.leave_subsection();

        prm.enter_subsection("Hubbard Parameters");
        {
          prm.declare_entry(
            "HUBBARD PARAMETERS FILE",
            "",
            dealii::Patterns::Anything(),
            "[Standard] Name of the file containing hubbard parameters. "
            "This file describes the orbitals and the hubbard U parameter for each hubbard species."
            " A sample file for Pt-Au dimer is as follows:  "
            "3 (row1 - number of hubbard species, The ID 0 is reserved for atoms with no hubbard correction ),"
            "0 0 (row2 - Hubbard species Id and the corresponding number of orbitals"
            "1 78 0.110248 1 9.0 (row3 - hubbard species Id corresponding to Pt, Atomic number, Hubbard U parameter in Ha, Number of orbitals on which the hubbard correction is applied (5D in this case), The initial occupancy of the orbitals)"
            "5 2 (row4 - the Quantum number n and Quantum number l of the orbital)"
            "2 79 0.1469976 1 10.0 (row3 - hubbard species Id corresponding to Au, Atomic number, Hubbard U parameter in Ha, Number of orbitals on which the hubbard correction is applied (5D in this case), The initial occupancy of the orbitals)"
            "5 2 (row4 - the Quantum number n and Quantum number l of the orbital)"
            "78 1 (row5 - the atomic number and the corresponding hubbard species Id. The list has to be copied from the coordinates file"
            "79 2 (row6 - the atomic number and the corresponding hubbard species Id. The list has to be copied from the coordinates file");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();


      prm.enter_subsection("SCF parameters");
      {
        prm.declare_entry(
          "TEMPERATURE",
          "500.0",
          dealii::Patterns::Double(1e-5),
          "[Standard] Fermi-Dirac smearing temperature (in Kelvin).");

        prm.declare_entry(
          "MAXIMUM ITERATIONS",
          "200",
          dealii::Patterns::Integer(1, 1000),
          "[Standard] Maximum number of iterations to be allowed for SCF convergence");

        prm.declare_entry(
          "TOLERANCE",
          "1e-05",
          dealii::Patterns::Double(1e-12, 1.0),
          "[Standard] SCF iterations stopping tolerance in terms of $L_2$ norm of the electron-density difference between two successive iterations. The default tolerance of is set to a tight value of 1e-5 for accurate ionic forces and cell stresses keeping structural optimization and molecular dynamics in mind. A tolerance of 1e-4 would be accurate enough for calculations without structural optimization and dynamics. CAUTION: A tolerance close to 1e-7 or lower can deteriorate the SCF convergence due to the round-off error accumulation.");

        prm.declare_entry(
          "ENERGY TOLERANCE",
          "1e-07",
          dealii::Patterns::Double(1e-12, 1.0),
          "[Standard] SCF iterations stopping tolerance in terms of difference between Harris-Foulkes and Kohn-Sham energies. The default tolerance of is set to a tight value of 1e-7 for accurate ionic forces and cell stresses keeping structural optimization and molecular dynamics in mind. A tolerance of 1e-6 would be accurate enough for calculations without structural optimization and dynamics.");

        prm.declare_entry(
          "MIXING HISTORY",
          "10",
          dealii::Patterns::Integer(1, 1000),
          "[Standard] Number of SCF iteration history to be considered for density mixing schemes. For metallic systems, a mixing history larger than the default value provides better scf convergence.");

        prm.declare_entry(
          "MIXING PARAMETER",
          "0.0",
          dealii::Patterns::Double(-1e-12, 1.0),
          "[Standard] Mixing parameter to be used in density mixing schemes. For default value of 0.0, it is heuristically set for different mixing schemes (0.2 for Anderson, and 0.5 for Kerker and LRD.");

        prm.declare_entry(
          "SPIN MIXING ENHANCEMENT FACTOR",
          "1.0",
          dealii::Patterns::Double(-1e-12, 100.0),
          "[Standard] Scales the mixing parameter for the spin densities as SPIN MIXING ENHANCEMENT FACTOR times MIXING PARAMETER. This parameter is not used for LOW\_RANK\_DIELECM\_PRECOND mixing method.");

        prm.declare_entry(
          "ADAPT ANDERSON MIXING PARAMETER",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Boolean parameter specifying whether to adapt the Anderson mixing parameter based on algorithm 1 in [CPC. 292, 108865 (2023)].");

        prm.declare_entry(
          "KERKER MIXING PARAMETER",
          "0.05",
          dealii::Patterns::Double(0.0, 1000.0),
          R"([Standard] Mixing parameter to be used in Kerker mixing scheme which usually represents Thomas Fermi wavevector (k\_{TF}**2).)");

        prm.declare_entry(
          "RESTA SCREENING LENGTH",
          "6.61",
          dealii::Patterns::Double(0.0, 1000.0),
          "[Standard] Screening length estimate (in Bohr) to be used in the Resta preconditioner.");

        prm.declare_entry(
          "RESTA FERMI WAVEVECTOR",
          "5.81",
          dealii::Patterns::Double(0.0, 1000.0),
          R"([Standard] Fermi wavevector estimate (in Bohr\^-1) to be used in the Resta preconditioner.)");

        prm.declare_entry(
          "MIXING METHOD",
          "ANDERSON",
          dealii::Patterns::Selection(
            "ANDERSON|ANDERSON_WITH_KERKER|ANDERSON_WITH_RESTA|LOW_RANK_DIELECM_PRECOND"),
          "[Standard] Method for density mixing. ANDERSON is the default option.");


        prm.declare_entry(
          "CONSTRAINT MAGNETIZATION",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Boolean parameter specifying whether to keep the starting magnetization fixed through the SCF iterations. Default is false.");


        prm.declare_entry(
          "PURE STATE",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Explictly solves for a pure Kohn-Sham state instead of an ensemble Kohn-Sham state implemented using Fermi-Dirac smearing. If this parameter is set to true, it overrides the Fermi-Dirac smearing temperature. Default is false.");

        prm.declare_entry(
          "STARTING WFC",
          "RANDOM",
          dealii::Patterns::Selection("ATOMIC|RANDOM"),
          "[Standard] Sets the type of the starting Kohn-Sham wavefunctions guess: Atomic(Superposition of single atom atomic orbitals. Atom types for which atomic orbitals are not available, random wavefunctions are taken. Currently, atomic orbitals data is not available for all atoms.), Random(The starting guess for all wavefunctions are taken to be random). Default: RANDOM.");

        prm.declare_entry(
          "COMPUTE ENERGY EACH ITER",
          "false",
          dealii::Patterns::Bool(),
          "[Advanced] Boolean parameter specifying whether to compute the total energy at the end of every SCF. Setting it to false can lead to some computational time savings. Default value is false but is internally set to true if VERBOSITY==5");

        prm.declare_entry(
          "USE ENERGY RESIDUAL METRIC",
          "false",
          dealii::Patterns::Bool(),
          "[Advanced] Boolean parameter specifying whether to use the energy residual metric (equation 7.23 of Richard Matrin second edition) for convergence check. Setting it to false can lead to some computational time savings. Default value is false");


        prm.enter_subsection("LOW RANK DIELECM PRECOND");
        {
          prm.declare_entry(
            "METHOD SUB TYPE",
            "ADAPTIVE",
            dealii::Patterns::Selection("ADAPTIVE|ACCUMULATED_ADAPTIVE"),
            R"([Advanced] Method subtype for LOW\_RANK\_DIELECM\_PRECOND.)");

          prm.declare_entry(
            "STARTING NORM LARGE DAMPING",
            "2.0",
            dealii::Patterns::Double(0.0, 10.0),
            "[Advanced] L2 norm electron density difference below which damping parameter is set to SCF parameters::MIXING PARAMETER, otherwise set to 0.1.");


          prm.declare_entry(
            "ADAPTIVE RANK REL TOL",
            "0.3",
            dealii::Patterns::Double(0.0, 1.0),
            "[Standard] Relative error metric on the low rank approximation error that adaptively sets the rank in each SCF iteration step.");

          prm.declare_entry(
            "BETA TOL",
            "0.1",
            dealii::Patterns::Double(0.0),
            R"([Advanced] Sets tolerance on deviation of linear indicator value from the ideal value of 1.0. For METHOD SUB TYPE=ACCUMULATED\_ADAPTIVE.)");

          prm.declare_entry(
            "POISSON SOLVER ABS TOL",
            "1e-6",
            dealii::Patterns::Double(0.0),
            "[Advanced] Absolute poisson solver tolerance for electrostatic potential response computation.");

          prm.declare_entry(
            "USE SINGLE PREC DENSITY RESPONSE",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Turns on single precision optimization in density response computation.");

          prm.declare_entry(
            "ESTIMATE JAC CONDITION NO",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Estimate condition number of the Jacobian at the final SCF iteration step using a low rank approximation with ADAPTIVE RANK REL TOL=1.0e-5.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Eigen-solver parameters");
        {
          prm.declare_entry(
            "NUMBER OF KOHN-SHAM WAVEFUNCTIONS",
            "0",
            dealii::Patterns::Integer(0),
            "[Standard] Number of Kohn-Sham wavefunctions to be computed. For spin-polarized calculations, this parameter denotes the number of Kohn-Sham wavefunctions to be computed for each spin. A recommended value for this parameter is to set it to N/2+Nb where N is the number of electrons. Use Nb to be 5-10 percent of N/2 for insulators and for metals use Nb to be 10-20 percent of N/2. If 5-20 percent of N/2 is less than 10 wavefunctions, set Nb to be atleast 10. Default value of 0 automatically sets the number of Kohn-Sham wavefunctions close to 20 percent more than N/2. CAUTION: use more states when using higher electronic temperature.");



          prm.declare_entry("NUMBER OF CORE EIGEN STATES FOR MIXED PREC RR",
                            "0",
                            dealii::Patterns::Integer(0),
                            "[Advanced] For mixed precision optimization.");



          prm.declare_entry(
            "CHEBYSHEV POLYNOMIAL DEGREE",
            "0",
            dealii::Patterns::Integer(0, 2000),
            "[Advanced] Chebyshev polynomial degree to be employed for the Chebyshev filtering subspace iteration procedure to dampen the unwanted spectrum of the Kohn-Sham Hamiltonian. If set to 0, a default value depending on the upper bound of the eigen-spectrum is used. See Phani Motamarri et.al., J. Comp. Phys. 253, 308-343 (2013).");

          prm.declare_entry(
            "CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF",
            "1.34",
            dealii::Patterns::Double(0, 2000),
            "[Advanced] Chebyshev polynomial degree first scf scaling factor. Only activated for pseudopotential calculations.");


          prm.declare_entry(
            "CHEBYSHEV FILTER TOLERANCE",
            "0.0",
            dealii::Patterns::Double(-1.0e-12),
            "[Advanced] Parameter specifying the accuracy of the occupied eigenvectors close to the Fermi-energy computed using Chebyshev filtering subspace iteration procedure. For default value of 0.0, we heuristically set the value between 1e-3 and 5e-2 depending on the MIXING METHOD used.");

          prm.declare_entry(
            "ORTHOGONALIZATION TYPE",
            "Auto",
            dealii::Patterns::Selection("GS|CGS|Auto"),
            "[Advanced] Parameter specifying the type of orthogonalization to be used: GS(Gram-Schmidt Orthogonalization using SLEPc library) and CGS(Cholesky-Gram-Schmidt Orthogonalization). Auto is the default and recommended option, which chooses GS for all-electron case and CGS for pseudopotential case. On GPUs CGS is the only route currently implemented.");

          prm.declare_entry(
            "CHEBY WFC BLOCK SIZE",
            "200",
            dealii::Patterns::Integer(1),
            "[Advanced] Chebyshev filtering procedure involves the matrix-matrix multiplication where one matrix corresponds to the discretized Hamiltonian and the other matrix corresponds to the wavefunction matrix. The matrix-matrix multiplication is accomplished in a loop over the number of blocks of the wavefunction matrix to reduce the memory footprint of the code. This parameter specifies the block size of the wavefunction matrix to be used in the matrix-matrix multiplication. The optimum value is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting CHEBY WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/CHEBY WFC BLOCK SIZE equals an integer value. Default value is 200.");

          prm.declare_entry(
            "WFC BLOCK SIZE",
            "400",
            dealii::Patterns::Integer(1),
            "[Advanced]  This parameter specifies the block size of the wavefunction matrix to be used for memory optimization purposes in the orthogonalization, Rayleigh-Ritz, and density computation steps. The optimum block size is dependent on the computing architecture. For optimum work sharing during band parallelization (NPBAND > 1), we recommend adjusting WFC BLOCK SIZE and NUMBER OF KOHN-SHAM WAVEFUNCTIONS such that NUMBER OF KOHN-SHAM WAVEFUNCTIONS/NPBAND/WFC BLOCK SIZE equals an integer value. Default value is 400.");

          prm.declare_entry(
            "SUBSPACE ROT DOFS BLOCK SIZE",
            "10000",
            dealii::Patterns::Integer(1),
            "[Developer] This block size is used for memory optimization purposes in subspace rotation step in Cholesky-Gram-Schmidt orthogonalization and Rayleigh-Ritz steps. Default value is 10000.");

          prm.declare_entry(
            "SCALAPACKPROCS",
            "0",
            dealii::Patterns::Integer(0, 300),
            "[Advanced] Uses a processor grid of SCALAPACKPROCS times SCALAPACKPROCS for parallel distribution of the subspace projected matrix in the Rayleigh-Ritz step and the overlap matrix in the Cholesky-Gram-Schmidt step. Default value is 0 for which a thumb rule is used (see http://netlib.org/scalapack/slug/node106.html). If ELPA is used, twice the value obtained from the thumb rule is used as ELPA scales much better than ScaLAPACK.");

          prm.declare_entry(
            "SCALAPACK BLOCK SIZE",
            "0",
            dealii::Patterns::Integer(0, 300),
            "[Advanced] ScaLAPACK process grid block size. Also sets the block size for ELPA if linked to ELPA. Default value of zero sets a heuristic block size. Note that if ELPA GPU KERNEL is set to true and ELPA is configured to run on GPUs, the SCALAPACK BLOCK SIZE is set to a power of 2.");

          prm.declare_entry(
            "USE ELPA",
            "true",
            dealii::Patterns::Bool(),
            "[Standard] Use ELPA instead of ScaLAPACK for diagonalization of subspace projected Hamiltonian and Cholesky-Gram-Schmidt orthogonalization.  Default setting is true.");

          prm.declare_entry(
            "USE APPROXIMATE OVERLAP MATRIX",
            "true",
            dealii::Patterns::Bool(),
            "[Standard] Use approximate overlap matrix (diagonal for FE basis overlap).  Default setting is true.");

          prm.declare_entry(
            "USE RESIDUAL CHFSI",
            "true",
            dealii::Patterns::Bool(),
            "[Advanced] Builds the Chebyshev filtered subspace in full precision based on a residual-ChFSI algorithm.");

          prm.declare_entry(
            "SUBSPACE PROJ SHEP GPU",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Solve a standard hermitian eigenvalue problem in the Rayleigh Ritz step instead of a generalized hermitian eigenvalue problem on GPUs. Default setting is true.");


          prm.declare_entry(
            "USE MIXED PREC CGS SR",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in subspace rotation step of CGS orthogonalization, if ORTHOGONALIZATION TYPE is set to CGS. Default setting is false.");

          prm.declare_entry(
            "USE MIXED PREC XTOX",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in X^{T} Overlap matrix times X computation. Default setting is false.");


          prm.declare_entry(
            "USE MIXED PREC XTHX",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in X^{T} Hamiltonian matrix times X computation. Default setting is false.");


          prm.declare_entry(
            "USE MIXED PREC RR_SR",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use mixed precision arithmetic in Rayleigh-Ritz subspace rotation step. Default setting is false.");

          prm.declare_entry(
            "USE SINGLE PREC COMMUN CHEBY",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use single precision communication in Chebyshev filtering. Default setting is false.");

          prm.declare_entry(
            "USE MIXED PREC COMMUN ONLY XTOX XTHX",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use mixed precision communication only for XtOX and XtHX instead of mixed precision compute and communication. This setting has been found to be more optimal on certain architectures. Default setting is false.");

          prm.declare_entry(
            "USE SINGLE PREC CHEBY",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Use a modified single precision algorithm for Chebyshev filtering. This cannot be used in conjunction with spectrum splitting. Default setting is false.");
          prm.declare_entry(
            "TENSOR OP TYPE SINGLE PREC CHEBY",
            "FP32",
            dealii::Patterns::Selection("FP32|TF32"),
            "[Advanced] Tensor operation datatype for the modified single precision algorithm for Chebyshev filtering, this only used on Nvidia GPUs with compute capability greater than 80. Default setting is FP32.");

          prm.declare_entry(
            "OVERLAP COMPUTE COMMUN CHEBY",
            "true",
            dealii::Patterns::Bool(),
            "[Advanced] Overlap communication and computation in Chebyshev filtering. This option can only be activated for USE GPU=true. Default setting is true.");

          prm.declare_entry(
            "OVERLAP COMPUTE COMMUN ORTHO RR",
            "true",
            dealii::Patterns::Bool(),
            "[Advanced] Overlap communication and computation in orthogonalization and Rayleigh-Ritz. This option can only be activated for USE GPU=true. Default setting is true.");

          prm.declare_entry(
            "ALGO",
            "NORMAL",
            dealii::Patterns::Selection("NORMAL|FAST"),
            "[Standard] In the FAST mode, spectrum splitting technique is used in Rayleigh-Ritz step, and mixed precision arithmetic algorithms are used in Rayleigh-Ritz and Cholesky factorization based orthogonalization step. For spectrum splitting, 85 percent of the total number of wavefunctions are taken to be core states, which holds good for most systems including metallic systems assuming NUMBER OF KOHN-SHAM WAVEFUNCTIONS to be around 10 percent more than N/2. FAST setting is strongly recommended for large-scale (> 10k electrons) system sizes. Both NORMAL and FAST setting use Chebyshev filtered subspace iteration technique. If manual options for mixed precision and spectum splitting are being used, please use NORMAL setting for ALGO. Default setting is NORMAL.");


          prm.declare_entry(
            "REUSE LANCZOS UPPER BOUND",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Reuse upper bound of unwanted spectrum computed in the first SCF iteration via Lanczos iterations. Default setting is false.");

          prm.declare_entry(
            "ALLOW MULTIPLE PASSES POST FIRST SCF",
            "true",
            dealii::Patterns::Bool(),
            "[Advanced] Allow multiple chebyshev filtering passes in the SCF iterations after the first one. Default setting is true.");

          prm.declare_entry(
            "HIGHEST STATE OF INTEREST FOR CHEBYSHEV FILTERING",
            "0",
            dealii::Patterns::Integer(0),
            "[Standard] The highest state till which the Kohn Sham wavefunctions are computed accurately during Chebyshev filtering in NSCF/BANDS calculations. By default, this is set to N/2*1.05 where N is the number of electrons. It is strongly encouraged to have at least 10-15 percent buffer between this parameter and the total number of wavefunctions employed for the NSCF/BANDS calculation.");

          prm.declare_entry(
            "RESTRICT TO SINGLE FILTER PASS",
            "false",
            dealii::Patterns::Bool(),
            "[Advanced] Restrict to single chebyshev filter pass in each SCF. This setting is only used for timing measurements of stable single SCF iteration.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Poisson problem parameters");
      {
        prm.declare_entry(
          "MAXIMUM ITERATIONS",
          "20000",
          dealii::Patterns::Integer(0, 20000),
          "[Advanced] Maximum number of iterations to be allowed for Poisson problem convergence.");

        prm.declare_entry(
          "TOLERANCE",
          "1e-10",
          dealii::Patterns::Double(0, 1.0),
          "[Advanced] Absolute tolerance on the residual as stopping criterion for Poisson problem convergence.");

        prm.declare_entry("GPU MODE",
                          "true",
                          dealii::Patterns::Bool(),
                          "[Advanced] Toggle GPU MODE in Poisson solve.");

        prm.declare_entry("VSELF GPU MODE",
                          "true",
                          dealii::Patterns::Bool(),
                          "[Advanced] Toggle GPU MODE in vself Poisson solve.");
      }
      prm.leave_subsection();


      prm.enter_subsection("Helmholtz problem parameters");
      {
        prm.declare_entry(
          "MAXIMUM ITERATIONS HELMHOLTZ",
          "10000",
          dealii::Patterns::Integer(0, 20000),
          "[Advanced] Maximum number of iterations to be allowed for Helmholtz problem convergence.");

        prm.declare_entry(
          "ABSOLUTE TOLERANCE HELMHOLTZ",
          "1e-10",
          dealii::Patterns::Double(0, 1.0),
          "[Advanced] Absolute tolerance on the residual as stopping criterion for Helmholtz problem convergence.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Molecular Dynamics");
      {
        prm.declare_entry(
          "ATOMIC MASSES FILE",
          "",
          dealii::Patterns::Anything(),
          "[Standard] Input atomic masses file name. File format: atomicNumber1 atomicMass1 (row1), atomicNumber2 atomicMass2 (row2) and so on. Units: a.m.u.");

        prm.declare_entry(
          "BOMD",
          "false",
          dealii::Patterns::Bool(),
          "[Standard] Perform Born-Oppenheimer molecular dynamics.");

        prm.declare_entry(
          "EXTRAPOLATE DENSITY",
          "0",
          dealii::Patterns::Integer(0, 2),
          "[Standard] Parameter controlling the reuse of ground-state density during molecular dynamics. The options are 0 default setting where superposition of atomic densities is the initial rho, 1 (second order extrapolation of density), and 2 (extrapolation of split density and the atomic densities are added) Option 2 is not enabled for spin-polarized case. Default setting is 0.");

        prm.declare_entry(
          "MAX JACOBIAN RATIO FACTOR",
          "1.5",
          dealii::Patterns::Double(0.9, 3.0),
          "[Developer] Maximum scaling factor for maximum jacobian ratio of FEM mesh when mesh is deformed.");

        prm.declare_entry(
          "STARTING TEMPERATURE",
          "300.0",
          dealii::Patterns::Double(0.0),
          "[Standard] Starting temperature in K for MD simulation.");

        prm.declare_entry(
          "THERMOSTAT TIME CONSTANT",
          "100",
          dealii::Patterns::Double(0.0),
          "[Standard] Ratio of Time constant of thermostat and MD timestep. ");

        prm.declare_entry(
          "TEMPERATURE CONTROLLER TYPE",
          "NO_CONTROL",
          dealii::Patterns::Selection(
            "NO_CONTROL|RESCALE|NOSE_HOVER_CHAINS|CSVR"),
          R"([Standard] Method of controlling temperature in the MD run. NO\_CONTROL is the default option.)");

        prm.declare_entry("TIME STEP",
                          "0.5",
                          dealii::Patterns::Double(0.0),
                          "[Standard] Time step in femtoseconds.");

        prm.declare_entry("NUMBER OF STEPS",
                          "1000",
                          dealii::Patterns::Integer(0, 200000),
                          "[Standard] Number of time steps.");
        prm.declare_entry("TRACKING ATOMIC NO",
                          "0",
                          dealii::Patterns::Integer(0, 200000),
                          "[Standard] The atom Number to track.");

        prm.declare_entry("MAX WALL TIME",
                          "2592000.0",
                          dealii::Patterns::Double(0.0),
                          "[Standard] Maximum Wall Time in seconds");
      }
      prm.leave_subsection();
    }
  } // namespace internalDftParameters

  dftParameters::dftParameters()
  {
    finiteElementPolynomialOrder               = 1;
    finiteElementPolynomialOrderElectrostatics = 1;
    n_refinement_steps                         = 1;
    numberEigenValues                          = 1;
    XCType                                     = "GGA-PBE";
    spinPolarized                              = 0;
    modelXCInputFile                           = "";
    auxBasisTypeXC                             = "";
    auxBasisDataXC                             = "";
    nkx                                        = 1;
    nky                                        = 1;
    nkz                                        = 1;
    offsetFlagX                                = 0;
    offsetFlagY                                = 0;
    offsetFlagZ                                = 0;
    chebyshevOrder                             = 1;
    numPass                                    = 1;
    numSCFIterations                           = 1;
    maxLinearSolverIterations                  = 1;
    poissonGPU                                 = true;
    vselfGPU                                   = true;
    mixingHistory                              = 1;
    npool                                      = 1;
    maxLinearSolverIterationsHelmholtz         = 1;

    functionalTestName                       = "";
    radiusAtomBall                           = 0.0;
    mixingParameter                          = 0.5;
    spinMixingEnhancementFactor              = 4.0;
    absLinearSolverTolerance                 = 1e-10;
    selfConsistentSolverTolerance            = 1e-10;
    TVal                                     = 500;
    tot_magnetization                        = 0.0;
    useAtomicMagnetizationGuessConstraintMag = false;
    absLinearSolverToleranceHelmholtz        = 1e-10;
    chebyshevTolerance                       = 1e-02;
    mixingMethod                             = "";
    optimizationMode                         = "";
    ionOptSolver                             = "";
    cellOptSolver                            = "";

    isPseudopotential           = false;
    periodicX                   = false;
    periodicY                   = false;
    periodicZ                   = false;
    useSymm                     = false;
    timeReversal                = false;
    pseudoTestsFlag             = false;
    constraintMagnetization     = false;
    pureState                   = false;
    writeDosFile                = false;
    writeLdosFile               = false;
    writePdosFile               = false;
    smearTval                   = 500;
    intervalSize                = 0.01;
    writeLocalizationLengths    = false;
    std::string coordinatesFile = "";
    domainBoundingVectorsFile   = "";
    kPointDataFile              = "";
    ionRelaxFlagsFile           = "";
    orthogType                  = "";
    algoType                    = "";
    pseudoPotentialFile         = "";
    meshSizesFile               = "";

    std::string coordinatesGaussianDispFile = "";

    outerAtomBallRadius            = 2.5;
    innerAtomBallRadius            = 0.0;
    meshSizeOuterDomain            = 10.0;
    meshSizeInnerBall              = 1.0;
    meshSizeOuterBall              = 1.0;
    numLevels                      = 1;
    numberWaveFunctionsForEstimate = 5;
    topfrac                        = 0.1;
    kerkerParameter                = 0.05;

    isIonForce             = false;
    isCellStress           = false;
    isBOMD                 = false;
    nonSelfConsistentForce = false;
    forceRelaxTol          = 1e-4; // Hartree/Bohr
    stressRelaxTol         = 1e-6; // Hartree/Bohr^3
    toleranceKinetic       = 1e-03;
    cellConstraintType     = 12; // all cell components to be relaxed

    verbosity                                      = 0;
    keepScratchFolder                              = false;
    restartFolder                                  = ".";
    saveQuadData                                   = false;
    loadQuadData                                   = false;
    restartSpinFromNoSpin                          = false;
    reproducible_output                            = false;
    meshAdaption                                   = false;
    pinnedNodeForPBC                               = true;
    startingWFCType                                = "";
    restrictToOnePass                              = false;
    writeWfcSolutionFields                         = false;
    printKE                                        = false;
    writeDensitySolutionFields                     = false;
    writeOutputTotalElectrostaticsPotential        = false;
    writePlanarAverageTotalElectrostaticsPotential = false;
    writePlanarAverageVeffPotential                = false;
    writeDensityQuadData                           = false;
    wfcBlockSize                                   = 400;
    chebyWfcBlockSize                              = 400;
    subspaceRotDofsBlockSize                       = 2000;
    nbandGrps                                      = 1;
    computeEnergyEverySCF                          = true;
    scalapackParalProcs                            = 0;
    scalapackBlockSize                             = 50;
    natoms                                         = 0;
    natomTypes                                     = 0;
    numCoreWfcForMixedPrecRR                       = 0;
    reuseWfcGeoOpt                                 = false;
    reuseDensityGeoOpt                             = 0;
    mpiAllReduceMessageBlockSizeMB                 = 2.0;
    useSubspaceProjectedSHEPGPU                    = false;
    useMixedPrecCGS_SR                             = false;
    useMixedPrecXtOX                               = false;
    useMixedPrecXtHX                               = false;
    approxOverlapMatrix                            = true;
    useReformulatedChFSI                           = false;
    useMixedPrecSubspaceRotRR                      = false;
    useMixedPrecCommunOnlyXtHXXtOX                 = false;
    useELPA                                        = false;
    constraintsParallelCheck                       = true;
    createConstraintsFromSerialDofhandler          = true;
    bandParalOpt                                   = true;
    autoAdaptBaseMeshSize                          = true;
    readWfcForPdosPspFile                          = false;
    useDevice                                      = false;
    deviceFineGrainedTimings                       = false;
    allowFullCPUMemSubspaceRot                     = true;
    useSinglePrecCommunCheby                       = false;
    overlapComputeCommunCheby                      = false;
    overlapComputeCommunOrthoRR                    = false;
    autoDeviceBlockSizes                           = true;
    maxJacobianRatioFactorForMD                    = 1.5;
    extrapolateDensity                             = 0;
    timeStepBOMD                                   = 0.5;
    numberStepsBOMD                                = 1000;
    gaussianConstantForce                          = 0.75;
    gaussianOrderForce                             = 4.0;
    gaussianOrderMoveMeshToAtoms                   = 4.0;
    useFlatTopGenerator                            = false;
    diracDeltaKernelScalingConstant                = 0.1;
    chebyshevFilterPolyDegreeFirstScfScalingFactor = 1.34;
    useDensityMatrixPerturbationRankUpdates        = false;
    smearedNuclearCharges                          = false;
    floatingNuclearCharges                         = false;
    multipoleBoundaryConditions                    = false;
    nonLinearCoreCorrection                        = false;
    maxLineSearchIterCGPRP                         = 5;
    atomicMassesFile                               = "";
    useDeviceDirectAllReduce                       = false;
    pspCutoffImageCharges                          = 15.0;
    netCharge                                      = 0;
    reuseLanczosUpperBoundFromFirstCall            = false;
    allowMultipleFilteringPassesAfterFirstScf      = true;
    useELPADeviceKernel                            = false;
    // New Paramters for moleculardyynamics class
    startingTempBOMD           = 300;
    thermostatTimeConstantBOMD = 100;
    MaxWallTime                = 2592000.0;
    tempControllerTypeBOMD     = "";
    MDTrack                    = 0;

    hubbardFileName = "";

    // New paramter for selecting mode and NEB parameters
    TotalImages = 1;


    dc_dispersioncorrectiontype = 0;
    dc_d3dampingtype            = 2;
    dc_d3ATM                    = false;
    dc_d4MBD                    = false;
    dc_dampingParameterFilename = "";
    dc_d3cutoff2                = 94.8683298050514;
    dc_d3cutoff3                = 40.0;
    dc_d3cutoffCN               = 40.0;

    /** parameters for LRD preconditioner **/
    startingNormLRDLargeDamping   = 2.0;
    adaptiveRankRelTolLRD         = 0.3;
    methodSubTypeLRD              = "";
    betaTol                       = 0.1;
    absPoissonSolverToleranceLRD  = 1.0e-6;
    singlePrecLRD                 = false;
    estimateJacCondNoFinalSCFIter = false;
    /*****************************************/
    bfgsStepMethod     = "QN";
    usePreconditioner  = false;
    lbfgsNumPastSteps  = 5;
    maxOptIter         = 300;
    maxStaggeredCycles = 100;
    maxIonUpdateStep   = 0.5;
    maxCellUpdateStep  = 0.1;

    // Parameters for confining potential
    confiningPotential   = false;
    confiningInnerPotRad = 17.0;
    confiningOuterPotRad = 20.0;
    confiningWParam      = 1.0;
    confiningCParam      = 1.0;


    writeStructreEnergyForcesFileForPostProcess = false;
  }


  void
  dftParameters::parse_parameters(const std::string &parameter_file,
                                  const MPI_Comm    &mpi_comm_parent,
                                  const bool         printParams,
                                  const std::string  mode,
                                  const std::string  restartFilesPath,
                                  const dftfe::Int   _verbosity,
                                  const bool         _useDevice)
  {
    dealii::ParameterHandler prm;
    internalDftParameters::declare_parameters(prm);
    // prm.parse_input(parameter_file);
    prm.parse_input(parameter_file, "", true);
    solverMode          = mode;
    verbosity           = _verbosity;
    useDevice           = _useDevice;
    reproducible_output = prm.get_bool("REPRODUCIBLE OUTPUT");
    keepScratchFolder   = prm.get_bool("KEEP SCRATCH FOLDER");
    restartFolder       = restartFilesPath;
    auto entriesNotSet  = prm.get_entries_wrongly_not_set();
    if (auto memOptSet = entriesNotSet.find("MEM_20OPT_20MODE");
        memOptSet != entriesNotSet.end())
      prm.set("MEM OPT MODE", solverMode == "NSCF" || solverMode == "BANDS");
    memOptMode = prm.get_bool("MEM OPT MODE");
    writeStructreEnergyForcesFileForPostProcess =
      prm.get_bool("WRITE STRUCTURE ENERGY FORCES DATA POST PROCESS");

    prm.enter_subsection("GPU");
    {
      deviceFineGrainedTimings =
        useDevice && prm.get_bool("FINE GRAINED GPU TIMINGS");
      allowFullCPUMemSubspaceRot =
        useDevice && prm.get_bool("SUBSPACE ROT FULL CPU MEM");
      autoDeviceBlockSizes = useDevice && prm.get_bool("AUTO GPU BLOCK SIZES");
      useDeviceDirectAllReduce =
        useDevice && prm.get_bool("USE GPUDIRECT MPI ALL REDUCE");
      useDCCL             = useDevice && prm.get_bool("USE DCCL");
      useELPADeviceKernel = useDevice && prm.get_bool("USE ELPA GPU KERNEL");
    }
    prm.leave_subsection();

    prm.enter_subsection("Post-processing Options");
    {
      writeWfcSolutionFields     = prm.get_bool("WRITE WFC FE MESH");
      printKE                    = prm.get_bool("PRINT KINETIC ENERGY");
      writeDensitySolutionFields = prm.get_bool("WRITE DENSITY FE MESH");
      writeOutputTotalElectrostaticsPotential =
        prm.get_bool("WRITE TOTAL ELECTROSTATICS POTENTIAL");
      writeDensityQuadData = prm.get_bool("WRITE DENSITY QUAD DATA");
      writeDosFile         = prm.get_bool("WRITE DENSITY OF STATES");
      writeLdosFile        = prm.get_bool("WRITE LOCAL DENSITY OF STATES");
      writePdosFile        = prm.get_bool("WRITE PROJECTED DENSITY OF STATES");
      smearTval            = prm.get_double("DOS SMEAR TEMPERATURE");
      intervalSize         = prm.get_double("DELTA ENERGY");
      writeLocalizationLengths = prm.get_bool("WRITE LOCALIZATION LENGTHS");
      readWfcForPdosPspFile =
        prm.get_bool("READ ATOMIC WFC PDOS FROM PSP FILE");
      writeLocalizationLengths = prm.get_bool("WRITE LOCALIZATION LENGTHS");

      prm.enter_subsection("XY-Planar Average post-processing");
      {
        zPlanarAverageDensity =
          prm.get_bool("XY-PLANAR AVERAGE OF ELECTRON DENSITY");
        zPlanarAveragePhi =
          prm.get_bool("XY-PLANAR AVERAGE OF ELECTROSTATIC POTENTIAL");
        zPlanarAverageVbare =
          prm.get_bool("XY-PLANAR AVERAGE OF BARE POTENTIAL");
        noOfSamplingPointsX =
          prm.get_integer("NUMBER OF SAMPLING POINTS ALONG X");
        noOfSamplingPointsY =
          prm.get_integer("NUMBER OF SAMPLING POINTS ALONG Y");
        noOfSamplingPointsZ =
          prm.get_integer("NUMBER OF SAMPLING POINTS ALONG Z");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("FunctionalTest");
    {
      functionalTestName = prm.get("TEST NAME");
    }
    prm.leave_subsection();
    prm.enter_subsection("Parallelization");
    {
      npool        = prm.get_integer("NPKPT");
      nbandGrps    = prm.get_integer("NPBAND");
      bandParalOpt = prm.get_bool("BAND PARAL OPT");
      mpiAllReduceMessageBlockSizeMB =
        prm.get_double("MPI ALLREDUCE BLOCK SIZE");
    }
    prm.leave_subsection();

    prm.enter_subsection("SCF Checkpointing and Restart");
    {
      saveQuadData          = prm.get_bool("SAVE QUAD DATA");
      loadQuadData          = prm.get_bool("LOAD QUAD DATA");
      restartSpinFromNoSpin = prm.get_bool("RESTART SP FROM NO SP");
      if (solverMode == "NEB")
        saveQuadData = true;
    }
    prm.leave_subsection();

    prm.enter_subsection("Geometry");
    {
      natoms                      = prm.get_integer("NATOMS");
      natomTypes                  = prm.get_integer("NATOM TYPES");
      coordinatesFile             = prm.get("ATOMIC COORDINATES FILE");
      coordinatesGaussianDispFile = prm.get("ATOMIC DISP COORDINATES FILE");
      domainBoundingVectorsFile   = prm.get("DOMAIN VECTORS FILE");
      prm.enter_subsection("Optimization");
      {
        optimizationMode       = prm.get("OPTIMIZATION MODE");
        ionOptSolver           = prm.get("ION OPT SOLVER");
        cellOptSolver          = prm.get("CELL OPT SOLVER");
        maxLineSearchIterCGPRP = prm.get_integer("MAX LINE SEARCH ITER");
        nonSelfConsistentForce = prm.get_bool("NON SELF CONSISTENT FORCE");
        isIonForce =
          prm.get_bool("ION FORCE") ||
          ((optimizationMode == "ION" || optimizationMode == "IONCELL") &&
           solverMode == "GEOOPT");
        forceRelaxTol     = prm.get_double("FORCE TOL");
        ionRelaxFlagsFile = prm.get("ION RELAX FLAGS FILE");
        isCellStress =
          prm.get_bool("CELL STRESS") ||
          ((optimizationMode == "CELL" || optimizationMode == "IONCELL") &&
           solverMode == "GEOOPT");
        stressRelaxTol     = prm.get_double("STRESS TOL");
        cellConstraintType = prm.get_integer("CELL CONSTRAINT TYPE");
        reuseWfcGeoOpt     = prm.get_bool("REUSE WFC");
        reuseDensityGeoOpt = prm.get_integer("REUSE DENSITY");
        bfgsStepMethod     = prm.get("BFGS STEP METHOD");
        usePreconditioner  = prm.get_bool("USE PRECONDITIONER");
        lbfgsNumPastSteps  = prm.get_integer("LBFGS HISTORY");
        maxOptIter         = prm.get_integer("MAXIMUM OPTIMIZATION STEPS");
        maxStaggeredCycles = prm.get_integer("MAXIMUM STAGGERED CYCLES");
        maxIonUpdateStep   = prm.get_double("MAXIMUM ION UPDATE STEP");
        maxCellUpdateStep  = prm.get_double("MAXIMUM CELL UPDATE STEP");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
      radiusAtomBall           = prm.get_double("SELF POTENTIAL RADIUS");
      periodicX                = prm.get_bool("PERIODIC1");
      periodicY                = prm.get_bool("PERIODIC2");
      periodicZ                = prm.get_bool("PERIODIC3");
      constraintsParallelCheck = prm.get_bool("CONSTRAINTS PARALLEL CHECK");
      createConstraintsFromSerialDofhandler =
        prm.get_bool("CONSTRAINTS FROM SERIAL DOFHANDLER");
      pinnedNodeForPBC       = prm.get_bool("POINT WISE DIRICHLET CONSTRAINT");
      smearedNuclearCharges  = prm.get_bool("SMEARED NUCLEAR CHARGES");
      floatingNuclearCharges = prm.get_bool("FLOATING NUCLEAR CHARGES");
      multipoleBoundaryConditions =
        prm.get_bool("MULTIPOLE BOUNDARY CONDITIONS");
      prm.enter_subsection("External Electric Potential");
      {
        applyExternalPotential =
          prm.get_bool("INCLUDE EXTERNAL ELECTROSTATIC POTENTIAL BIAS");
        includeVselfInConstraints = true; // No more a parameter
        externalPotentialType =
          prm.get("TYPE OF EXTERNAL ELECTOSTATIC POTENTIAL BIAS");
        externalPotentialSlope = prm.get_double("SLOPE OF EXTERNAL POTENTIAL");
        appliedPotentialDifference =
          prm.get_double("CONSTRAINED POTENTIAL DIFFERENCE VALUE");
        potentialValueL =
          prm.get_double("CONSTRAINED POTENTIAL DIFFERENCE LEFT SURFACE VALUE");
        zCoordinateL =
          prm.get_double("CONSTRAINED POTENTIAL LEFT SURFACE POSITION");
        zCoordinateR =
          prm.get_double("CONSTRAINED POTENTIAL RIGHT SURFACE POSITION");
        localizedWidth =
          prm.get_double("CONSTRAINED POTENTIAL LOCALIZED WIDTH");
        emaxPos = prm.get_double("POSITION OF SAW TOOTH POTENTIAL MAX");
        eopreg  = prm.get_double("REGION OF SAW TOOTH POTENTIAL DECREASE");
        // probeWidth = prm.get_double("PROBE WIDTH OF PROBE POTENTIAL
        // DIFFERENCE");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Finite element mesh parameters");
    {
      finiteElementPolynomialOrder = prm.get_integer("POLYNOMIAL ORDER");
      finiteElementPolynomialOrderElectrostatics =
        prm.get_integer("POLYNOMIAL ORDER ELECTROSTATICS") == 0 ?
          prm.get_integer("POLYNOMIAL ORDER") :
          prm.get_integer("POLYNOMIAL ORDER ELECTROSTATICS");
      finiteElementPolynomialOrderRhoNodal =
        prm.get_integer("POLYNOMIAL ORDER DENSITY NODAL") == 0 ?
          (finiteElementPolynomialOrder + 2 >
               finiteElementPolynomialOrderElectrostatics ?
             finiteElementPolynomialOrder + 2 :
             finiteElementPolynomialOrderElectrostatics) :
          prm.get_integer("POLYNOMIAL ORDER DENSITY NODAL");
      densityQuadratureRule = prm.get_integer("DENSITY QUADRATURE RULE");
      prm.enter_subsection("Auto mesh generation parameters");
      {
        outerAtomBallRadius   = prm.get_double("ATOM BALL RADIUS");
        innerAtomBallRadius   = prm.get_double("INNER ATOM BALL RADIUS");
        meshSizeOuterDomain   = prm.get_double("BASE MESH SIZE");
        meshSizeInnerBall     = prm.get_double("MESH SIZE AT ATOM");
        meshSizeOuterBall     = prm.get_double("MESH SIZE AROUND ATOM");
        meshAdaption          = prm.get_bool("MESH ADAPTION");
        autoAdaptBaseMeshSize = prm.get_bool("AUTO ADAPT BASE MESH SIZE");
        topfrac               = prm.get_double("TOP FRAC");
        numLevels             = prm.get_double("NUM LEVELS");
        numberWaveFunctionsForEstimate =
          prm.get_integer("ERROR ESTIMATE WAVEFUNCTIONS");
        toleranceKinetic = prm.get_double("TOLERANCE FOR MESH ADAPTION");
        gaussianConstantForce =
          prm.get_double("GAUSSIAN CONSTANT FORCE GENERATOR");
        gaussianOrderForce = prm.get_double("GAUSSIAN ORDER FORCE GENERATOR");
        gaussianOrderMoveMeshToAtoms =
          prm.get_double("GAUSSIAN ORDER MOVE MESH TO ATOMS");
        useFlatTopGenerator = prm.get_bool("USE FLAT TOP GENERATOR");
        meshSizesFile       = prm.get("MESH SIZES FILE");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Brillouin zone k point sampling options");
    {
      prm.enter_subsection("Monkhorst-Pack (MP) grid generation");
      {
        nkx         = prm.get_integer("SAMPLING POINTS 1");
        nky         = prm.get_integer("SAMPLING POINTS 2");
        nkz         = prm.get_integer("SAMPLING POINTS 3");
        offsetFlagX = prm.get_integer("SAMPLING SHIFT 1");
        offsetFlagY = prm.get_integer("SAMPLING SHIFT 2");
        offsetFlagZ = prm.get_integer("SAMPLING SHIFT 3");
      }
      prm.leave_subsection();

      useSymm        = prm.get_bool("USE GROUP SYMMETRY");
      timeReversal   = prm.get_bool("USE TIME REVERSAL SYMMETRY");
      kPointDataFile = prm.get("kPOINT RULE FILE");
    }
    prm.leave_subsection();

    prm.enter_subsection("DFT functional parameters");
    {
      // Parameters for confining potential
      prm.enter_subsection("CONFINING POTENTIAL parameters");
      {
        confiningPotential   = prm.get_bool("APPLY CONFINING POTENTIAL");
        confiningInnerPotRad = prm.get_double("INNER RADIUS");
        confiningOuterPotRad = prm.get_double("OUTER RADIUS");
        confiningWParam      = prm.get_double("W PARAM");
        confiningCParam      = prm.get_double("C PARAM");
      }
      prm.leave_subsection();

      prm.enter_subsection("Dispersion Correction");
      {
        dc_dispersioncorrectiontype =
          prm.get_integer("DISPERSION CORRECTION TYPE");
        dc_d3dampingtype            = prm.get_integer("D3 DAMPING TYPE");
        dc_d3ATM                    = prm.get_bool("D3 ATM");
        dc_d4MBD                    = prm.get_bool("D4 MBD");
        dc_dampingParameterFilename = prm.get("DAMPING PARAMETERS FILE");
        dc_d3cutoff2                = prm.get_double("TWO BODY CUTOFF");
        dc_d3cutoff3                = prm.get_double("THREE BODY CUTOFF");
        dc_d3cutoffCN               = prm.get_double("CN CUTOFF");
      }
      prm.leave_subsection();
      isPseudopotential   = prm.get_bool("PSEUDOPOTENTIAL CALCULATION");
      pseudoTestsFlag     = prm.get_bool("PSEUDO TESTS FLAG");
      pseudoPotentialFile = prm.get("PSEUDOPOTENTIAL FILE NAMES LIST");
      XCType              = prm.get("EXCHANGE CORRELATION TYPE");
      spinPolarized       = prm.get_integer("SPIN POLARIZATION");
      modelXCInputFile    = prm.get("MODEL XC INPUT FILE");
      auxBasisTypeXC      = prm.get("AUX BASIS TYPE");
      auxBasisDataXC      = prm.get("AUX BASIS DATA");
      tot_magnetization   = prm.get_double("TOTAL MAGNETIZATION");
      useAtomicMagnetizationGuessConstraintMag =
        prm.get_bool("USE ATOMIC MAGNETIZATION GUESS FOR CONSTRAINT MAG");
      pspCutoffImageCharges = prm.get_double("PSP CUTOFF IMAGE CHARGES");
      netCharge             = prm.get_double("NET CHARGE");

      prm.enter_subsection("Hubbard Parameters");
      {
        hubbardFileName = prm.get("HUBBARD PARAMETERS FILE");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("SCF parameters");
    {
      TVal                          = prm.get_double("TEMPERATURE");
      numSCFIterations              = prm.get_integer("MAXIMUM ITERATIONS");
      selfConsistentSolverTolerance = prm.get_double("TOLERANCE");
      selfConsistentSolverEnergyTolerance = prm.get_double("ENERGY TOLERANCE");
      mixingHistory                       = prm.get_integer("MIXING HISTORY");
      mixingParameter                     = prm.get_double("MIXING PARAMETER");
      spinMixingEnhancementFactor =
        prm.get_double("SPIN MIXING ENHANCEMENT FACTOR");
      adaptAndersonMixingParameter =
        prm.get_bool("ADAPT ANDERSON MIXING PARAMETER");
      kerkerParameter            = prm.get_double("KERKER MIXING PARAMETER");
      restaFermiWavevector       = prm.get_double("RESTA FERMI WAVEVECTOR");
      restaScreeningLength       = prm.get_double("RESTA SCREENING LENGTH");
      mixingMethod               = prm.get("MIXING METHOD");
      constraintMagnetization    = prm.get_bool("CONSTRAINT MAGNETIZATION");
      pureState                  = prm.get_bool("PURE STATE");
      startingWFCType            = prm.get("STARTING WFC");
      computeEnergyEverySCF      = prm.get_bool("COMPUTE ENERGY EACH ITER");
      useEnergyResidualTolerance = prm.get_bool("USE ENERGY RESIDUAL METRIC");

      prm.enter_subsection("LOW RANK DIELECM PRECOND");
      {
        methodSubTypeLRD = prm.get("METHOD SUB TYPE");
        startingNormLRDLargeDamping =
          prm.get_double("STARTING NORM LARGE DAMPING");
        adaptiveRankRelTolLRD        = prm.get_double("ADAPTIVE RANK REL TOL");
        betaTol                      = prm.get_double("BETA TOL");
        absPoissonSolverToleranceLRD = prm.get_double("POISSON SOLVER ABS TOL");
        singlePrecLRD = prm.get_bool("USE SINGLE PREC DENSITY RESPONSE");
        estimateJacCondNoFinalSCFIter =
          prm.get_bool("ESTIMATE JAC CONDITION NO");
      }
      prm.leave_subsection();

      prm.enter_subsection("Eigen-solver parameters");
      {
        numberEigenValues =
          prm.get_integer("NUMBER OF KOHN-SHAM WAVEFUNCTIONS");
        numCoreWfcForMixedPrecRR =
          prm.get_integer("NUMBER OF CORE EIGEN STATES FOR MIXED PREC RR");
        chebyshevOrder       = prm.get_integer("CHEBYSHEV POLYNOMIAL DEGREE");
        useELPA              = prm.get_bool("USE ELPA");
        approxOverlapMatrix  = prm.get_bool("USE APPROXIMATE OVERLAP MATRIX");
        useReformulatedChFSI = prm.get_bool("USE RESIDUAL CHFSI");
        orthogType           = prm.get("ORTHOGONALIZATION TYPE");
        chebyshevTolerance   = prm.get_double("CHEBYSHEV FILTER TOLERANCE");
        wfcBlockSize         = prm.get_integer("WFC BLOCK SIZE");
        chebyWfcBlockSize    = prm.get_integer("CHEBY WFC BLOCK SIZE");
        subspaceRotDofsBlockSize =
          prm.get_integer("SUBSPACE ROT DOFS BLOCK SIZE");
        scalapackParalProcs       = prm.get_integer("SCALAPACKPROCS");
        scalapackBlockSize        = prm.get_integer("SCALAPACK BLOCK SIZE");
        useMixedPrecCGS_SR        = prm.get_bool("USE MIXED PREC CGS SR");
        useMixedPrecXtOX          = prm.get_bool("USE MIXED PREC XTOX");
        useMixedPrecXtHX          = prm.get_bool("USE MIXED PREC XTHX");
        useMixedPrecSubspaceRotRR = prm.get_bool("USE MIXED PREC RR_SR");
        useMixedPrecCommunOnlyXtHXXtOX =
          prm.get_bool("USE MIXED PREC COMMUN ONLY XTOX XTHX");
        useSinglePrecCommunCheby = prm.get_bool("USE SINGLE PREC COMMUN CHEBY");
        useSinglePrecCheby       = prm.get_bool("USE SINGLE PREC CHEBY");
        tensorOpType             = prm.get("TENSOR OP TYPE SINGLE PREC CHEBY");
        overlapComputeCommunCheby =
          prm.get_bool("OVERLAP COMPUTE COMMUN CHEBY");
        overlapComputeCommunOrthoRR =
          prm.get_bool("OVERLAP COMPUTE COMMUN ORTHO RR");
        algoType                                       = prm.get("ALGO");
        chebyshevFilterPolyDegreeFirstScfScalingFactor = prm.get_double(
          "CHEBYSHEV POLYNOMIAL DEGREE SCALING FACTOR FIRST SCF");
        reuseLanczosUpperBoundFromFirstCall =
          prm.get_bool("REUSE LANCZOS UPPER BOUND");
        ;
        allowMultipleFilteringPassesAfterFirstScf =
          prm.get_bool("ALLOW MULTIPLE PASSES POST FIRST SCF");
        highestStateOfInterestForChebFiltering =
          prm.get_integer("HIGHEST STATE OF INTEREST FOR CHEBYSHEV FILTERING");
        useSubspaceProjectedSHEPGPU = prm.get_bool("SUBSPACE PROJ SHEP GPU");
        restrictToOnePass = prm.get_bool("RESTRICT TO SINGLE FILTER PASS");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();


    prm.enter_subsection("Poisson problem parameters");
    {
      maxLinearSolverIterations = prm.get_integer("MAXIMUM ITERATIONS");
      absLinearSolverTolerance  = prm.get_double("TOLERANCE");
      poissonGPU                = prm.get_bool("GPU MODE");
      vselfGPU                  = prm.get_bool("VSELF GPU MODE");
    }
    prm.leave_subsection();

    prm.enter_subsection("Helmholtz problem parameters");
    {
      maxLinearSolverIterationsHelmholtz =
        prm.get_integer("MAXIMUM ITERATIONS HELMHOLTZ");
      absLinearSolverToleranceHelmholtz =
        prm.get_double("ABSOLUTE TOLERANCE HELMHOLTZ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Molecular Dynamics");
    {
      atomicMassesFile            = prm.get("ATOMIC MASSES FILE");
      extrapolateDensity          = prm.get_integer("EXTRAPOLATE DENSITY");
      isBOMD                      = prm.get_bool("BOMD");
      maxJacobianRatioFactorForMD = prm.get_double("MAX JACOBIAN RATIO FACTOR");
      timeStepBOMD                = prm.get_double("TIME STEP");
      numberStepsBOMD             = prm.get_integer("NUMBER OF STEPS");
      MDTrack                     = prm.get_integer("TRACKING ATOMIC NO");
      startingTempBOMD            = prm.get_double("STARTING TEMPERATURE");
      thermostatTimeConstantBOMD  = prm.get_double("THERMOSTAT TIME CONSTANT");
      MaxWallTime                 = prm.get_double("MAX WALL TIME");



      tempControllerTypeBOMD = prm.get("TEMPERATURE CONTROLLER TYPE");
    }
    prm.leave_subsection();

    check_parameters(mpi_comm_parent);

    const bool printParametersToFile = false;
    if (printParametersToFile &&
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
      {
        prm.print_parameters(std::cout,
                             dealii::ParameterHandler::OutputStyle::LaTeX);
        exit(0);
      }

    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0 &&
        verbosity >= 1 && printParams)
      {
        prm.print_parameters(std::cout, dealii::ParameterHandler::ShortText);
      }

    //
    setAutoParameters(mpi_comm_parent);
  }



  void
  dftParameters::check_parameters(const MPI_Comm &mpi_comm_parent) const
  {
    AssertThrow(
      !((periodicX || periodicY || periodicZ) && writeLdosFile),
      dealii::ExcMessage(
        "DFT-FE Error: LOCAL DENSITY OF STATES is currently not implemented in the case of periodic and semi-periodic boundary conditions."));


    if (floatingNuclearCharges)
      AssertThrow(
        smearedNuclearCharges,
        dealii::ExcMessage(
          "DFT-FE Error: FLOATING NUCLEAR CHARGES can only be used if SMEARED NUCLEAR CHARGES is set to true."));
#ifdef USE_COMPLEX
    if (isIonForce || isCellStress)
      AssertThrow(
        !useSymm,
        dealii::ExcMessage(
          "DFT-FE Error: USE GROUP SYMMETRY must be set to false if either ION FORCE or CELL STRESS is set to true. This functionality will be added in a future release"));
    if (solverMode == "BANDS")
      AssertThrow(
        kPointDataFile != "",
        dealii::ExcMessage(
          "DFT-FE Error: kPOINT RULE FILE must be provided for bands."));
#endif
#ifndef USE_COMPLEX
    AssertThrow(
      nkx == 1 && nky == 1 && nkz == 1 && offsetFlagX == 0 &&
        offsetFlagY == 0 && offsetFlagZ == 0,
      dealii::ExcMessage(
        "DFT-FE Error: Real executable cannot be used for non-zero k point."));
    AssertThrow(solverMode != "BANDS",
                dealii::ExcMessage(
                  "DFT-FE Error: Real executable cannot be used for bands."));
#endif
    if (numberEigenValues != 0)
      AssertThrow(
        nbandGrps <= numberEigenValues,
        dealii::ExcMessage(
          "DFT-FE Error: NPBAND is greater than NUMBER OF KOHN-SHAM WAVEFUNCTIONS."));

    if (nonSelfConsistentForce)
      AssertThrow(
        false,
        dealii::ExcMessage(
          "DFT-FE Error: Implementation of this feature is not completed yet."));

    if (spinPolarized == 1 &&
        (extrapolateDensity >= 1 || reuseDensityGeoOpt == 2))
      AssertThrow(
        false,
        dealii::ExcMessage(
          "DFT-FE Error: Implementation of this feature is not completed yet."));

    AssertThrow(!coordinatesFile.empty(),
                dealii::ExcMessage(
                  "DFT-FE Error: ATOMIC COORDINATES FILE not given."));

    AssertThrow(!domainBoundingVectorsFile.empty(),
                dealii::ExcMessage(
                  "DFT-FE Error: DOMAIN VECTORS FILE not given."));

    if (solverMode == "NSCF" || solverMode == "BANDS")
      AssertThrow(
        loadQuadData == true,
        dealii::ExcMessage(
          "DFT-FE Error: Cant run NSCF/BANDS without load Quad data set to true"));

    if (isPseudopotential)
      AssertThrow(
        !pseudoPotentialFile.empty(),
        dealii::ExcMessage(
          "DFT-FE Error: PSEUDOPOTENTIAL FILE NAMES LIST not given."));

    if (spinPolarized == 0)
      AssertThrow(
        !constraintMagnetization,
        dealii::ExcMessage(
          "DFT-FE Error: This is a SPIN UNPOLARIZED calculation. Can't have CONSTRAINT MAGNETIZATION ON."));

    if (verbosity >= 1 &&
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
      if (pureState)
        std::cout
          << " WARNING: PURE STATE mode is ON. Integer occupations will be used no matter what temperature is provided at input"
          << std::endl;

    AssertThrow(
      natoms != 0,
      dealii::ExcMessage(
        "DFT-FE Error: Number of atoms not specified or given a value of zero, which is not allowed."));

    AssertThrow(
      natomTypes != 0,
      dealii::ExcMessage(
        "DFT-FE Error: Number of atom types not specified or given a value of zero, which is not allowed."));

    if (nbandGrps > 1)
      AssertThrow(
        wfcBlockSize == chebyWfcBlockSize,
        dealii::ExcMessage(
          "DFT-FE Error: WFC BLOCK SIZE and CHEBY WFC BLOCK SIZE must be same for band parallelization."));
    if (applyExternalPotential)
      AssertThrow(
        ((periodicZ == false && applyExternalPotential == true)),
        dealii::ExcMessage(
          "DFT-FE Error: External electrostatic potential bias can only when the system is non-periodic alpong Z axis."));

    if ((zCoordinateR - zCoordinateL) <= 1.0e-3)
      AssertThrow(
        false,
        dealii::ExcMessage(
          "DFT-FE Error: The location of CONSTRAINED POTENTIAL surfaces are too close."));
    if (XCType.substr(0, 4) == "MGGA")
      {
        AssertThrow(
          !isCellStress,
          dealii::ExcMessage(
            "DFT-FE Error: Computation of CELL STRESS with MGGA functional is not completed yet."));
        if (!floatingNuclearCharges)
          AssertThrow(
            !isIonForce,
            dealii::ExcMessage(
              "DFT-FE Error: Computation of ION FORCE with MGGA functional in all-electron calculation is not completed yet."));
        AssertThrow(
          !(mixingMethod == "LOW_RANK_DIELECM_PRECOND" ||
            mixingMethod == "ANDERSON_WITH_KERKER" ||
            mixingMethod == "ANDERSON_WITH_RESTA"),
          dealii::ExcMessage(
            "DFT-FE Error: ANDERSON_WITH_RESTA or ANDERSON_WITH_KERKER or LRDM mixing scheme in MGGA functional is not completed yet."));
      }

    bool isHubbard = (XCType.substr(XCType.size() - 2) == "+U");
    if (isHubbard)
      AssertThrow(
        !(mixingMethod == "ANDERSON_WITH_KERKER" ||
          mixingMethod == "ANDERSON_WITH_RESTA"),
        dealii::ExcMessage(
          "DFT-FE Error: ANDERSON_WITH_RESTA or ANDERSON_WITH_KERKER for Hubbard is not completed yet."));

    if (dc_dispersioncorrectiontype == 1 || dc_dispersioncorrectiontype == 2)
      {
        bool customParameters = !(dc_dampingParameterFilename == "");
        if (!customParameters)
          {
            if (XCType == "GGA-PBE")
              {
                if (dc_dispersioncorrectiontype == 1)
                  AssertThrow(
                    dc_d3dampingtype != 4,
                    dealii::ExcMessage(std::string(
                      "The OP damping functions has not been parametrized for this functional.")));
              }
            else if (XCType == "GGA-RPBE")
              {
                if (dc_dispersioncorrectiontype == 1)
                  AssertThrow(
                    dc_d3dampingtype == 0 || dc_d3dampingtype == 1,
                    dealii::ExcMessage(std::string(
                      "The OP, BJM and ZEROM damping functions have not been parametrized for this functional.")));
              }
            else if (XCType == "MGGA-R2SCAN")
              {
                if (dc_dispersioncorrectiontype == 1)
                  AssertThrow(
                    dc_d3dampingtype == 1,
                    dealii::ExcMessage(std::string(
                      "Only BJ damping function has been parametrized for this functional.")));

                if (dc_dispersioncorrectiontype == 2)
                  AssertThrow(
                    !dc_d4MBD,
                    dealii::ExcMessage(std::string(
                      "D4 MBD has not been parametrized for this functional.")));
              }

            else if (XCType == "MGGA-SCAN")
              {
                if (dc_dispersioncorrectiontype == 1)
                  AssertThrow(
                    dc_d3dampingtype == 0 || dc_d3dampingtype == 1,
                    dealii::ExcMessage(std::string(
                      "Only ZERO and BJ damping functions have been parametrized for this functional.")));
              }
            else
              {
                AssertThrow(
                  false,
                  dealii::ExcMessage(std::string(
                    "DFTD3/4 have not been parametrized for this functional.")));
              }
          }
      }
  }


  void
  dftParameters::setAutoParameters(const MPI_Comm &mpi_comm_parent)
  {
    //
    // Automated choice of mesh related parameters
    //

    if (isBOMD)
      isIonForce = true;

    if (solverMode == "NEB" || solverMode == "MD")
      isIonForce = true;

    if (!isPseudopotential)
      {
        if (!reproducible_output)
          smearedNuclearCharges = false;
        floatingNuclearCharges = false;
      }

    if (meshSizeOuterDomain < 1.0e-6)
      if (periodicX || periodicY || periodicZ)
        meshSizeOuterDomain = 4.0;
      else
        meshSizeOuterDomain = 13.0;

    if (meshSizeInnerBall < 1.0e-6)
      if (isPseudopotential)
        meshSizeInnerBall = 10.0 * meshSizeOuterBall;
      else
        meshSizeInnerBall = 0.1 * meshSizeOuterBall;

    if (outerAtomBallRadius < 1.0e-6)
      {
        if (isPseudopotential)
          {
            if (!floatingNuclearCharges)
              outerAtomBallRadius = 2.5;
            else
              {
                if (!(periodicX || periodicY || periodicZ))
                  outerAtomBallRadius = 6.0;
                else
                  outerAtomBallRadius = 10.0;
              }
          }
        else
          outerAtomBallRadius = 2.0;
      }

#ifdef DFTFE_WITH_CUSTOMIZED_DEALII
    if (!(periodicX || periodicY || periodicZ) && !reproducible_output)
      {
        constraintsParallelCheck              = false;
        createConstraintsFromSerialDofhandler = false;
      }
    else if (reproducible_output)
      createConstraintsFromSerialDofhandler = true;
#else
    createConstraintsFromSerialDofhandler = false;
#endif

    if (reproducible_output)
      {
        gaussianOrderMoveMeshToAtoms = 4.0;
      }

    //
    // Automated choice of eigensolver parameters
    //
    if (isPseudopotential && orthogType == "Auto")
      {
        if (verbosity >= 1 &&
            dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=CGS for pseudopotential calculations "
            << std::endl;
        orthogType = "CGS";
      }
    else if (!isPseudopotential && orthogType == "Auto" && !useDevice)
      {
#ifdef USE_PETSC;
        if (verbosity >= 1 &&
            dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=GS for all-electron calculations as DFT-FE is linked to dealii with Petsc and Slepc"
            << std::endl;

        orthogType = "GS";
#else
        if (verbosity >= 1 &&
            dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=CGS for all-electron calculations as DFT-FE is not linked to dealii with Petsc and Slepc "
            << std::endl;

        orthogType = "CGS";
#endif
      }
    else if (orthogType == "GS" && !useDevice)
      {
#ifndef USE_PETSC;
        AssertThrow(
          orthogType != "GS",
          dealii::ExcMessage(
            "DFT-FE Error: Please use ORTHOGONALIZATION TYPE to be CGS/Auto as GS option is only available if DFT-FE is linked to dealii with Petsc and Slepc."));
#endif
      }
    else if (!isPseudopotential && orthogType == "Auto" && useDevice)
      {
        if (verbosity >= 1 &&
            dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          std::cout
            << "Setting ORTHOGONALIZATION TYPE=CGS for all-electron calculations on GPUs "
            << std::endl;
        orthogType = "CGS";
      }
    else if (orthogType == "GS" && useDevice)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: GS is not implemented on GPUs. Use Auto option."));
      }


    if (algoType == "FAST")
      {
        useMixedPrecXtOX                    = true;
        useMixedPrecXtHX                    = true;
        useMixedPrecCGS_SR                  = true;
        useSinglePrecCommunCheby            = true;
        reuseLanczosUpperBoundFromFirstCall = true;
      }

#ifdef DFTFE_WITH_DEVICE
    if (!isPseudopotential && useDevice)
      {
        overlapComputeCommunCheby = false;
      }
#endif


#ifndef DFTFE_WITH_DEVICE
    useDevice           = false;
    useELPADeviceKernel = false;
#endif

    if (scalapackBlockSize == 0)
      {
        if (useELPADeviceKernel)
          scalapackBlockSize = 16;
        else
          scalapackBlockSize = 32;
      }

#if !defined(DFTFE_WITH_CUDA_NCCL) && !defined(DFTFE_WITH_HIP_RCCL) && \
  !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
    useDeviceDirectAllReduce = false;
#endif
#if !defined(DFTFE_WITH_CUDA_NCCL) && !defined(DFTFE_WITH_HIP_RCCL)
    useDCCL = false;
#endif
#if !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
    useDeviceDirectAllReduce = useDCCL && useDeviceDirectAllReduce;
#endif

    if (verbosity >= 5)
      computeEnergyEverySCF = true;

    if (std::fabs(chebyshevTolerance - 0.0) < 1.0e-20)
      {
        if (restrictToOnePass)
          chebyshevTolerance = 1.0e+4;
        else if (mixingMethod == "LOW_RANK_DIELECM_PRECOND")
          chebyshevTolerance = 2.0e-3;
        else if (mixingMethod == "ANDERSON_WITH_KERKER" ||
                 mixingMethod == "ANDERSON_WITH_RESTA")
          chebyshevTolerance = 1.0e-2;
        else if (solverMode != "NSCF" && solverMode != "BANDS")
          chebyshevTolerance = 5.0e-2;
      }

    if (std::fabs(mixingParameter - 0.0) < 1.0e-12)
      {
        if (mixingMethod == "LOW_RANK_DIELECM_PRECOND")
          mixingParameter = 0.5;
        else if (mixingMethod == "ANDERSON_WITH_KERKER" ||
                 mixingMethod == "ANDERSON_WITH_RESTA")
          mixingParameter = 0.5;
        else
          mixingParameter = 0.2;
      }

    if (std::fabs(netCharge - 0.0) > 1.0e-12 &&
        !(periodicX || periodicY || periodicZ))
      {
        multipoleBoundaryConditions = true;
      }
    if (reproducible_output)
      {
        spinMixingEnhancementFactor = 1.0;
      }

    if (densityQuadratureRule == 0)
      {
        densityQuadratureRule = XCType.substr(0, 4) == "MGGA" ?
                                  16 :
                                  finiteElementPolynomialOrderRhoNodal + 1;
      }


    // checking if the XC type is compatible with
    // overlap compute communication cheby

    bool isHubbard = (XCType.substr(XCType.size() - 2) == "+U");
    bool isLocalXC = (XCType.substr(0, 3) == "LDA") ||
                     (XCType.substr(0, 3) == "GGA") ||
                     ((XCType.substr(0, 4) == "MGGA"));
    if (isHubbard || !isLocalXC)
      {
        overlapComputeCommunCheby = false;
        if (verbosity >= 1 &&
            dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)
          {
            std::cout
              << "DFT-FE Warning: Hubbard cannot be used with OVERLAP COMPUTE COMMUN CHEBY = true. Setting OVERLAP COMPUTE COMMUN CHEBY to false"
              << std::endl;
          }
      }
    potentialValueR = potentialValueL + appliedPotentialDifference;
  }


} // namespace dftfe
