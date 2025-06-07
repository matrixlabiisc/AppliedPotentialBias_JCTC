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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

// Include header files
#include <dft.h>
#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <dealiiLinearSolver.h>
#include <densityFirstOrderResponseCalculator.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <energyCalculator.h>
#include <fileReaders.h>
#include <force.h>
#include <linalg.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <meshMovementAffineTransform.h>
#include <meshMovementGaussian.h>
#include <poissonSolverProblem.h>
#include <pseudoConverter.h>
#include <pseudoUtils.h>
#include <symmetry.h>
#include <vectorUtilities.h>
#include <MemoryTransfer.h>
#include <QuadDataCompositeWrite.h>
#include <MPIWriteOnFile.h>
#include <functionalTest.h>
#include <computeAuxProjectedDensityMatrixFromPSI.h>

#include <algorithm>
#include <cmath>
#include <complex>
// #include <stdafx.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>

#include <spglib.h>
#include <stdafx.h>
#include <sys/stat.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <chrono>
#include <sys/time.h>
#include <ctime>

#ifdef DFTFE_WITH_DEVICE
#  include <linearAlgebraOperationsDevice.h>
#endif

#include <elpa/elpa.h>
#include "AuxDensityMatrixFE.h"


#include "hubbardClass.h"
#include "ExcDFTPlusU.h"
namespace dftfe
{
  //
  // dft constructor
  //
  template <dftfe::utils::MemorySpace memorySpace>
  dftClass<memorySpace>::dftClass(const MPI_Comm    &mpi_comm_parent,
                                  const MPI_Comm    &mpi_comm_domain,
                                  const MPI_Comm    &_interpoolcomm,
                                  const MPI_Comm    &_interBandGroupComm,
                                  const std::string &scratchFolderName,
                                  dftParameters     &dftParams)
    : FE(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(
           dftParams.finiteElementPolynomialOrder + 1)),
         1)
    ,
#ifdef USE_COMPLEX
    FEEigen(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(
              dftParams.finiteElementPolynomialOrder + 1)),
            2)
    ,
#else
    FEEigen(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(
              dftParams.finiteElementPolynomialOrder + 1)),
            1)
    ,
#endif
    mpi_communicator(mpi_comm_domain)
    , d_mpiCommParent(mpi_comm_parent)
    , interpoolcomm(_interpoolcomm)
    , interBandGroupComm(_interBandGroupComm)
    , d_dftfeScratchFolderName(scratchFolderName)
    , d_dftParamsPtr(&dftParams)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , numElectrons(0)
    , numLevels(0)
    , d_autoMesh(1)
    , d_mesh(mpi_comm_parent,
             mpi_comm_domain,
             _interpoolcomm,
             _interBandGroupComm,
             dftParams.finiteElementPolynomialOrder,
             dftParams)
    , d_affineTransformMesh(mpi_comm_parent, mpi_comm_domain, dftParams)
    , d_gaussianMovePar(mpi_comm_parent, mpi_comm_domain, dftParams)
    , d_vselfBinsManager(mpi_comm_parent,
                         mpi_comm_domain,
                         _interpoolcomm,
                         dftParams)
    , d_dispersionCorr(mpi_comm_parent,
                       mpi_comm_domain,
                       _interpoolcomm,
                       _interBandGroupComm,
                       dftParams)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0) &&
              dftParams.verbosity >= 0)
    , d_kohnShamDFTOperatorsInitialized(false)
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParams.reproducible_output || dftParams.verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::every_call_and_summary,
                      dealii::TimerOutput::wall_times)
    , computingTimerStandard(mpi_comm_domain,
                             pcout,
                             dftParams.reproducible_output ||
                                 dftParams.verbosity < 1 ?
                               dealii::TimerOutput::never :
                               dealii::TimerOutput::every_call_and_summary,
                             dealii::TimerOutput::wall_times)
    , d_subspaceIterationSolver(mpi_comm_parent,
                                mpi_comm_domain,
                                0.0,
                                0.0,
                                0.0,
                                dftParams)
#ifdef DFTFE_WITH_DEVICE
    , d_subspaceIterationSolverDevice(mpi_comm_parent,
                                      mpi_comm_domain,
                                      0.0,
                                      0.0,
                                      0.0,
                                      dftParams)
    , d_phiTotalSolverProblemDevice(
        dftParams.finiteElementPolynomialOrderElectrostatics,
        mpi_comm_domain)
    , d_phiPrimeSolverProblemDevice(
        dftParams.finiteElementPolynomialOrderElectrostatics,
        mpi_comm_domain)
#endif
    , d_phiTotalSolverProblem(
        dftParams.finiteElementPolynomialOrderElectrostatics,
        mpi_comm_domain)
    , d_phiPrimeSolverProblem(
        dftParams.finiteElementPolynomialOrderElectrostatics,
        mpi_comm_domain)
    , d_mixingScheme(mpi_comm_parent, mpi_comm_domain, dftParams.verbosity)
  {
    d_nOMPThreads = 1;
    d_useHubbard  = false;
#ifdef _OPENMP
    if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
      {
        try
          {
            d_nOMPThreads = std::stoi(std::string(penv));
          }
        catch (...)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                std::string(
                  "When specifying the <DFTFE_NUM_THREADS> environment "
                  "variable, it needs to be something that can be interpreted "
                  "as an integer. The text you have in the environment "
                  "variable is <") +
                penv + ">"));
          }

        AssertThrow(d_nOMPThreads > 0,
                    dealii::ExcMessage(
                      "When specifying the <DFTFE_NUM_THREADS> environment "
                      "variable, it needs to be a positive number."));
      }
#endif
    if (d_dftParamsPtr->verbosity > 0)
      pcout << "Threads per MPI task: " << d_nOMPThreads << std::endl;

    d_elpaScala = new dftfe::elpaScalaManager(mpi_comm_domain);

    forcePtr    = new forceClass<memorySpace>(this,
                                           mpi_comm_parent,
                                           mpi_comm_domain,
                                           dftParams);
    symmetryPtr = new symmetryClass<memorySpace>(this,
                                                 mpi_comm_parent,
                                                 mpi_comm_domain,
                                                 _interpoolcomm);

    d_excManagerPtr = std::make_shared<excManager<memorySpace>>();
    d_isRestartGroundStateCalcFromChk = false;

#if defined(DFTFE_WITH_DEVICE)
    d_devicecclMpiCommDomainPtr = new utils::DeviceCCLWrapper;
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)

      d_devicecclMpiCommDomainPtr->init(mpi_comm_domain,
                                        d_dftParamsPtr->useDCCL);
#endif
    d_pspCutOff =
      d_dftParamsPtr->reproducible_output ?
        30.0 :
        (std::max(d_dftParamsPtr->pspCutoffImageCharges, d_pspCutOffTrunc));

    d_smearedChargeMoments.resize(13, 0.0);
    std::fill(d_smearedChargeMoments.begin(),
              d_smearedChargeMoments.end(),
              0.0);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::runFunctionalTest()
  {
    if (d_dftParamsPtr->functionalTestName == "TestDataTransfer")
      {
        functionalTest::testTransferFromParentToChildIncompatiblePartitioning(
          d_BLASWrapperPtrHost,
          d_mpiCommParent,
          mpi_communicator,
          interpoolcomm,
          interBandGroupComm,
          d_dftParamsPtr->finiteElementPolynomialOrder,
          *d_dftParamsPtr,
          atomLocations,
          d_imagePositionsAutoMesh,
          d_imageIds,
          d_nearestAtomDistances,
          d_domainBoundingVectors,
          false,  // bool generateSerialTria,
          false); // bool generateElectrostaticsTria)
      }
    else if (d_dftParamsPtr->functionalTestName ==
             "TestMultiVectorPoissonSolver")
      {
        functionalTest::testMultiVectorPoissonSolver(
          d_basisOperationsPtrElectroHost,
          d_matrixFreeDataPRefined,
          d_BLASWrapperPtrHost,
          d_constraintsVectorElectro,
          d_densityInQuadValues[0],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          d_dftParamsPtr->verbosity,
          d_mpiCommParent,
          mpi_communicator);
      }
  }
  template <dftfe::utils::MemorySpace memorySpace>
  dftClass<memorySpace>::~dftClass()
  {
    finalizeKohnShamDFTOperator();
    delete symmetryPtr;
    matrix_free_data.clear();
    delete forcePtr;
#if defined(DFTFE_WITH_DEVICE)
    delete d_devicecclMpiCommDomainPtr;
#endif

    d_elpaScala->elpaDeallocateHandles(*d_dftParamsPtr);
    delete d_elpaScala;
  }

  namespace internaldft
  {
    void
    convertToCellCenteredCartesianCoordinates(
      std::vector<std::vector<double>>       &atomLocations,
      const std::vector<std::vector<double>> &latticeVectors)
    {
      std::vector<double> cartX(atomLocations.size(), 0.0);
      std::vector<double> cartY(atomLocations.size(), 0.0);
      std::vector<double> cartZ(atomLocations.size(), 0.0);

      //
      // convert fractional atomic coordinates to cartesian coordinates
      //
      for (dftfe::Int i = 0; i < atomLocations.size(); ++i)
        {
          cartX[i] = atomLocations[i][2] * latticeVectors[0][0] +
                     atomLocations[i][3] * latticeVectors[1][0] +
                     atomLocations[i][4] * latticeVectors[2][0];
          cartY[i] = atomLocations[i][2] * latticeVectors[0][1] +
                     atomLocations[i][3] * latticeVectors[1][1] +
                     atomLocations[i][4] * latticeVectors[2][1];
          cartZ[i] = atomLocations[i][2] * latticeVectors[0][2] +
                     atomLocations[i][3] * latticeVectors[1][2] +
                     atomLocations[i][4] * latticeVectors[2][2];
        }

      //
      // define cell centroid (confirm whether it will work for non-orthogonal
      // lattice vectors)
      //
      double cellCentroidX =
        0.5 *
        (latticeVectors[0][0] + latticeVectors[1][0] + latticeVectors[2][0]);
      double cellCentroidY =
        0.5 *
        (latticeVectors[0][1] + latticeVectors[1][1] + latticeVectors[2][1]);
      double cellCentroidZ =
        0.5 *
        (latticeVectors[0][2] + latticeVectors[1][2] + latticeVectors[2][2]);

      for (dftfe::Int i = 0; i < atomLocations.size(); ++i)
        {
          atomLocations[i][2] = cartX[i] - cellCentroidX;
          atomLocations[i][3] = cartY[i] - cellCentroidY;
          atomLocations[i][4] = cartZ[i] - cellCentroidZ;
        }
    }
  } // namespace internaldft

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::computeVolume(const dealii::DoFHandler<3> &_dofHandler)
  {
    double                       domainVolume = 0;
    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(_dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_JxW_values);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = _dofHandler.begin_active(),
      endc = _dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          for (dftfe::uInt q_point = 0; q_point < quadrature.size(); ++q_point)
            domainVolume += fe_values.JxW(q_point);
        }

    domainVolume = dealii::Utilities::MPI::sum(domainVolume, mpi_communicator);
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Volume of the domain (Bohr^3): " << domainVolume << std::endl;
    return domainVolume;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::set()
  {
    computingTimerStandard.enter_subsection("Atomic system initialization");

    d_numEigenValues = d_dftParamsPtr->numberEigenValues;

    //
    // read coordinates
    //
    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        //
        // read fractionalCoordinates of atoms in periodic case
        //
        dftUtils::readFile(atomLocations, d_dftParamsPtr->coordinatesFile);
        AssertThrow(
          d_dftParamsPtr->natoms == atomLocations.size(),
          dealii::ExcMessage(
            "DFT-FE Error: The number atoms"
            "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
            "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
            "blank row at the end can cause this issue too."));
        pcout << "number of atoms: " << atomLocations.size() << "\n";
        atomLocationsFractional.resize(atomLocations.size());
        //
        // find unique atom types
        //
        for (std::vector<std::vector<double>>::iterator it =
               atomLocations.begin();
             it < atomLocations.end();
             it++)
          {
            atomTypes.insert((dftfe::uInt)((*it)[0]));
            d_atomTypeAtributes[(dftfe::uInt)((*it)[0])] =
              (dftfe::uInt)((*it)[1]);

            if (!d_dftParamsPtr->isPseudopotential)
              AssertThrow(
                (*it)[0] <= 50,
                dealii::ExcMessage(
                  "DFT-FE Error: One of the atomic numbers exceeds 50."
                  "Currently, for all-electron calculations we have single atom wavefunction and electron-density"
                  "initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
                  "added in the next release. In the mean time, you could also contact the developers of DFT-FE, who can provide"
                  "you the data for the single atom wavefunction and electron-density data for"
                  "atomic numbers beyond 50."));
          }

        //
        // print fractional coordinates
        //
        for (dftfe::Int i = 0; i < atomLocations.size(); ++i)
          {
            atomLocationsFractional[i] = atomLocations[i];
          }
      }
    else
      {
        dftUtils::readFile(atomLocations, d_dftParamsPtr->coordinatesFile);

        AssertThrow(
          d_dftParamsPtr->natoms == atomLocations.size(),
          dealii::ExcMessage(
            "DFT-FE Error: The number atoms"
            "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
            "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
            "blank row at the end can cause this issue too."));
        pcout << "number of atoms: " << atomLocations.size() << "\n";

        //
        // find unique atom types
        //
        for (std::vector<std::vector<double>>::iterator it =
               atomLocations.begin();
             it < atomLocations.end();
             it++)
          {
            atomTypes.insert((dftfe::uInt)((*it)[0]));
            d_atomTypeAtributes[(dftfe::uInt)((*it)[0])] =
              (dftfe::uInt)((*it)[1]);

            if (!d_dftParamsPtr->isPseudopotential)
              AssertThrow(
                (*it)[0] <= 50,
                dealii::ExcMessage(
                  "DFT-FE Error: One of the atomic numbers exceeds 50."
                  "Currently, for all-electron calculations we have single atom wavefunction and electron-density"
                  "initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
                  "added in the next release. You could also contact the developers of DFT-FE, who can provide"
                  "you with the code to generate the single atom wavefunction and electron-density data for"
                  "atomic numbers beyond 50."));
          }
      }

    //
    // read Gaussian atomic displacements
    //
    std::vector<std::vector<double>> atomsDisplacementsGaussian;
    d_atomsDisplacementsGaussianRead.resize(atomLocations.size(),
                                            dealii::Tensor<1, 3, double>());
    d_gaussianMovementAtomsNetDisplacements.resize(
      atomLocations.size(), dealii::Tensor<1, 3, double>());
    if (d_dftParamsPtr->coordinatesGaussianDispFile != "")
      {
        dftUtils::readFile(3,
                           atomsDisplacementsGaussian,
                           d_dftParamsPtr->coordinatesGaussianDispFile);

        for (dftfe::Int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
          for (dftfe::Int j = 0; j < 3; ++j)
            d_atomsDisplacementsGaussianRead[i][j] =
              atomsDisplacementsGaussian[i][j];

        d_isAtomsGaussianDisplacementsReadFromFile = true;
      }

    //
    // read domain bounding Vectors
    //
    dftfe::uInt numberColumnsLatticeVectorsFile = 3;
    dftUtils::readFile(numberColumnsLatticeVectorsFile,
                       d_domainBoundingVectors,
                       d_dftParamsPtr->domainBoundingVectorsFile);

    AssertThrow(
      d_domainBoundingVectors.size() == 3,
      dealii::ExcMessage(
        "DFT-FE Error: The number of domain bounding"
        "vectors read from input file (input through DOMAIN VECTORS FILE) should be 3. Please check"
        "your domain vectors file. Sometimes an extra blank row at the end can cause this issue too."));

    //
    // evaluate cross product of
    //
    std::vector<double> cross;
    dftUtils::cross_product(d_domainBoundingVectors[0],
                            d_domainBoundingVectors[1],
                            cross);

    double scalarConst = d_domainBoundingVectors[2][0] * cross[0] +
                         d_domainBoundingVectors[2][1] * cross[1] +
                         d_domainBoundingVectors[2][2] * cross[2];
    AssertThrow(
      scalarConst > 0,
      dealii::ExcMessage(
        "DFT-FE Error: Domain bounding vectors or lattice vectors read from"
        "input file (input through DOMAIN VECTORS FILE) should form a right-handed coordinate system."
        "Please check your domain vectors file. This is usually fixed by changing the order of the"
        "vectors in the domain vectors file."));

    pcout << "number of atoms types: " << atomTypes.size() << "\n";

    //
    // determine number of electrons
    //
    for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        const dftfe::uInt Z        = atomLocations[iAtom][0];
        const dftfe::uInt valenceZ = atomLocations[iAtom][1];

        if (d_dftParamsPtr->isPseudopotential)
          numElectrons += valenceZ;
        else
          numElectrons += Z;
      }

    numElectrons = numElectrons + d_dftParamsPtr->netCharge;
    if (d_dftParamsPtr->verbosity >= 1 and
        std::abs(d_dftParamsPtr->netCharge) > 1e-12)
      pcout << "Setting netcharge " << d_dftParamsPtr->netCharge << std::endl;
    if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
      d_dftParamsPtr->highestStateOfInterestForChebFiltering =
        std::floor(numElectrons * 1.05 / 2.0);
    if (d_dftParamsPtr->solverMode == "NSCF" ||
        d_dftParamsPtr->solverMode == "BANDS")
      {
        d_numEigenValues = std::max(
          static_cast<double>(d_dftParamsPtr->numberEigenValues),
          std::max(d_dftParamsPtr->highestStateOfInterestForChebFiltering * 1.1,
                   d_dftParamsPtr->highestStateOfInterestForChebFiltering +
                     20.0));
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Setting the number of Kohn-Sham wave functions to be 10 percent more than the HIGHEST STATE OF INTEREST FOR CHEBYSHEV FILTERING "
              << d_numEigenValues << std::endl;
          }
      }
    else if (d_dftParamsPtr->numberEigenValues <= numElectrons / 2.0 ||
             d_dftParamsPtr->numberEigenValues == 0)
      {
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Warning: User has requested the number of Kohn-Sham wavefunctions to be less than or"
                 "equal to half the number of electrons in the system. Setting the Kohn-Sham wavefunctions"
                 "to half the number of electrons with a 20 percent buffer to avoid convergence issues in"
                 "SCF iterations"
              << std::endl;
          }
        d_numEigenValues =
          (numElectrons / 2.0) +
          std::max((d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND" ?
                      0.22 :
                      0.2) *
                     (numElectrons / 2.0),
                   20.0);

        // start with 17-20% buffer to leave room for additional modifications
        // due to block size restrictions
#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice && d_dftParamsPtr->autoDeviceBlockSizes)
          d_numEigenValues =
            (numElectrons / 2.0) + std::max((d_dftParamsPtr->mixingMethod ==
                                                 "LOW_RANK_DIELECM_PRECOND" ?
                                               0.2 :
                                               0.17) *
                                              (numElectrons / 2.0),
                                            20.0);
#endif

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << " Setting the number of Kohn-Sham wave functions to be "
                  << d_numEigenValues << std::endl;
          }
      }

    if (d_dftParamsPtr->algoType == "FAST")
      {
        if (d_dftParamsPtr->TVal < 1000)
          {
            d_dftParamsPtr->numCoreWfcForMixedPrecRR = 0.8 * numElectrons / 2.0;
            pcout << " Setting MIXED PREC CORE EIGENSTATES to be "
                  << d_dftParamsPtr->numCoreWfcForMixedPrecRR
                  << std::endl; //@Kartick check if this is okay
          }
      }

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice && d_dftParamsPtr->autoDeviceBlockSizes)
      {
        const dftfe::uInt numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


        d_numEigenValues =
          std::ceil(d_numEigenValues / (numberBandGroups * 1.0)) *
          numberBandGroups;

        AssertThrow(
          (d_numEigenValues % numberBandGroups == 0 ||
           d_numEigenValues / numberBandGroups == 0),
          dealii::ExcMessage(
            "DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for Device run."));

        const dftfe::uInt bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, d_numEigenValues, bandGroupLowHighPlusOneIndices);

        const dftfe::uInt eigenvaluesInBandGroup =
          bandGroupLowHighPlusOneIndices[1];

        if (eigenvaluesInBandGroup <= 100)
          {
            d_dftParamsPtr->chebyWfcBlockSize = eigenvaluesInBandGroup;
            d_dftParamsPtr->wfcBlockSize      = eigenvaluesInBandGroup;
          }
        else if (eigenvaluesInBandGroup <= 600)
          {
            std::vector<dftfe::Int> temp1(4, 0);
            std::vector<dftfe::Int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 90.0) * 90.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 100.0) * 100.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 110.0) * 110.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 120.0) * 120.0 *
                       numberBandGroups;

            temp2[0] = 90;
            temp2[1] = 100;
            temp2[2] = 110;
            temp2[3] = 120;

            dftfe::Int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            dftfe::Int minElement =
              *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex];
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else if (eigenvaluesInBandGroup <= 1000)
          {
            std::vector<dftfe::Int> temp1(4, 0);
            std::vector<dftfe::Int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 150.0) * 150.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 160.0) * 160.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 170.0) * 170.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 180.0) * 180.0 *
                       numberBandGroups;

            temp2[0] = 150;
            temp2[1] = 160;
            temp2[2] = 170;
            temp2[3] = 180;

            dftfe::Int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            dftfe::Int minElement =
              *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else if (eigenvaluesInBandGroup <= 2000)
          {
            std::vector<dftfe::Int> temp1(4, 0);
            std::vector<dftfe::Int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 200.0) * 200.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 220.0) * 220.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 240.0) * 240.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 260.0) * 260.0 *
                       numberBandGroups;

            temp2[0] = 200;
            temp2[1] = 220;
            temp2[2] = 240;
            temp2[3] = 260;

            dftfe::Int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            dftfe::Int minElement =
              *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else
          {
            std::vector<dftfe::Int> temp1(4, 0);
            std::vector<dftfe::Int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 360.0) * 360.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 380.0) * 380.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 400.0) * 400.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 440.0) * 440.0 *
                       numberBandGroups;

            temp2[0] = 360;
            temp2[1] = 380;
            temp2[2] = 400;
            temp2[3] = 440;

            dftfe::Int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            dftfe::Int minElement =
              *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }

        if (d_dftParamsPtr->algoType == "FAST")
          d_dftParamsPtr->numCoreWfcForMixedPrecRR =
            std::floor(d_dftParamsPtr->numCoreWfcForMixedPrecRR /
                       d_dftParamsPtr->wfcBlockSize) *
            d_dftParamsPtr
              ->numCoreWfcForMixedPrecRR; //@Kartick check if this is okay

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Setting the number of Kohn-Sham wave functions for Device run to be: "
              << d_numEigenValues << std::endl;
            pcout << " Setting CHEBY WFC BLOCK SIZE for Device run to be "
                  << d_dftParamsPtr->chebyWfcBlockSize << std::endl;
            pcout << " Setting WFC BLOCK SIZE for Device run to be "
                  << d_dftParamsPtr->wfcBlockSize << std::endl;
            if (d_dftParamsPtr->algoType == "FAST")
              pcout
                << " Setting CORE EIGENSTATES for MIXED PRECISION STRATEGY on  Device run to be "
                << d_dftParamsPtr->numCoreWfcForMixedPrecRR << std::endl;
          }
      }
#endif
    if (d_dftParamsPtr->constraintMagnetization)
      {
        //
        const double netMagnetization =
          static_cast<double>(numElectrons) * d_dftParamsPtr->tot_magnetization;

        numElectronsUp =
          0.5 * (static_cast<double>(numElectrons) + netMagnetization);
        numElectronsDown =
          0.5 * (static_cast<double>(numElectrons) - netMagnetization);

        //
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << " Number of spin up electrons " << numElectronsUp
                  << std::endl;
            pcout << " Number of spin down electrons " << numElectronsDown
                  << std::endl;
          }
      }
    // convert pseudopotential files in upf format to dftfe format
    if (d_dftParamsPtr->verbosity >= 1)
      {
        pcout
          << std::endl
          << "Reading Pseudo-potential data for each atom from the list given in : "
          << d_dftParamsPtr->pseudoPotentialFile << std::endl;
      }

    dftfe::Int              nlccFlag = 0;
    std::vector<dftfe::Int> pspFlags(2, 0);
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0 &&
        d_dftParamsPtr->isPseudopotential == true)
      pspFlags = pseudoUtils::convert(d_dftParamsPtr->pseudoPotentialFile,
                                      d_dftfeScratchFolderName,
                                      d_dftParamsPtr->verbosity,
                                      d_dftParamsPtr->natomTypes,
                                      d_dftParamsPtr->pseudoTestsFlag);

    nlccFlag = pspFlags[0];
    nlccFlag = dealii::Utilities::MPI::sum(nlccFlag, d_mpiCommParent);
    if (nlccFlag > 0 && d_dftParamsPtr->isPseudopotential == true)
      d_dftParamsPtr->nonLinearCoreCorrection = true;
    // estimate total number of wave functions from atomic orbital filling
    if (d_dftParamsPtr->startingWFCType == "ATOMIC")
      determineOrbitalFilling();

    AssertThrow(
      d_dftParamsPtr->numCoreWfcForMixedPrecRR <= d_numEigenValues,
      dealii::ExcMessage(
        "DFT-FE Error: Incorrect input value used- CORE EIGENSTATES should be less than the total number of wavefunctions."));

#ifdef USE_COMPLEX
    if (d_dftParamsPtr->solverMode == "BANDS")
      readkPointData();
    else
      generateMPGrid();
#else
    d_kPointCoordinates.resize(3, 0.0);
    d_kPointWeights.resize(1, 1.0);
#endif

    // set size of eigenvalues and eigenvectors data structures
    eigenValues.resize(d_kPointWeights.size());

    if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      d_densityMatDerFermiEnergy.resize((d_dftParamsPtr->spinPolarized + 1) *
                                        d_kPointWeights.size());

    a0.clear();
    bLow.clear();

    a0.resize((d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(),
              0.0);
    bLow.resize((d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(),
                0.0);

    d_upperBoundUnwantedSpectrumValues.clear();
    d_upperBoundUnwantedSpectrumValues.resize(
      (d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(), 0.0);


    for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        eigenValues[kPoint].resize((d_dftParamsPtr->spinPolarized + 1) *
                                   d_numEigenValues);
      }

    d_partialOccupancies = eigenValues;

    if (d_dftParamsPtr->isPseudopotential == true)
      {
        // pcout<<"dft.cc 827 ONCV Number of cells DEBUG:
        // "<<basisOperationsPtrHost->nCells()<<std::endl;
        d_oncvClassPtr =
          std::make_shared<dftfe::oncvClass<dataTypes::number, memorySpace>>(
            mpi_communicator, // domain decomposition communicator
            d_dftfeScratchFolderName,
            atomTypes,
            d_dftParamsPtr->floatingNuclearCharges,
            d_nOMPThreads,
            d_atomTypeAtributes,
            d_dftParamsPtr->reproducible_output,
            d_dftParamsPtr->verbosity,
            d_dftParamsPtr->useDevice,
            d_dftParamsPtr->memOptMode);
      }
    if (d_dftParamsPtr->solverMode == "NSCF")
      {
        if (d_dftParamsPtr->writePdosFile)
          {
            d_atomCenteredOrbitalsPostProcessingPtr = std::make_shared<
              dftfe::atomCenteredOrbitalsPostProcessing<dataTypes::number,
                                                        memorySpace>>(
              d_mpiCommParent,
              mpi_communicator,
              d_dftfeScratchFolderName,
              atomTypes,
              d_dftParamsPtr->reproducible_output,
              d_dftParamsPtr->verbosity,
              d_dftParamsPtr->useDevice,
              d_dftParamsPtr);
          }
      }

    if (d_dftParamsPtr->verbosity >= 1)
      {
        if (d_dftParamsPtr->nonLinearCoreCorrection == true)
          {
            pcout
              << "Atleast one atom has pseudopotential with nonlinear core correction"
              << std::endl;
            AssertThrow(
              !(d_dftParamsPtr->XCType.substr(0, 4) == "MGGA" &&
                d_dftParamsPtr->isIonForce),
              dealii::ExcMessage(
                "DFT-FE Error : Computation of ION FORCE with MGGA functional with the pseudopotentials"
                " with NLCC is not completed yet."));
          }
      }

    d_elpaScala->processGridELPASetup(d_numEigenValues, *d_dftParamsPtr);
    MPI_Barrier(d_mpiCommParent);
    computingTimerStandard.leave_subsection("Atomic system initialization");
  }

  // dft pseudopotential init
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initPseudoPotentialAll(
    const bool updateNonlocalSparsity)
  {
    if (d_dftParamsPtr->isPseudopotential)
      {
        dealii::TimerOutput::Scope scope(computing_timer, "psp init");
        pcout << std::endl << "Pseudopotential initalization...." << std::endl;

        double init_core;
        MPI_Barrier(d_mpiCommParent);
        init_core = MPI_Wtime();

        if (d_dftParamsPtr->nonLinearCoreCorrection == true)
          initCoreRho();

        MPI_Barrier(d_mpiCommParent);
        init_core = MPI_Wtime() - init_core;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << "initPseudoPotentialAll: Time taken for initializing core density for non-linear core correction: "
            << init_core << std::endl;
        determineAtomsOfInterstPseudopotential(atomLocations);
        MPI_Barrier(d_mpiCommParent);
        if (d_dftParamsPtr->isPseudopotential == true)
          {
            d_oncvClassPtr->initialiseNonLocalContribution(
              d_atomLocationsInterestPseudopotential,
              d_imageIdsTrunc,
              d_imagePositionsTrunc,
              d_kPointWeights,     // accounts for interpool
              d_kPointCoordinates, // accounts for interpool
              updateNonlocalSparsity);
          }
        if (d_dftParamsPtr->solverMode == "NSCF")
          {
            if (d_dftParamsPtr->writePdosFile)
              {
                d_atomCenteredOrbitalsPostProcessingPtr
                  ->initialiseNonLocalContribution(
                    d_atomLocationsInterestPseudopotential,
                    d_imageIdsTrunc,
                    d_imagePositionsTrunc,
                    d_kPointWeights,
                    d_kPointCoordinates,
                    updateNonlocalSparsity);
              }
          }
      }
  }


  // generate image charges and update k point cartesian coordinates based on
  // current lattice vectors
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initImageChargesUpdateKPoints(bool flag)
  {
    dealii::TimerOutput::Scope scope(computing_timer,
                                     "image charges and k point generation");
    pcout
      << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
      << std::endl;
    for (dftfe::Int i = 0; i < d_domainBoundingVectors.size(); ++i)
      {
        pcout << "v" << i + 1 << " : " << d_domainBoundingVectors[i][0] << " "
              << d_domainBoundingVectors[i][1] << " "
              << d_domainBoundingVectors[i][2] << std::endl;
      }
    pcout
      << "-----------------------------------------------------------------------------------------"
      << std::endl;

    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        pcout << "-----Fractional coordinates of atoms------ " << std::endl;
        for (dftfe::uInt i = 0; i < atomLocations.size(); ++i)
          {
            atomLocations[i] = atomLocationsFractional[i];
            pcout << "AtomId " << i << ":  " << atomLocationsFractional[i][2]
                  << " " << atomLocationsFractional[i][3] << " "
                  << atomLocationsFractional[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
        // sanity check on fractional coordinates
        std::vector<bool> periodicBc(3, false);
        periodicBc[0]    = d_dftParamsPtr->periodicX;
        periodicBc[1]    = d_dftParamsPtr->periodicY;
        periodicBc[2]    = d_dftParamsPtr->periodicZ;
        const double tol = 1e-6;

        if (flag)
          {
            for (dftfe::uInt i = 0; i < atomLocationsFractional.size(); ++i)
              {
                for (dftfe::uInt idim = 0; idim < 3; ++idim)
                  {
                    if (periodicBc[idim])
                      AssertThrow(
                        atomLocationsFractional[i][2 + idim] > -tol &&
                          atomLocationsFractional[i][2 + idim] < 1.0 + tol,
                        dealii::ExcMessage(
                          "DFT-FE Error: periodic direction fractional coordinates doesn't lie in [0,1]. Please check input"
                          "fractional coordinates, or if this is an ionic relaxation step, please check the corresponding"
                          "algorithm."));
                    if (!periodicBc[idim])
                      AssertThrow(
                        atomLocationsFractional[i][2 + idim] > tol &&
                          atomLocationsFractional[i][2 + idim] < 1.0 - tol,
                        dealii::ExcMessage(
                          "DFT-FE Error: non-periodic direction fractional coordinates doesn't lie in (0,1). Please check"
                          "input fractional coordinates, or if this is an ionic relaxation step, please check the"
                          "corresponding algorithm."));
                  }
              }
          }

        generateImageCharges(d_pspCutOff,
                             d_imageIds,
                             d_imageCharges,
                             d_imagePositions);

        generateImageCharges(d_pspCutOffTrunc,
                             d_imageIdsTrunc,
                             d_imageChargesTrunc,
                             d_imagePositionsTrunc);

        if ((d_dftParamsPtr->verbosity >= 4 ||
             d_dftParamsPtr->reproducible_output))
          pcout << "Number Image Charges  " << d_imageIds.size() << std::endl;

        internaldft::convertToCellCenteredCartesianCoordinates(
          atomLocations, d_domainBoundingVectors);
#ifdef USE_COMPLEX
        recomputeKPointCoordinates();
#endif
        if (d_dftParamsPtr->verbosity >= 4)
          {
            // FIXME: Print all k points across all pools
            pcout
              << "-------------------k points cartesian coordinates and weights-----------------------------"
              << std::endl;
            for (dftfe::uInt i = 0; i < d_kPointWeights.size(); ++i)
              {
                pcout << " [" << d_kPointCoordinates[3 * i + 0] << ", "
                      << d_kPointCoordinates[3 * i + 1] << ", "
                      << d_kPointCoordinates[3 * i + 2] << "] "
                      << d_kPointWeights[i] << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------------"
              << std::endl;
          }
      }
    else
      {
        //
        // print cartesian coordinates
        //
        pcout
          << "------------Cartesian coordinates of atoms (origin at center of domain)------------------"
          << std::endl;
        for (dftfe::uInt i = 0; i < atomLocations.size(); ++i)
          {
            pcout << "AtomId " << i << ":  " << atomLocations[i][2] << " "
                  << atomLocations[i][3] << " " << atomLocations[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;

        //
        // redundant call (check later)
        //
        generateImageCharges(d_pspCutOff,
                             d_imageIds,
                             d_imageCharges,
                             d_imagePositions);

        generateImageCharges(d_pspCutOffTrunc,
                             d_imageIdsTrunc,
                             d_imageChargesTrunc,
                             d_imagePositionsTrunc);
      }
  }

  // dft init
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::init()
  {
    computingTimerStandard.enter_subsection("KSDFT problem initialization");

    d_BLASWrapperPtrHost = std::make_shared<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>();
    d_basisOperationsPtrHost = std::make_shared<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>(
      d_BLASWrapperPtrHost);
    d_basisOperationsPtrElectroHost = std::make_shared<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>(
      d_BLASWrapperPtrHost);
#if defined(DFTFE_WITH_DEVICE)
    if (d_dftParamsPtr->useDevice)
      {
        d_BLASWrapperPtr = std::make_shared<dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>>();
        d_basisOperationsPtrDevice = std::make_shared<
          dftfe::basis::FEBasisOperations<dataTypes::number,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>(
          d_BLASWrapperPtr);
        d_basisOperationsPtrElectroDevice = std::make_shared<
          dftfe::basis::FEBasisOperations<double,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>(
          d_BLASWrapperPtr);
      }
#endif
    initImageChargesUpdateKPoints();

    calculateNearestAtomDistances();

    computing_timer.enter_subsection("mesh generation");
    //
    // generate mesh (both parallel and serial)
    // while parallel meshes are always generated, serial meshes are only
    // generated for following three cases: symmetrization is on, ionic
    // optimization is on as well as reuse wfcs and density from previous ionic
    // step is on, or if serial constraints generation is on.
    //
    // if (d_dftParamsPtr->loadRhoData)
    //   {
    //     d_mesh.generateCoarseMeshesForRestart(
    //       atomLocations,
    //       d_imagePositionsTrunc,
    //       d_imageIdsTrunc,
    //       d_nearestAtomDistances,
    //       d_domainBoundingVectors,
    //       d_dftParamsPtr->useSymm ||
    //         d_dftParamsPtr->createConstraintsFromSerialDofhandler);

    //     loadTriaInfoAndRhoNodalData();
    //   }
    // else
    //{
    d_mesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      d_imagePositionsTrunc,
      d_imageIdsTrunc,
      d_nearestAtomDistances,
      d_domainBoundingVectors,
      d_dftParamsPtr->useSymm ||
        d_dftParamsPtr->createConstraintsFromSerialDofhandler);
    //}
    computing_timer.leave_subsection("mesh generation");

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Mesh generation completed");
    //
    // get access to triangulation objects from meshGenerator class
    //
    dealii::parallel::distributed::Triangulation<3> &triangulationPar =
      d_mesh.getParallelMeshMoved();

    //
    // initialize dofHandlers and hanging-node constraints and periodic
    // constraints on the unmoved Mesh
    //
    initUnmovedTriangulation(triangulationPar);

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initUnmovedTriangulation completed");
#ifdef USE_COMPLEX
    if (d_dftParamsPtr->useSymm)
      symmetryPtr->initSymmetry();
#endif



    //
    // move triangulation to have atoms on triangulation vertices
    //
    if (!d_dftParamsPtr->floatingNuclearCharges)
      moveMeshToAtoms(triangulationPar, d_mesh.getSerialMeshUnmoved());


    if (d_dftParamsPtr->smearedNuclearCharges)
      calculateSmearedChargeWidths();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "moveMeshToAtoms completed");
    //
    // initialize dirichlet BCs for total potential and vSelf poisson solutions
    //
    initBoundaryConditions();
    d_smearedChargeMomentsComputed = false;

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initBoundaryConditions completed");

    //
    // initialize pseudopotential data for both local and nonlocal part
    //
    if (d_dftParamsPtr->isPseudopotential == true)
      d_oncvClassPtr->initialise(d_basisOperationsPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                                 d_basisOperationsPtrDevice,
#endif
                                 d_BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                                 d_BLASWrapperPtr,
#endif
                                 d_densityQuadratureId,
                                 d_lpspQuadratureId,
                                 d_sparsityPatternQuadratureId,
                                 d_nlpspQuadratureId,
                                 d_densityQuadratureIdElectro,
                                 d_excManagerPtr,
                                 atomLocations,
                                 d_numEigenValues,
                                 d_dftParamsPtr->useSinglePrecCheby,
                                 (d_dftParamsPtr->isIonForce) ||
                                   d_dftParamsPtr->isCellStress);

    if (d_dftParamsPtr->solverMode == "NSCF")
      {
        if (d_dftParamsPtr->writePdosFile)
          {
            d_atomCenteredOrbitalsPostProcessingPtr->initialise(
              d_basisOperationsPtrHost,
#if defined(DFTFE_WITH_DEVICE)
              d_basisOperationsPtrDevice,
#endif
              d_BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
              d_BLASWrapperPtr,
#endif
              d_sparsityPatternQuadratureId,
              d_nlpspQuadratureId,
              atomLocations,
              d_numEigenValues);
          }
      }
    //
    // initialize guesses for electron-density and wavefunctions
    //
    initElectronicFields();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initElectronicFields completed");


    initPseudoPotentialAll();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initPseudopotential completed");

    //
    // Apply Gaussian displacments to atoms and mesh if input gaussian
    // displacments are read from file. When restarting a relaxation, this must
    // be done only once at the begining- this is why the flag is to false after
    // the Gaussian movement. The last flag to updateAtomPositionsAndMoveMesh is
    // set to true to force use of single atom solutions.
    //
    if (d_isAtomsGaussianDisplacementsReadFromFile)
      {
        updateAtomPositionsAndMoveMesh(d_atomsDisplacementsGaussianRead,
                                       1e+4,
                                       true);
        d_isAtomsGaussianDisplacementsReadFromFile = false;
      }

    // if (d_dftParamsPtr->loadRhoData)
    //   {
    //     if (d_dftParamsPtr->verbosity >= 1)
    //       pcout
    //         << "Overwriting input density data to SCF solve with data read
    //         from restart file.."
    //         << std::endl;

    //     // Note: d_rhoInNodalValuesRead is not compatible with
    //     // d_matrixFreeDataPRefined
    //     for (dftfe::uInt i = 0;
    //          i < d_densityInNodalValues[0].locally_owned_size();
    //          i++)
    //       d_densityInNodalValues[0].local_element(i) =
    //         d_rhoInNodalValuesRead.local_element(i);

    //     bool isGradDensityDataDependent =
    //       (d_excManagerPtr->getExcSSDFunctionalObj()
    //          ->getDensityBasedFamilyType() == densityFamilyType::GGA);

    //     if (d_dftParamsPtr->spinPolarized == 1)
    //       {
    //         d_densityInNodalValues[1] = 0;
    //         for (dftfe::uInt i = 0;
    //              i < d_densityInNodalValues[1].locally_owned_size();
    //              i++)
    //           {
    //             d_densityInNodalValues[1].local_element(i) =
    //               d_magInNodalValuesRead.local_element(i);
    //           }
    //       }

    //     if (d_dftParamsPtr->spinPolarized == 1 &&
    //         d_dftParamsPtr->constraintMagnetization)
    //       {
    //         // normalize rho mag
    //         const double netMag =
    //           totalCharge(d_matrixFreeDataPRefined,
    //           d_densityInNodalValues[1]);

    //         const double shift =
    //           (d_dftParamsPtr->tot_magnetization * numElectrons - shift) /
    //           numElectrons;

    //         d_densityInNodalValues[1].add(shift, d_densityInNodalValues[0]);

    //         if (d_dftParamsPtr->verbosity >= 1)
    //           {
    //             pcout << "Net magnetization before Normalizing:  " << netMag
    //                   << std::endl;
    //             pcout << "Net magnetization after Normalizing: "
    //                   << totalCharge(d_matrixFreeDataPRefined,
    //                                  d_densityInNodalValues[1])
    //                   << std::endl;
    //           }
    //       }

    //     for (dftfe::uInt iComp = 0; iComp < d_densityInNodalValues.size();
    //          ++iComp)
    //       interpolateDensityNodalDataToQuadratureDataGeneral(
    //         d_basisOperationsPtrElectroHost,
    //         d_densityDofHandlerIndexElectro,
    //         d_densityQuadratureIdElectro,
    //         d_densityInNodalValues[iComp],
    //         d_densityInQuadValues[iComp],
    //         d_gradDensityInQuadValues[iComp],
    //         d_gradDensityInQuadValues[iComp],
    //         isGradDensityDataDependent);


    //     if ((d_dftParamsPtr->solverMode == "GEOOPT"))
    //       {
    //         d_densityOutNodalValues = d_densityInNodalValues;
    //         for (dftfe::uInt iComp = 0; iComp <
    //         d_densityOutNodalValues.size();
    //              ++iComp)
    //           d_densityOutNodalValues[iComp].update_ghost_values();

    //         d_densityOutQuadValues = d_densityInQuadValues;

    //         if (isGradDensityDataDependent)
    //           d_gradDensityOutQuadValues = d_gradDensityInQuadValues;
    //       }

    //     d_isRestartGroundStateCalcFromChk = true;
    //   }

    d_isFirstFilteringCall.clear();
    d_isFirstFilteringCall.resize((d_dftParamsPtr->spinPolarized + 1) *
                                    d_kPointWeights.size(),
                                  true);


    initHubbardOperator();
    if (d_useHubbard && (d_dftParamsPtr->loadQuadData))
      {
        d_hubbardClassPtr->readHubbOccFromFile();
      }
    initializeKohnShamDFTOperator();

    d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.clear();
    d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.resize(
      atomLocations.size() * 3, 0.0);

    computingTimerStandard.leave_subsection("KSDFT problem initialization");
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initHubbardOperator()
  {
    if (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
        ExcFamilyType::DFTPlusU)
      {
        double init_hubbOp;
        MPI_Barrier(d_mpiCommParent);
        init_hubbOp = MPI_Wtime();

        std::shared_ptr<ExcDFTPlusU<dataTypes::number, memorySpace>>
          excHubbPtr = std::dynamic_pointer_cast<
            ExcDFTPlusU<dataTypes::number, memorySpace>>(
            d_excManagerPtr->getSSDSharedObj());

        excHubbPtr->initialiseHubbardClass(
          d_mpiCommParent,
          mpi_communicator,
          interpoolcomm,
          interBandGroupComm,
          getBasisOperationsMemSpace(),
          getBasisOperationsHost(),
          getBLASWrapperMemSpace(),
          getBLASWrapperHost(),
          d_densityDofHandlerIndex,
          d_nlpspQuadratureId,
          d_sparsityPatternQuadratureId,
          d_numEigenValues, // The total number of waveFunctions that are passed
                            // to the operator
          d_dftParamsPtr->spinPolarized == 1 ? 2 : 1,
          *d_dftParamsPtr,
          d_dftfeScratchFolderName,
          false, // singlePrecNonLocalOperator
          true,  // updateNonlocalSparsity
          atomLocations,
          atomLocationsFractional,
          d_imageIds,
          d_imagePositions,
          d_kPointCoordinates,
          d_kPointWeights,
          d_domainBoundingVectors);

        d_hubbardClassPtr = excHubbPtr->getHubbardClass();

        d_useHubbard = true;

        AssertThrow(d_nOMPThreads == 1,
                    dealii::ExcMessage(
                      "open mp is not compatible with hubbard "));

        AssertThrow(d_dftParamsPtr->mixingMethod != "LOW_RANK_DIELECM_PRECOND",
                    dealii::ExcMessage(
                      "LRDM preconditioner is not compatible with hubbard "));

        init_hubbOp = MPI_Wtime() - init_hubbOp;

        if (d_dftParamsPtr->verbosity >= 2)
          pcout << "Time taken for hubbard class initialization: "
                << init_hubbOp << std::endl;
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initNoRemesh(
    const bool updateImagesAndKPointsAndVselfBins,
    const bool checkSmearedChargeWidthsForOverlap,
    const bool useSingleAtomSolutionOverride,
    const bool isMeshDeformed)
  {
    computingTimerStandard.enter_subsection("KSDFT problem initialization");
    if (updateImagesAndKPointsAndVselfBins)
      {
        initImageChargesUpdateKPoints();
      }

    if (checkSmearedChargeWidthsForOverlap)
      {
        calculateNearestAtomDistances();

        if (d_dftParamsPtr->smearedNuclearCharges)
          calculateSmearedChargeWidths();

        d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.clear();
        d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.resize(
          atomLocations.size() * 3, 0.0);
      }

    //
    // reinitialize dirichlet BCs for total potential and vSelf poisson
    // solutions
    //
    double init_bc;
    MPI_Barrier(d_mpiCommParent);
    init_bc = MPI_Wtime();


    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);

    // false option reinitializes vself bins from scratch wheras true option
    // only updates the boundary conditions
    const bool updateOnlyBinsBc = !updateImagesAndKPointsAndVselfBins;
    initBoundaryConditions(isMeshDeformed || d_dftParamsPtr->isCellStress,
                           updateOnlyBinsBc);
    d_smearedChargeMomentsComputed = false;
    MPI_Barrier(d_mpiCommParent);
    init_bc = MPI_Wtime() - init_bc;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "
        << init_bc << std::endl;

    double init_rho;
    MPI_Barrier(d_mpiCommParent);
    init_rho = MPI_Wtime();

    if (useSingleAtomSolutionOverride)
      {
        readPSI();
        initRho();
      }
    else
      {
        //
        // rho init (use previous ground state electron density)
        //
        // if(d_dftParamsPtr->mixingMethod != "ANDERSON_WITH_KERKER")
        //   solveNoSCF();

        if (!d_dftParamsPtr->reuseWfcGeoOpt)
          readPSI();

        noRemeshRhoDataInit();

        if (d_dftParamsPtr->reuseDensityGeoOpt >= 1 &&
            d_dftParamsPtr->solverMode == "GEOOPT")
          {
            if (d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
                d_dftParamsPtr->spinPolarized != 1)
              {
                d_rhoOutNodalValuesSplit.add(
                  -totalCharge(d_matrixFreeDataPRefined,
                               d_rhoOutNodalValuesSplit) /
                  d_domainVolume);

                initAtomicRho();

                d_basisOperationsPtrElectroHost->interpolate(
                  d_rhoOutNodalValuesSplit,
                  d_densityDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_densityInQuadValues[0],
                  d_gradDensityInQuadValues[0],
                  d_gradDensityInQuadValues[0],
                  isGradDensityDataDependent);

                addAtomicRhoQuadValuesGradients(d_densityInQuadValues[0],
                                                d_gradDensityInQuadValues[0],
                                                isGradDensityDataDependent);

                normalizeRhoInQuadValues();

                l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                        d_constraintsRhoNodal,
                                        d_densityDofHandlerIndexElectro,
                                        d_densityQuadratureIdElectro,
                                        d_densityInQuadValues[0],
                                        d_densityInNodalValues[0]);
              }
          }

        else if (d_dftParamsPtr->extrapolateDensity == 1 &&
                 d_dftParamsPtr->spinPolarized != 1 &&
                 d_dftParamsPtr->solverMode == "MD")
          {
            d_basisOperationsPtrElectroHost->interpolate(
              d_densityOutNodalValues[0],
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_densityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              isGradDensityDataDependent);

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_densityInQuadValues[0],
                                    d_densityInNodalValues[0]);
          }
        else if (d_dftParamsPtr->extrapolateDensity == 2 &&
                 d_dftParamsPtr->spinPolarized != 1 &&
                 d_dftParamsPtr->solverMode == "MD")
          {
            initAtomicRho();
            d_basisOperationsPtrElectroHost->interpolate(
              d_rhoOutNodalValuesSplit,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_densityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              isGradDensityDataDependent);

            addAtomicRhoQuadValuesGradients(d_densityInQuadValues[0],
                                            d_gradDensityInQuadValues[0],
                                            isGradDensityDataDependent);

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_densityInQuadValues[0],
                                    d_densityInNodalValues[0]);
          }
        else
          {
            initRho();
          }
      }

    MPI_Barrier(d_mpiCommParent);
    init_rho = MPI_Wtime() - init_rho;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "updateAtomPositionsAndMoveMesh: Time taken for initRho: "
            << init_rho << std::endl;

    //
    // reinitialize pseudopotential related data structures
    //
    double init_pseudo;
    MPI_Barrier(d_mpiCommParent);
    init_pseudo = MPI_Wtime();

    initPseudoPotentialAll();

    MPI_Barrier(d_mpiCommParent);
    init_pseudo = MPI_Wtime() - init_pseudo;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Time taken for initPseudoPotentialAll: " << init_pseudo
            << std::endl;

    d_isFirstFilteringCall.clear();
    d_isFirstFilteringCall.resize((d_dftParamsPtr->spinPolarized + 1) *
                                    d_kPointWeights.size(),
                                  true);

    if (d_useHubbard)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &hubbOccIn = d_hubbardClassPtr->getOccMatIn();

        initHubbardOperator();

        d_hubbardClassPtr->setInOccMatrix(hubbOccIn);
      }

    double init_ksoperator;
    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime();

    if (isMeshDeformed)
      initializeKohnShamDFTOperator();
    else
      reInitializeKohnShamDFTOperator();

    init_ksoperator = MPI_Wtime() - init_ksoperator;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Time taken for kohnShamDFTOperator class reinitialization: "
            << init_ksoperator << std::endl;

    computingTimerStandard.leave_subsection("KSDFT problem initialization");
  }

  //
  // deform domain and call appropriate reinits
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::deformDomain(
    const dealii::Tensor<2, 3, double> &deformationGradient,
    const bool                          vselfPerturbationUpdateForStress,
    const bool                          useSingleAtomSolutionsOverride,
    const bool                          print)
  {
    d_affineTransformMesh.initMoved(d_domainBoundingVectors);
    d_affineTransformMesh.transform(deformationGradient);

    dftUtils::transformDomainBoundingVectors(d_domainBoundingVectors,
                                             deformationGradient);

    if (print)
      {
        pcout
          << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
          << std::endl;
        for (dftfe::Int i = 0; i < d_domainBoundingVectors.size(); ++i)
          {
            pcout << "v" << i + 1 << " : " << d_domainBoundingVectors[i][0]
                  << " " << d_domainBoundingVectors[i][1] << " "
                  << d_domainBoundingVectors[i][2] << std::endl;
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

#ifdef USE_COMPLEX
    if (!vselfPerturbationUpdateForStress)
      recomputeKPointCoordinates();
#endif

    // update atomic and image positions without any wrapping across periodic
    // boundary
    std::vector<dealii::Tensor<1, 3, double>> imageDisplacements(
      d_imagePositions.size());
    std::vector<dealii::Tensor<1, 3, double>> imageDisplacementsTrunc(
      d_imagePositionsTrunc.size());

    for (dftfe::Int iImage = 0; iImage < d_imagePositions.size(); ++iImage)
      {
        dealii::Point<3>  imageCoor;
        const dftfe::uInt imageChargeId = d_imageIds[iImage];
        imageCoor[0]                    = d_imagePositions[iImage][0];
        imageCoor[1]                    = d_imagePositions[iImage][1];
        imageCoor[2]                    = d_imagePositions[iImage][2];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacements[iImage] = imageCoor - atomCoor;
      }

    for (dftfe::Int iImage = 0; iImage < d_imagePositionsTrunc.size(); ++iImage)
      {
        dealii::Point<3>  imageCoor;
        const dftfe::uInt imageChargeId = d_imageIdsTrunc[iImage];
        imageCoor[0]                    = d_imagePositionsTrunc[iImage][0];
        imageCoor[1]                    = d_imagePositionsTrunc[iImage][1];
        imageCoor[2]                    = d_imagePositionsTrunc[iImage][2];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacementsTrunc[iImage] = imageCoor - atomCoor;
      }

    for (dftfe::uInt i = 0; i < atomLocations.size(); ++i)
      atomLocations[i] = atomLocationsFractional[i];

    if (print)
      {
        pcout << "-----Fractional coordinates of atoms------ " << std::endl;
        for (dftfe::uInt i = 0; i < atomLocations.size(); ++i)
          {
            pcout << "AtomId " << i << ":  " << atomLocationsFractional[i][2]
                  << " " << atomLocationsFractional[i][3] << " "
                  << atomLocationsFractional[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

    internaldft::convertToCellCenteredCartesianCoordinates(
      atomLocations, d_domainBoundingVectors);


    for (dftfe::Int iImage = 0; iImage < d_imagePositions.size(); ++iImage)
      {
        const dftfe::uInt imageChargeId = d_imageIds[iImage];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacements[iImage] =
          deformationGradient * imageDisplacements[iImage];

        d_imagePositions[iImage][0] =
          atomCoor[0] + imageDisplacements[iImage][0];
        d_imagePositions[iImage][1] =
          atomCoor[1] + imageDisplacements[iImage][1];
        d_imagePositions[iImage][2] =
          atomCoor[2] + imageDisplacements[iImage][2];
      }

    for (dftfe::Int iImage = 0; iImage < d_imagePositionsTrunc.size(); ++iImage)
      {
        const dftfe::uInt imageChargeId = d_imageIdsTrunc[iImage];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacementsTrunc[iImage] =
          deformationGradient * imageDisplacementsTrunc[iImage];

        d_imagePositionsTrunc[iImage][0] =
          atomCoor[0] + imageDisplacementsTrunc[iImage][0];
        d_imagePositionsTrunc[iImage][1] =
          atomCoor[1] + imageDisplacementsTrunc[iImage][1];
        d_imagePositionsTrunc[iImage][2] =
          atomCoor[2] + imageDisplacementsTrunc[iImage][2];
      }

    if (vselfPerturbationUpdateForStress)
      {
        //
        // reinitialize dirichlet BCs for total potential and vSelf poisson
        // solutions
        //
        double init_bc;
        MPI_Barrier(d_mpiCommParent);
        init_bc = MPI_Wtime();


        // first true option only updates the boundary conditions
        // second true option signals update is only for vself perturbation
        initBoundaryConditions(true, true, true);
        d_smearedChargeMomentsComputed = false;
        MPI_Barrier(d_mpiCommParent);
        init_bc = MPI_Wtime() - init_bc;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << "updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "
            << init_bc << std::endl;
      }
    else
      {
        initNoRemesh(false, true, useSingleAtomSolutionsOverride, true);
      }
  }


  //
  // generate a-posteriori mesh
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::aposterioriMeshGenerate()
  {
    //
    // get access to triangulation objects from meshGenerator class
    //
    dealii::parallel::distributed::Triangulation<3> &triangulationPar =
      d_mesh.getParallelMeshMoved();
    dftfe::uInt numberLevelRefinements = d_dftParamsPtr->numLevels;
    dftfe::uInt numberWaveFunctionsErrorEstimate =
      d_dftParamsPtr->numberWaveFunctionsForEstimate;
    bool        refineFlag = true;
    dftfe::uInt countLevel = 0;
    double      traceXtKX  = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
    double      traceXtKXPrev = traceXtKX;

    while (refineFlag)
      {
        if (numberLevelRefinements > 0)
          {
            distributedCPUVec<double> tempVec;
            matrix_free_data.initialize_dof_vector(tempVec);

            std::vector<distributedCPUVec<double>> eigenVectorsArray(
              numberWaveFunctionsErrorEstimate);

            for (dftfe::uInt i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
              eigenVectorsArray[i].reinit(tempVec);


            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedHost.data(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              std::make_pair(0, numberWaveFunctionsErrorEstimate),
              eigenVectorsArray);


            for (dftfe::uInt i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
              {
                constraintsNone.distribute(eigenVectorsArray[i]);
                eigenVectorsArray[i].update_ghost_values();
              }


            d_mesh.generateAutomaticMeshApriori(
              dofHandler,
              triangulationPar,
              eigenVectorsArray,
              d_dftParamsPtr->finiteElementPolynomialOrder);
          }


        //
        // initialize dofHandlers of refined mesh and move triangulation
        //
        initUnmovedTriangulation(triangulationPar);
        moveMeshToAtoms(triangulationPar, d_mesh.getSerialMeshUnmoved());
        initBoundaryConditions();
        d_smearedChargeMomentsComputed = false;
        initElectronicFields();
        initPseudoPotentialAll();

        //
        // compute Tr(XtKX) for each level of mesh
        //
        traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
        if (d_dftParamsPtr->verbosity > 0)
          pcout << " Tr(XtKX) value for Level: " << countLevel << " "
                << traceXtKX << std::endl;

        // compute change in traceXtKX
        double deltaKinetic =
          std::abs(traceXtKX - traceXtKXPrev) / atomLocations.size();

        // reset traceXtkXPrev to traceXtKX
        traceXtKXPrev = traceXtKX;

        //
        // set refineFlag
        //
        countLevel += 1;
        if (countLevel >= numberLevelRefinements ||
            deltaKinetic <= d_dftParamsPtr->toleranceKinetic)
          refineFlag = false;
      }
  }


  //
  // dft run
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::run()
  {
    if (d_dftParamsPtr->meshAdaption)
      aposterioriMeshGenerate();

    if (d_dftParamsPtr->restartFolder != "." &&
        (d_dftParamsPtr->saveQuadData) &&
        dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      mkdir(d_dftParamsPtr->restartFolder.c_str(), ACCESSPERMS);

    if (d_dftParamsPtr->solverMode == "FUNCTIONAL_TEST")
      runFunctionalTest();
    else if (d_dftParamsPtr->solverMode == "GS")
      {
        bool flag = d_dftParamsPtr->applyExternalPotential;
        solve(true, true, d_isRestartGroundStateCalcFromChk, flag);
      }
    else if (d_dftParamsPtr->solverMode == "NSCF")
      {
        solveNoSCF();
        if (d_dftParamsPtr->writePdosFile)
          {
            if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
              {
                d_atomCenteredOrbitalsPostProcessingPtr
                  ->computeAtomCenteredEntries(
                    &d_eigenVectorsFlattenedHost,
                    d_numEigenValues,
                    eigenValues,
                    d_basisOperationsPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                    d_BLASWrapperPtr,
#endif
                    d_BLASWrapperPtrHost,
                    d_lpspQuadratureId,
                    d_kPointWeights,
                    interBandGroupComm,
                    interpoolcomm,
                    d_dftParamsPtr,
                    fermiEnergy,
                    d_dftParamsPtr->highestStateOfInterestForChebFiltering);
              }
#ifdef DFTFE_WITH_DEVICE
            else if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
              {
                d_atomCenteredOrbitalsPostProcessingPtr
                  ->computeAtomCenteredEntries(
                    &d_eigenVectorsFlattenedDevice,
                    d_numEigenValues,
                    eigenValues,
                    d_basisOperationsPtrDevice,
                    d_BLASWrapperPtr,
                    d_BLASWrapperPtrHost,
                    d_lpspQuadratureId,
                    d_kPointWeights,
                    interBandGroupComm,
                    interpoolcomm,
                    d_dftParamsPtr,
                    fermiEnergy,
                    d_dftParamsPtr->highestStateOfInterestForChebFiltering);
              }

#endif
          }
      }
    else if (d_dftParamsPtr->solverMode == "BANDS")
      {
        solveBands();
        writeBands();
      }

    if (d_dftParamsPtr->writeStructreEnergyForcesFileForPostProcess)
      writeStructureEnergyForcesDataPostProcess(
        "structureEnergyForcesGSData.txt");

    if (d_dftParamsPtr->writeWfcSolutionFields)
      outputWfc();

    if (d_dftParamsPtr->printKE)
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          kineticEnergyDensityValues;
        computeAndPrintKE(kineticEnergyDensityValues);
      }


    if (d_dftParamsPtr->writeDensitySolutionFields)
      outputDensity();
    if (d_dftParamsPtr->zPlanarAverageDensity)
      {
        distributedCPUVec<double> rhoNodalField;
        d_matrixFreeDataPRefined.initialize_dof_vector(
          rhoNodalField, d_densityDofHandlerIndexElectro);
        rhoNodalField = 0;
        l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                d_constraintsRhoNodal,
                                d_densityDofHandlerIndexElectro,
                                d_densityQuadratureIdElectro,
                                d_densityOutQuadValues[0],
                                rhoNodalField);

        d_constraintsRhoNodal.distribute(rhoNodalField);
        rhoNodalField.update_ghost_values();

        double charge = totalCharge(d_matrixFreeDataPRefined.get_dof_handler(
                                      d_densityDofHandlerIndexElectro),
                                    rhoNodalField);
        pcout << "Total charge in nodal Field: " << charge << std::endl;
        std::vector<std::vector<double>> densityPlanarValues;
        computePlanarAverageField(rhoNodalField,
                                  d_matrixFreeDataPRefined,
                                  d_densityDofHandlerIndexElectro,
                                  mpi_communicator,
                                  interpoolcomm,
                                  densityPlanarValues);
        std::vector<std::string> labels = {"Z-Coordinate",
                                           "planar avg. density(e/bohr)"};
        dftfe::dftUtils::writeDataIntoFile(labels,
                                           densityPlanarValues,
                                           "planarAverageDensity.txt",
                                           d_mpiCommParent);
      }
    if (d_dftParamsPtr->zPlanarAveragePhi)
      {
        d_constraintsForTotalPotentialElectro.distribute(d_phiTotRhoOut);
        d_phiTotRhoOut.update_ghost_values();
        std::vector<std::vector<double>> phiOutPlanarValues;
        computePlanarAverageField(d_phiTotRhoOut,
                                  d_matrixFreeDataPRefined,
                                  d_phiTotDofHandlerIndexElectro,
                                  mpi_communicator,
                                  interpoolcomm,
                                  phiOutPlanarValues);
        d_phiTotRhoOut.zero_out_ghost_values();
        d_constraintsForTotalPotentialElectro.set_zero(d_phiTotRhoOut);
        std::vector<std::string> labels = {
          "Z-Coordinate", "planar avg. electrostatic Potential(Ha/e)"};
        dftfe::dftUtils::writeDataIntoFile(
          labels,
          phiOutPlanarValues,
          "planarAverageElectrostaticPotential.txt",
          d_mpiCommParent);
      }
    if (d_dftParamsPtr->zPlanarAverageVbare)
      {
        std::vector<std::vector<double>> phiOutPlanarValues,
          pseudLocPlanarValues, uExtPlanarValues;

        std::vector<std::vector<double>> delta_phiOutPlanarValues,
          delta_pseudLocPlanarValues, delta_uExtPlanarValues, delta_total;
        delta_total.resize(2, std::vector<double>(2, 0.0));
        distributedCPUVec<double> pseudoVLocNodal;
        d_matrixFreeDataPRefined.initialize_dof_vector(
          pseudoVLocNodal, d_phiTotDofHandlerIndexElectro);
        pseudoVLocNodal = 0;
        l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                d_constraintsPRefined,
                                d_phiExtDofHandlerIndexElectro,
                                d_lpspQuadratureIdElectro,
                                d_pseudoVLoc,
                                pseudoVLocNodal);
        d_constraintsPRefined.distribute(pseudoVLocNodal);
        pseudoVLocNodal.update_ghost_values();
        computePlanarAverageField(pseudoVLocNodal,
                                  d_matrixFreeDataPRefined,
                                  d_phiExtDofHandlerIndexElectro,
                                  mpi_communicator,
                                  interpoolcomm,
                                  pseudLocPlanarValues);


        computeDeltaPlanarAverageField(pseudoVLocNodal,
                                       d_matrixFreeDataPRefined,
                                       d_phiExtDofHandlerIndexElectro,
                                       mpi_communicator,
                                       interpoolcomm,
                                       delta_pseudLocPlanarValues);


        d_constraintsForTotalPotentialElectro.distribute(d_phiTotRhoOut);
        d_phiTotRhoOut.update_ghost_values();
        computePlanarAverageField(d_phiTotRhoOut,
                                  d_matrixFreeDataPRefined,
                                  d_phiTotDofHandlerIndexElectro,
                                  mpi_communicator,
                                  interpoolcomm,
                                  phiOutPlanarValues);
        computeDeltaPlanarAverageField(d_phiTotRhoOut,
                                       d_matrixFreeDataPRefined,
                                       d_phiTotDofHandlerIndexElectro,
                                       mpi_communicator,
                                       interpoolcomm,
                                       delta_phiOutPlanarValues);

        d_phiTotRhoOut.zero_out_ghost_values();
        d_constraintsForTotalPotentialElectro.set_zero(d_phiTotRhoOut);
        delta_total[0][0] = delta_phiOutPlanarValues[0][0];
        delta_total[1][0] = delta_phiOutPlanarValues[1][0];
        delta_total[0][1] =
          delta_phiOutPlanarValues[0][1] + delta_pseudLocPlanarValues[0][1];
        delta_total[1][1] =
          delta_phiOutPlanarValues[1][1] + delta_pseudLocPlanarValues[1][1];
        if (d_dftParamsPtr->applyExternalPotential &&
            (d_dftParamsPtr->externalPotentialType == "CEF"))
          {
            distributedCPUVec<double> uExt;
            d_matrixFreeDataPRefined.initialize_dof_vector(
              uExt, d_baseDofHandlerIndexElectro);
            uExt = 0;
            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsPRefined,
                                    d_baseDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_uExtQuadValuesRho,
                                    uExt);
            d_constraintsPRefined.distribute(uExt);
            uExt.update_ghost_values();
            computePlanarAverageField(uExt,
                                      d_matrixFreeDataPRefined,
                                      d_baseDofHandlerIndexElectro,
                                      mpi_communicator,
                                      interpoolcomm,
                                      uExtPlanarValues);
            computeDeltaPlanarAverageField(uExt,
                                           d_matrixFreeDataPRefined,
                                           d_baseDofHandlerIndexElectro,
                                           mpi_communicator,
                                           interpoolcomm,
                                           delta_uExtPlanarValues);
            delta_total[0][1] += delta_uExtPlanarValues[0][1];
            delta_total[1][1] += delta_uExtPlanarValues[1][1];
            for (int i = 0; i < phiOutPlanarValues.size(); i++)
              {
                phiOutPlanarValues[i][1] +=
                  pseudLocPlanarValues[i][1] + uExtPlanarValues[i][1];
              }
          }
        else
          {
            for (int i = 0; i < phiOutPlanarValues.size(); i++)
              {
                phiOutPlanarValues[i][1] += pseudLocPlanarValues[i][1];
              }
          }
        std::vector<std::string> labels = {"Z-Coordinate",
                                           "planar avg. bare Potential(Ha/e)"};
        dftfe::dftUtils::writeDataIntoFile(labels,
                                           phiOutPlanarValues,
                                           "planarAverageBarePotential.txt",
                                           d_mpiCommParent);
        pcout << "------------------------------" << std::endl;
        pcout << "Vtot at Left Boundary: " << delta_total[0][0]
              << " is: " << delta_total[0][1] << std::endl;
        pcout << "Vtot at Right Boundary: " << delta_total[1][0]
              << " is: " << delta_total[1][1] << std::endl;
        pcout << "Delta Vtot: " << (delta_total[1][1] - delta_total[0][1])
              << std::endl;
        pcout << "------------------------------" << std::endl;
      }

    if (d_dftParamsPtr->writeDensityQuadData)
      writeGSElectronDensity("densityQuadData.txt");

    if (d_dftParamsPtr->writeDosFile)
      compute_tdos(eigenValues, "dosData.out");

    if (d_dftParamsPtr->writeLdosFile)
      compute_ldos(eigenValues, "ldosData.out");

    if (d_dftParamsPtr->writeLocalizationLengths)
      compute_localizationLength("localizationLengths.out");

    if (d_dftParamsPtr->applyExternalPotential)
      computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                              d_densityQuadratureIdElectro,
                              d_densityOutQuadValues[0],
                              &d_bQuadValuesAllAtoms);

    if (d_dftParamsPtr->writeOutputTotalElectrostaticsPotential)
      outputPotential();

    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << std::endl
        << "------------------DFT-FE ground-state solve completed---------------------------"
        << std::endl;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::trivialSolveForStress()
  {
    initBoundaryConditions();
    noRemeshRhoDataInit();
    solve(false, true);
  }


  //
  // initialize
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initializeKohnShamDFTOperator(
    const bool initializeCublas)
  {
    dealii::TimerOutput::Scope scope(computing_timer,
                                     "kohnShamDFTOperator init");
    double                     init_ksoperator;
    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime();

    if (d_kohnShamDFTOperatorsInitialized)
      finalizeKohnShamDFTOperator();


#ifdef DFTFE_WITH_DEVICE
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      d_kohnShamDFTOperatorPtr =
        new KohnShamDFTStandardEigenOperator<memorySpace>(
          d_BLASWrapperPtr,
          d_basisOperationsPtrDevice,
          d_basisOperationsPtrHost,
          d_oncvClassPtr,
          d_excManagerPtr,
          d_dftParamsPtr,
          d_densityQuadratureId,
          d_lpspQuadratureId,
          d_feOrderPlusOneQuadratureId,
          d_mpiCommParent,
          mpi_communicator);
    else
#endif
      d_kohnShamDFTOperatorPtr =
        new KohnShamDFTStandardEigenOperator<memorySpace>(
          d_BLASWrapperPtrHost,
          d_basisOperationsPtrHost,
          d_basisOperationsPtrHost,
          d_oncvClassPtr,
          d_excManagerPtr,
          d_dftParamsPtr,
          d_densityQuadratureId,
          d_lpspQuadratureId,
          d_feOrderPlusOneQuadratureId,
          d_mpiCommParent,
          mpi_communicator);


    KohnShamDFTBaseOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    kohnShamDFTEigenOperator.init(d_kPointCoordinates, d_kPointWeights);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        AssertThrow(
          (d_numEigenValues % d_dftParamsPtr->chebyWfcBlockSize == 0 ||
           d_numEigenValues / d_dftParamsPtr->chebyWfcBlockSize == 0),
          dealii::ExcMessage(
            "DFT-FE Error: total number wavefunctions must be exactly divisible by cheby wfc block size for Device run."));


        AssertThrow(
          (d_numEigenValues % d_dftParamsPtr->wfcBlockSize == 0 ||
           d_numEigenValues / d_dftParamsPtr->wfcBlockSize == 0),
          dealii::ExcMessage(
            "DFT-FE Error: total number wavefunctions must be exactly divisible by wfc block size for Device run."));

        AssertThrow(
          (d_dftParamsPtr->wfcBlockSize % d_dftParamsPtr->chebyWfcBlockSize ==
             0 &&
           d_dftParamsPtr->wfcBlockSize / d_dftParamsPtr->chebyWfcBlockSize >=
             0),
          dealii::ExcMessage(
            "DFT-FE Error: wfc block size must be exactly divisible by cheby wfc block size and also larger for Device run."));


        // AssertThrow(
        //   (d_numEigenValuesRR % d_dftParamsPtr->wfcBlockSize == 0 ||
        //    d_numEigenValuesRR / d_dftParamsPtr->wfcBlockSize == 0),
        //   dealii::ExcMessage(
        //     "DFT-FE Error: total number RR wavefunctions must be exactly
        //     divisible by wfc block size for Device run."));

        // band group parallelization data structures
        const dftfe::uInt numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

        AssertThrow(
          (d_numEigenValues % numberBandGroups == 0 ||
           d_numEigenValues / numberBandGroups == 0),
          dealii::ExcMessage(
            "DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for Device run."));

        const dftfe::uInt bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, d_numEigenValues, bandGroupLowHighPlusOneIndices);

        AssertThrow(
          (bandGroupLowHighPlusOneIndices[1] %
             d_dftParamsPtr->chebyWfcBlockSize ==
           0),
          dealii::ExcMessage(
            "DFT-FE Error: band parallelization group size must be exactly divisible by CHEBY WFC BLOCK SIZE for Device run."));

        AssertThrow(
          (bandGroupLowHighPlusOneIndices[1] % d_dftParamsPtr->wfcBlockSize ==
           0),
          dealii::ExcMessage(
            "DFT-FE Error: band parallelization group size must be exactly divisible by WFC BLOCK SIZE for Device run."));
      }
#endif


    d_kohnShamDFTOperatorsInitialized = true;

    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime() - init_ksoperator;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "init: Time taken for kohnShamDFTOperator class initialization: "
            << init_ksoperator << std::endl;
  }


  //
  // re-initialize (significantly cheaper than initialize)
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::reInitializeKohnShamDFTOperator()
  {
    d_kohnShamDFTOperatorPtr->resetKohnShamOp();
  }

  //
  // finalize
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::finalizeKohnShamDFTOperator()
  {
    if (d_kohnShamDFTOperatorsInitialized)
      {
        delete d_kohnShamDFTOperatorPtr;
        d_kohnShamDFTOperatorsInitialized = false;
      }
  }

  //
  // dft solve
  //
  template <dftfe::utils::MemorySpace memorySpace>
  std::tuple<bool, double>
  dftClass<memorySpace>::solve(const bool computeForces,
                               const bool computestress,
                               const bool isRestartGroundStateCalcFromChk,
                               const bool computeExternalPotentialFlag)
  {
    if (d_dftParamsPtr->applyExternalPotential &&
        (d_dftParamsPtr->externalPotentialType == "CEF"))
      {
        computeUExtPotentialAtDensityQuadPoints();
      }


    KohnShamDFTBaseOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;


    // computingTimerStandard.enter_subsection("Total scf solve");
    energyCalculator<memorySpace> energyCalc(d_mpiCommParent,
                                             mpi_communicator,
                                             interpoolcomm,
                                             interBandGroupComm,
                                             *d_dftParamsPtr);


    // set up linear solver
    dealiiLinearSolver CGSolver(d_mpiCommParent,
                                mpi_communicator,
                                dealiiLinearSolver::CG);

    // set up linear solver Device
#ifdef DFTFE_WITH_DEVICE
    linearSolverCGDevice CGSolverDevice(d_mpiCommParent,
                                        mpi_communicator,
                                        linearSolverCGDevice::CG,
                                        d_BLASWrapperPtr);
#endif

    //
    // set up solver functions for Helmholtz to be used only when Kerker mixing
    // is on use higher polynomial order dofHandler
    //
    kerkerSolverProblemWrapperClass kerkerPreconditionedResidualSolverProblem(
      d_dftParamsPtr->finiteElementPolynomialOrderRhoNodal,
      d_mpiCommParent,
      mpi_communicator);

    // set up solver functions for Helmholtz Device
#ifdef DFTFE_WITH_DEVICE
    kerkerSolverProblemDeviceWrapperClass
      kerkerPreconditionedResidualSolverProblemDevice(
        d_dftParamsPtr->finiteElementPolynomialOrderRhoNodal,
        d_mpiCommParent,
        mpi_communicator);
#endif

    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
      {
        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges)
          {
#ifdef DFTFE_WITH_DEVICE
            kerkerPreconditionedResidualSolverProblemDevice.init(
              d_basisOperationsPtrElectroDevice,
              d_constraintsRhoNodal,
              d_preCondTotalDensityResidualVector,
              d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ?
                d_dftParamsPtr->kerkerParameter :
                (d_dftParamsPtr->restaFermiWavevector / 4.0 / M_PI / 4.0 /
                 M_PI),
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro);
#endif
          }
        else
          kerkerPreconditionedResidualSolverProblem.init(
            d_basisOperationsPtrElectroHost,
            d_constraintsRhoNodal,
            d_preCondTotalDensityResidualVector,
            d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ?
              d_dftParamsPtr->kerkerParameter :
              (d_dftParamsPtr->restaFermiWavevector / 4.0 / M_PI / 4.0 / M_PI),
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro);
      }

    // FIXME: Check if this call can be removed
    d_phiTotalSolverProblem.clear();

    //
    // solve vself in bins
    //
    computing_timer.enter_subsection("Nuclear self-potential solve");
    computingTimerStandard.enter_subsection("Nuclear self-potential solve");
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->vselfGPU)
      d_vselfBinsManager.solveVselfInBinsDevice(
        d_basisOperationsPtrElectroHost,
        d_baseDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_binsStartDofHandlerIndexElectro,
        d_dftParamsPtr->finiteElementPolynomialOrder ==
            d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics ?
          d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
          d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
        d_BLASWrapperPtr,
        d_constraintsPRefined,
        d_imagePositionsTrunc,
        d_imageIdsTrunc,
        d_imageChargesTrunc,
        d_localVselfs,
        d_bQuadValuesAllAtoms,
        d_bQuadAtomIdsAllAtoms,
        d_bQuadAtomIdsAllAtomsImages,
        d_bCellNonTrivialAtomIds,
        d_bCellNonTrivialAtomIdsBins,
        d_bCellNonTrivialAtomImageIds,
        d_bCellNonTrivialAtomImageIdsBins,
        d_smearedChargeWidths,
        d_smearedChargeScaling,
        d_smearedChargeQuadratureIdElectro,
        d_dftParamsPtr->smearedNuclearCharges);
    else
      d_vselfBinsManager.solveVselfInBins(
        d_basisOperationsPtrElectroHost,
        d_binsStartDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_constraintsPRefined,
        d_imagePositionsTrunc,
        d_imageIdsTrunc,
        d_imageChargesTrunc,
        d_localVselfs,
        d_bQuadValuesAllAtoms,
        d_bQuadAtomIdsAllAtoms,
        d_bQuadAtomIdsAllAtomsImages,
        d_bCellNonTrivialAtomIds,
        d_bCellNonTrivialAtomIdsBins,
        d_bCellNonTrivialAtomImageIds,
        d_bCellNonTrivialAtomImageIdsBins,
        d_smearedChargeWidths,
        d_smearedChargeScaling,
        d_smearedChargeQuadratureIdElectro,
        d_dftParamsPtr->smearedNuclearCharges);
#else
    d_vselfBinsManager.solveVselfInBins(d_basisOperationsPtrElectroHost,
                                        d_binsStartDofHandlerIndexElectro,
                                        d_phiTotAXQuadratureIdElectro,
                                        d_constraintsPRefined,
                                        d_imagePositionsTrunc,
                                        d_imageIdsTrunc,
                                        d_imageChargesTrunc,
                                        d_localVselfs,
                                        d_bQuadValuesAllAtoms,
                                        d_bQuadAtomIdsAllAtoms,
                                        d_bQuadAtomIdsAllAtomsImages,
                                        d_bCellNonTrivialAtomIds,
                                        d_bCellNonTrivialAtomIdsBins,
                                        d_bCellNonTrivialAtomImageIds,
                                        d_bCellNonTrivialAtomImageIdsBins,
                                        d_smearedChargeWidths,
                                        d_smearedChargeScaling,
                                        d_smearedChargeQuadratureIdElectro,
                                        d_dftParamsPtr->smearedNuclearCharges);
#endif

    if (d_dftParamsPtr->applyExternalPotential &&
        (d_dftParamsPtr->externalPotentialType == "CEF"))
      {
        computeUExtPotentialAtNuclearQuadPoints(d_bCellNonTrivialAtomIds);
      }
    computingTimerStandard.leave_subsection("Nuclear self-potential solve");
    computing_timer.leave_subsection("Nuclear self-potential solve");

    if ((d_dftParamsPtr->isPseudopotential ||
         d_dftParamsPtr->smearedNuclearCharges))
      {
        computingTimerStandard.enter_subsection("Init local PSP");
        initLocalPseudoPotential(d_dofHandlerPRefined,
                                 d_lpspQuadratureIdElectro,
                                 d_matrixFreeDataPRefined,
                                 d_phiExtDofHandlerIndexElectro,
                                 d_constraintsPRefinedOnlyHanging,
                                 d_supportPointsPRefined,
                                 d_vselfBinsManager,
                                 d_phiExt,
                                 d_pseudoVLoc,
                                 d_pseudoVLocAtoms);
        if (d_dftParamsPtr->applyExternalPotential &&
            (d_dftParamsPtr->externalPotentialType == "CPD-APD") &&
            d_dftParamsPtr->includeVselfInConstraints)
          {
            distributedCPUVec<double> pseudoVLocNodal;
            d_matrixFreeDataPRefined.initialize_dof_vector(
              pseudoVLocNodal, d_phiTotDofHandlerIndexElectro);
            pseudoVLocNodal = 0;
            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsPRefined,
                                    d_phiExtDofHandlerIndexElectro,
                                    d_lpspQuadratureIdElectro,
                                    d_pseudoVLoc,
                                    pseudoVLocNodal);
            pseudoVLocNodal.update_ghost_values();
            d_constraintsPRefinedOnlyHanging.distribute(pseudoVLocNodal);
            d_constraintsForTotalPotentialElectro.clear();
            d_constraintsForTotalPotentialElectro.reinit(
              d_locallyRelevantDofsPRefined);
            std::vector<double> nodalFieldVector(pseudoVLocNodal.size(), 0.0);
            for (dftfe::uInt iDof = 0;
                 iDof < pseudoVLocNodal.locally_owned_size();
                 iDof++)
              {
                dftfe::uInt globalId =
                  pseudoVLocNodal.get_partitioner()->local_to_global(iDof);
                nodalFieldVector[globalId] =
                  pseudoVLocNodal.local_element(iDof);
              }
            MPI_Allreduce(MPI_IN_PLACE,
                          &nodalFieldVector[0],
                          nodalFieldVector.size(),
                          dataTypes::mpi_type_id(&nodalFieldVector[0]),
                          MPI_SUM,
                          mpi_communicator);
            modifyConstrainedNodesWithVself(
              d_dftParamsPtr->potentialValueL,
              d_dftParamsPtr->potentialValueR,
              d_dofHandlerPRefined,
              d_constraintsPRefined,
              nodalFieldVector,
              d_constraintsForTotalPotentialElectro);
            MPI_Barrier(mpi_communicator);
            d_constraintsForTotalPotentialElectro.close();
            d_constraintsForTotalPotentialElectro.merge(
              d_constraintsPRefined,
              dealii::AffineConstraints<
                double>::MergeConflictBehavior::right_object_wins);
            d_constraintsForTotalPotentialElectro.close();
          }
        else if (d_dftParamsPtr->applyExternalPotential &&
                 (d_dftParamsPtr->externalPotentialType == "CPD-PPD") &&
                 d_dftParamsPtr->includeVselfInConstraints)
          {
            distributedCPUVec<double> pseudoVLocNodal;
            d_matrixFreeDataPRefined.initialize_dof_vector(
              pseudoVLocNodal, d_phiTotDofHandlerIndexElectro);
            pseudoVLocNodal = 0;
            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsPRefined,
                                    d_phiExtDofHandlerIndexElectro,
                                    d_lpspQuadratureIdElectro,
                                    d_pseudoVLoc,
                                    pseudoVLocNodal);
            pseudoVLocNodal.update_ghost_values();
            d_constraintsPRefinedOnlyHanging.distribute(pseudoVLocNodal);
            d_constraintsForTotalPotentialElectro.clear();
            d_constraintsForTotalPotentialElectro.reinit(
              d_locallyRelevantDofsPRefined);
            std::vector<double> nodalFieldVector(pseudoVLocNodal.size(), 0.0);
            for (dftfe::uInt iDof = 0;
                 iDof < pseudoVLocNodal.locally_owned_size();
                 iDof++)
              {
                dftfe::uInt globalId =
                  pseudoVLocNodal.get_partitioner()->local_to_global(iDof);
                nodalFieldVector[globalId] =
                  pseudoVLocNodal.local_element(iDof);
              }
            MPI_Allreduce(MPI_IN_PLACE,
                          &nodalFieldVector[0],
                          nodalFieldVector.size(),
                          dataTypes::mpi_type_id(&nodalFieldVector[0]),
                          MPI_SUM,
                          mpi_communicator);
            modifyConstrainedNodesWithVself(
              d_dftParamsPtr->potentialValueL,
              d_dftParamsPtr->potentialValueR,
              d_dofHandlerPRefined,
              d_constraintsPRefined,
              nodalFieldVector,
              d_constraintsForTotalPotentialElectro);
            MPI_Barrier(mpi_communicator);
            d_constraintsForTotalPotentialElectro.close();
            d_constraintsForTotalPotentialElectro.merge(
              d_constraintsPRefined,
              dealii::AffineConstraints<
                double>::MergeConflictBehavior::right_object_wins);
            d_constraintsForTotalPotentialElectro.close();
          }

        kohnShamDFTEigenOperator.computeVEffExternalPotCorr(d_pseudoVLoc);
        if (d_dftParamsPtr->applyExternalPotential &&
            (d_dftParamsPtr->externalPotentialType == "CEF"))
          kohnShamDFTEigenOperator.computeVEffAppliedExternalPotCorr(
            d_uExtQuadValuesRho);
        computingTimerStandard.leave_subsection("Init local PSP");
      }


    computingTimerStandard.enter_subsection("Total scf solve");

    //
    // solve
    //
    computing_timer.enter_subsection("scf solve");

    double firstScfChebyTol =
      d_dftParamsPtr->restrictToOnePass ?
        1e+4 :
        (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
             d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA" ?
           1e-2 :
           2e-2);


    if (d_dftParamsPtr->solverMode == "MD")
      firstScfChebyTol = d_dftParamsPtr->chebyshevTolerance > 1e-4 ?
                           1e-4 :
                           d_dftParamsPtr->chebyshevTolerance;
    else if (d_dftParamsPtr->solverMode == "GEOOPT")
      firstScfChebyTol = d_dftParamsPtr->chebyshevTolerance > 1e-3 ?
                           1e-3 :
                           d_dftParamsPtr->chebyshevTolerance;

    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);
    // call the mixing scheme with the mixing variables
    // Have to be called once for each variable
    // initialise the variables in the mixing scheme
    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          rhoNodalMassVec;
        computeRhoNodalMassVector(rhoNodalMassVec);
        d_mixingScheme.addMixingVariable(
          mixingVariable::rho,
          rhoNodalMassVec,
          true, // call MPI REDUCE while computing dot products
          d_dftParamsPtr->mixingParameter,
          d_dftParamsPtr->adaptAndersonMixingParameter);
        if (d_dftParamsPtr->spinPolarized == 1)
          d_mixingScheme.addMixingVariable(
            mixingVariable::magZ,
            rhoNodalMassVec,
            true, // call MPI REDUCE while computing dot products
            d_dftParamsPtr->mixingParameter *
              d_dftParamsPtr->spinMixingEnhancementFactor,
            d_dftParamsPtr->adaptAndersonMixingParameter);
      }
    else if (d_dftParamsPtr->mixingMethod == "ANDERSON")
      {
        d_basisOperationsPtrElectroHost->reinit(0,
                                                0,
                                                d_densityQuadratureIdElectro,
                                                false);
        d_mixingScheme.addMixingVariable(
          mixingVariable::rho,
          d_basisOperationsPtrElectroHost->JxWBasisData(),
          true, // call MPI REDUCE while computing dot products
          d_dftParamsPtr->mixingParameter,
          d_dftParamsPtr->adaptAndersonMixingParameter);
        if (d_dftParamsPtr->spinPolarized == 1)
          d_mixingScheme.addMixingVariable(
            mixingVariable::magZ,
            d_basisOperationsPtrElectroHost->JxWBasisData(),
            true, // call MPI REDUCE while computing dot products
            d_dftParamsPtr->mixingParameter *
              d_dftParamsPtr->spinMixingEnhancementFactor,
            d_dftParamsPtr->adaptAndersonMixingParameter);
        if (isGradDensityDataDependent)
          {
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              gradRhoJxW;
            gradRhoJxW.resize(0);
            d_mixingScheme.addMixingVariable(
              mixingVariable::gradRho,
              gradRhoJxW, // this is just a dummy variable to make it
                          // compatible with rho
              false,      // call MPI REDUCE while computing dot products
              d_dftParamsPtr->mixingParameter,
              d_dftParamsPtr->adaptAndersonMixingParameter);
            if (d_dftParamsPtr->spinPolarized == 1)
              d_mixingScheme.addMixingVariable(
                mixingVariable::gradMagZ,
                gradRhoJxW,
                false, // call MPI REDUCE while computing dot products
                d_dftParamsPtr->mixingParameter *
                  d_dftParamsPtr->spinMixingEnhancementFactor,
                d_dftParamsPtr->adaptAndersonMixingParameter);
          }

        if (isTauMGGA)
          {
            d_mixingScheme.addMixingVariable(
              mixingVariable::tau,
              d_basisOperationsPtrElectroHost->JxWBasisData(),
              true,
              d_dftParamsPtr->mixingParameter *
                d_dftParamsPtr->spinMixingEnhancementFactor,
              d_dftParamsPtr->adaptAndersonMixingParameter);
            if (d_dftParamsPtr->spinPolarized == 1)
              {
                d_mixingScheme.addMixingVariable(
                  mixingVariable::tauMagZ,
                  d_basisOperationsPtrElectroHost->JxWBasisData(),
                  true,
                  d_dftParamsPtr->mixingParameter *
                    d_dftParamsPtr->spinMixingEnhancementFactor,
                  d_dftParamsPtr->adaptAndersonMixingParameter);
              }
          }


        if (d_useHubbard)
          {
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              hubbOccJxW;
            hubbOccJxW.resize(0);
            d_mixingScheme.addMixingVariable(
              mixingVariable::hubbardOccupation,
              hubbOccJxW,
              false,
              d_dftParamsPtr->mixingParameter,
              d_dftParamsPtr->adaptAndersonMixingParameter);
          }
      }

    if (d_dftParamsPtr->confiningPotential)
      {
        d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId);
        d_expConfiningPot.init(d_basisOperationsPtrHost,
                               *d_dftParamsPtr,
                               atomLocations);
      }
    //
    // Begin SCF iteration
    //
    dftfe::uInt scfIter                   = 0;
    double      norm                      = 1.0;
    double      energyResidual            = 1.0;
    d_rankCurrentLRD                      = 0;
    d_relativeErrorJacInvApproxPrevScfLRD = 100.0;
    // CAUTION: Choosing a looser tolerance might lead to failed tests
    const double adaptiveChebysevFilterPassesTol =
      d_dftParamsPtr->chebyshevTolerance;
    bool scfConverged = false;
    pcout << std::endl;
    if (d_dftParamsPtr->verbosity == 0)
      pcout << "Starting SCF iterations...." << std::endl;
    while (!scfConverged && (scfIter < d_dftParamsPtr->numSCFIterations))
      {
        dealii::Timer local_timer(d_mpiCommParent, true);
        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************"
            << std::endl;
        //
        // Mixing scheme
        //
        computing_timer.enter_subsection("density mixing");
        if (scfIter > 0)
          {
            if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  norm =
                    lowrankApproxScfDielectricMatrixInvSpinPolarized(scfIter);
                else
                  norm = lowrankApproxScfDielectricMatrixInv(scfIter);
                if (d_dftParamsPtr->verbosity >= 1)
                  pcout << d_dftParamsPtr->mixingMethod
                        << " mixing, L2 norm of electron-density difference: "
                        << norm << std::endl;
              }
            else if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
                     d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
              {
                // Fill in New Kerker framework here
                std::vector<double> norms(
                  d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
                if (scfIter == 1)
                  d_densityResidualNodalValues.resize(
                    d_densityOutNodalValues.size());
                for (dftfe::uInt iComp = 0;
                     iComp < d_densityOutNodalValues.size();
                     ++iComp)
                  {
                    norms[iComp] = computeResidualNodalData(
                      d_densityOutNodalValues[iComp],
                      d_densityInNodalValues[iComp],
                      d_densityResidualNodalValues[iComp]);
                  }
                for (dftfe::uInt iComp = 0;
                     iComp < d_densityOutNodalValues.size();
                     ++iComp)
                  {
                    d_mixingScheme.addVariableToInHist(
                      iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                      d_densityInNodalValues[iComp].begin(),
                      d_densityInNodalValues[iComp].locally_owned_size());
                    d_mixingScheme.addVariableToResidualHist(
                      iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                      d_densityResidualNodalValues[iComp].begin(),
                      d_densityResidualNodalValues[iComp].locally_owned_size());
                  }
                // Delete old history if it exceeds a pre-described
                // length
                d_mixingScheme.popOldHistory(d_dftParamsPtr->mixingHistory);

                // Compute the mixing coefficients
                d_mixingScheme.computeAndersonMixingCoeff(
                  d_dftParamsPtr->spinPolarized == 1 ?
                    std::vector<mixingVariable>{mixingVariable::rho,
                                                mixingVariable::magZ} :
                    std::vector<mixingVariable>{mixingVariable::rho});
                d_mixingScheme.getOptimizedResidual(
                  mixingVariable::rho,
                  d_densityResidualNodalValues[0].begin(),
                  d_densityResidualNodalValues[0].locally_owned_size());
                applyKerkerPreconditionerToTotalDensityResidual(
#ifdef DFTFE_WITH_DEVICE
                  kerkerPreconditionedResidualSolverProblemDevice,
                  CGSolverDevice,
#endif
                  kerkerPreconditionedResidualSolverProblem,
                  CGSolver,
                  d_densityResidualNodalValues[0],
                  d_preCondTotalDensityResidualVector);
                d_mixingScheme.mixPreconditionedResidual(
                  mixingVariable::rho,
                  d_preCondTotalDensityResidualVector.begin(),
                  d_densityInNodalValues[0].begin(),
                  d_densityInNodalValues[0].locally_owned_size());
                for (dftfe::uInt iComp = 1; iComp < norms.size(); ++iComp)
                  d_mixingScheme.mixVariable(
                    iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                    d_densityInNodalValues[iComp].begin(),
                    d_densityInNodalValues[iComp].locally_owned_size());
                norm = 0.0;
                for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                  norm += norms[iComp] * norms[iComp];
                norm = std::sqrt(norm / ((double)norms.size()));
                // interpolate nodal data to quadrature data
                if (d_dftParamsPtr->verbosity >= 1)
                  for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                    pcout << d_dftParamsPtr->mixingMethod
                          << " mixing, L2 norm of "
                          << (iComp == 0 ? "electron" : "magnetization")
                          << "-density difference: " << norms[iComp]
                          << std::endl;
                for (dftfe::uInt iComp = 0;
                     iComp < d_densityInNodalValues.size();
                     ++iComp)
                  {
                    d_basisOperationsPtrElectroHost->interpolate(
                      d_densityInNodalValues[iComp],
                      d_densityDofHandlerIndexElectro,
                      d_densityQuadratureIdElectro,
                      d_densityInQuadValues[iComp],
                      d_gradDensityInQuadValues[iComp],
                      d_gradDensityInQuadValues[iComp],
                      isGradDensityDataDependent);
                  }
              }
            else if (d_dftParamsPtr->mixingMethod == "ANDERSON")
              {
                std::vector<double> norms(
                  d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
                std::vector<double> normsTau(
                  d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
                // Update the history of mixing variables
                if (scfIter == 1)
                  d_densityResidualQuadValues.resize(
                    d_densityOutQuadValues.size());
                for (dftfe::uInt iComp = 0;
                     iComp < d_densityOutQuadValues.size();
                     ++iComp)
                  {
                    if (scfIter == 1)
                      d_densityResidualQuadValues[iComp].resize(
                        d_densityOutQuadValues[iComp].size());
                    d_basisOperationsPtrElectroHost->reinit(
                      0, 0, d_densityQuadratureIdElectro, false);
                    norms[iComp] = computeResidualQuadData(
                      d_densityOutQuadValues[iComp],
                      d_densityInQuadValues[iComp],
                      d_densityResidualQuadValues[iComp],
                      d_basisOperationsPtrElectroHost->JxWBasisData(),
                      true);
                    d_mixingScheme.addVariableToInHist(
                      iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                      d_densityInQuadValues[iComp].data(),
                      d_densityInQuadValues[iComp].size());
                    d_mixingScheme.addVariableToResidualHist(
                      iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                      d_densityResidualQuadValues[iComp].data(),
                      d_densityResidualQuadValues[iComp].size());
                  }
                if (isGradDensityDataDependent)
                  {
                    if (scfIter == 1)
                      d_gradDensityResidualQuadValues.resize(
                        d_gradDensityOutQuadValues.size());
                    for (dftfe::uInt iComp = 0;
                         iComp < d_gradDensityResidualQuadValues.size();
                         ++iComp)
                      {
                        if (scfIter == 1)
                          d_gradDensityResidualQuadValues[iComp].resize(
                            d_gradDensityOutQuadValues[iComp].size());
                        computeResidualQuadData(
                          d_gradDensityOutQuadValues[iComp],
                          d_gradDensityInQuadValues[iComp],
                          d_gradDensityResidualQuadValues[iComp],
                          d_basisOperationsPtrElectroHost->JxWBasisData(),
                          false);
                        d_mixingScheme.addVariableToInHist(
                          iComp == 0 ? mixingVariable::gradRho :
                                       mixingVariable::gradMagZ,
                          d_gradDensityInQuadValues[iComp].data(),
                          d_gradDensityInQuadValues[iComp].size());
                        d_mixingScheme.addVariableToResidualHist(
                          iComp == 0 ? mixingVariable::gradRho :
                                       mixingVariable::gradMagZ,
                          d_gradDensityResidualQuadValues[iComp].data(),
                          d_gradDensityResidualQuadValues[iComp].size());
                      }
                  }

                if (isTauMGGA)
                  {
                    if (scfIter == 1)
                      {
                        d_tauResidualQuadValues.resize(
                          d_tauOutQuadValues.size());
                      }

                    for (dftfe::uInt iComp = 0;
                         iComp < d_tauOutQuadValues.size();
                         iComp++)
                      {
                        if (scfIter == 1)
                          d_tauResidualQuadValues[iComp].resize(
                            d_tauOutQuadValues[iComp].size());
                        d_basisOperationsPtrElectroHost->reinit(
                          0, 0, d_densityQuadratureIdElectro, false);
                        double normTau;
                        normTau = computeResidualQuadData(
                          d_tauOutQuadValues[iComp],
                          d_tauInQuadValues[iComp],
                          d_tauResidualQuadValues[iComp],
                          d_basisOperationsPtrElectroHost->JxWBasisData(),
                          true);

                        normsTau[iComp] = computeResidualQuadData(
                          d_tauOutQuadValues[iComp],
                          d_tauInQuadValues[iComp],
                          d_tauResidualQuadValues[iComp],
                          d_basisOperationsPtrElectroHost->JxWBasisData(),
                          true);
                        d_mixingScheme.addVariableToInHist(
                          iComp == 0 ? mixingVariable::tau :
                                       mixingVariable::tauMagZ,
                          d_tauInQuadValues[iComp].data(),
                          d_tauInQuadValues[iComp].size());
                        d_mixingScheme.addVariableToResidualHist(
                          iComp == 0 ? mixingVariable::tau :
                                       mixingVariable::tauMagZ,
                          d_tauResidualQuadValues[iComp].data(),
                          d_tauResidualQuadValues[iComp].size());
                      }
                  }

                if (d_useHubbard == true)
                  {
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &hubbOccIn = d_hubbardClassPtr->getOccMatIn();

                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &hubbOccRes = d_hubbardClassPtr->getOccMatRes();
                    d_mixingScheme.addVariableToInHist(
                      mixingVariable::hubbardOccupation,
                      hubbOccIn.data(),
                      hubbOccIn.size());

                    d_mixingScheme.addVariableToResidualHist(
                      mixingVariable::hubbardOccupation,
                      hubbOccRes.data(),
                      hubbOccRes.size());
                  }



                // Delete old history if it exceeds a pre-described
                // length
                d_mixingScheme.popOldHistory(d_dftParamsPtr->mixingHistory);

                // Compute the mixing coefficients
                d_mixingScheme.computeAndersonMixingCoeff(
                  d_dftParamsPtr->spinPolarized == 1 ?
                    (isTauMGGA ?
                       std::vector<mixingVariable>{mixingVariable::rho,
                                                   mixingVariable::tau,
                                                   mixingVariable::magZ,
                                                   mixingVariable::tauMagZ} :
                       std::vector<mixingVariable>{mixingVariable::rho,
                                                   mixingVariable::magZ}) :
                    (isTauMGGA ?
                       std::vector<mixingVariable>{mixingVariable::rho,
                                                   mixingVariable::tau} :
                       std::vector<mixingVariable>{mixingVariable::rho}));


                // update the mixing variables
                for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                  d_mixingScheme.mixVariable(
                    iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                    d_densityInQuadValues[iComp].data(),
                    d_densityInQuadValues[iComp].size());
                norm = 0.0;
                for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                  norm += norms[iComp] * norms[iComp];
                norm = std::sqrt(norm / ((double)norms.size()));
                if (isGradDensityDataDependent)
                  {
                    for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                      d_mixingScheme.mixVariable(
                        iComp == 0 ? mixingVariable::gradRho :
                                     mixingVariable::gradMagZ,
                        d_gradDensityInQuadValues[iComp].data(),
                        d_gradDensityInQuadValues[iComp].size());
                  }

                if (isTauMGGA)
                  {
                    for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                      {
                        d_mixingScheme.mixVariable(
                          iComp == 0 ? mixingVariable::tau :
                                       mixingVariable::tauMagZ,
                          d_tauInQuadValues[iComp].data(),
                          d_tauInQuadValues[iComp].size());
                      }
                  }


                if (d_useHubbard == true)
                  {
                    dftfe::utils::MemoryStorage<double,
                                                dftfe::utils::MemorySpace::HOST>
                      &hubbOccMatAfterMixing =
                        d_hubbardClassPtr->getHubbMatrixForMixing();

                    std::fill(hubbOccMatAfterMixing.begin(),
                              hubbOccMatAfterMixing.end(),
                              0.0);

                    d_mixingScheme.mixVariable(
                      mixingVariable::hubbardOccupation,
                      hubbOccMatAfterMixing.data(),
                      hubbOccMatAfterMixing.size());

                    d_hubbardClassPtr->setInOccMatrix(hubbOccMatAfterMixing);
                  }


                if (d_dftParamsPtr->verbosity >= 1)
                  for (dftfe::uInt iComp = 0; iComp < norms.size(); ++iComp)
                    {
                      pcout << d_dftParamsPtr->mixingMethod
                            << " mixing, L2 norm of "
                            << (iComp == 0 ? "electron" : "magnetization")
                            << "-density difference: " << norms[iComp]
                            << std::endl;

                      if (isTauMGGA)
                        {
                          pcout << d_dftParamsPtr->mixingMethod
                                << " mixing, L2 norm of "
                                << (iComp == 0 ?
                                      "Kinetic energy density" :
                                      "magnetization (Kinetic energy density)")
                                << " difference: " << normsTau[iComp]
                                << std::endl;
                        }
                    }
              }

            if (d_dftParamsPtr->verbosity >= 1 &&
                d_dftParamsPtr->spinPolarized == 1)
              pcout << d_dftParamsPtr->mixingMethod
                    << " mixing, L2 norm of total density difference: " << norm
                    << std::endl;
          }

        if (d_dftParamsPtr->computeEnergyEverySCF)
          d_phiTotRhoIn = d_phiTotRhoOut;
        computing_timer.leave_subsection("density mixing");

        if (!((norm > d_dftParamsPtr->selfConsistentSolverTolerance) ||
              (d_dftParamsPtr->useEnergyResidualTolerance &&
               energyResidual >
                 d_dftParamsPtr->selfConsistentSolverEnergyTolerance)))
          scfConverged = true;

        if (d_dftParamsPtr->multipoleBoundaryConditions)
          {
            computing_timer.enter_subsection("Update inhomogenous BC");
            computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                    d_densityQuadratureIdElectro,
                                    d_densityInQuadValues[0],
                                    &d_bQuadValuesAllAtoms);
            updatePRefinedConstraints();
            computing_timer.leave_subsection("Update inhomogenous BC");
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          densityInQuadValuesCopy = d_densityInQuadValues[0];
        if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
            (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
             d_dftParamsPtr->periodicZ))
          {
            double *tempvec = densityInQuadValuesCopy.data();
            for (dftfe::uInt iquad = 0; iquad < densityInQuadValuesCopy.size();
                 iquad++)
              tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
          }
        //
        // phiTot with rhoIn
        //
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << std::endl
            << "Poisson solve for total electrostatic potential (rhoIn+b): ";

        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
          {
#ifdef DFTFE_WITH_DEVICE
            if (scfIter > 0)
              d_phiTotalSolverProblemDevice.reinit(
                d_basisOperationsPtrElectroHost,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                densityInQuadValuesCopy,
                d_BLASWrapperPtr,
                false,
                false,
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                false,
                true,
                d_dftParamsPtr->multipoleBoundaryConditions);
            else
              {
                d_phiTotalSolverProblemDevice.reinit(
                  d_basisOperationsPtrElectroHost,
                  d_phiTotRhoIn,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  densityInQuadValuesCopy,
                  d_BLASWrapperPtr,
                  true,
                  (d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                   d_dftParamsPtr->periodicZ &&
                   !d_dftParamsPtr->pinnedNodeForPBC) ||
                    (d_dftParamsPtr->applyExternalPotential &&
                     d_dftParamsPtr->externalPotentialType == "CEF"),
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  true,
                  false,
                  true);
              }
#endif
          }
        else
          {
            if (scfIter > 0)
              d_phiTotalSolverProblem.reinit(
                d_basisOperationsPtrElectroHost,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                densityInQuadValuesCopy,
                false,
                false,
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                false,
                true,
                d_dftParamsPtr->multipoleBoundaryConditions);
            else
              d_phiTotalSolverProblem.reinit(
                d_basisOperationsPtrElectroHost,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                densityInQuadValuesCopy,
                true,
                (d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                 d_dftParamsPtr->periodicZ &&
                 !d_dftParamsPtr->pinnedNodeForPBC) ||
                  (d_dftParamsPtr->applyExternalPotential &&
                   d_dftParamsPtr->externalPotentialType == "CEF"),
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                true,
                false,
                true);
          }

        computing_timer.enter_subsection("phiTot solve");

        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
          {
#ifdef DFTFE_WITH_DEVICE
            CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                                 d_dftParamsPtr->absLinearSolverTolerance,
                                 d_dftParamsPtr->maxLinearSolverIterations,
                                 d_dftParamsPtr->verbosity);
#endif
          }
        else
          {
            CGSolver.solve(d_phiTotalSolverProblem,
                           d_dftParamsPtr->absLinearSolverTolerance,
                           d_dftParamsPtr->maxLinearSolverIterations,
                           d_dftParamsPtr->verbosity);
          }



        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          dummy;
        d_basisOperationsPtrElectroHost->interpolate(
          d_phiTotRhoIn,
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiInQuadValues,
          dummy,
          dummy);


        if (d_dftParamsPtr->confiningPotential)
          {
            d_expConfiningPot.addConfiningPotential(d_phiInQuadValues);
          }

        //
        // impose integral phi equals 0
        //
        /*
        if(d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
        d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC)
        {
          if (d_dftParamsPtr->verbosity>=2)
            pcout<<"Value of integPhiIn:
        "<<totalCharge(d_dofHandlerPRefined,d_phiTotRhoIn)<<std::endl;
        }
        */

        computing_timer.leave_subsection("phiTot solve");

        dftfe::uInt numberChebyshevSolvePasses = 0;
        //
        // eigen solve
        //
        std::vector<std::vector<std::vector<double>>> eigenValuesSpins(
          d_dftParamsPtr->spinPolarized + 1,
          std::vector<std::vector<double>>(
            d_kPointWeights.size(), std::vector<double>(d_numEigenValues)));

        std::vector<std::vector<std::vector<double>>>
          residualNormWaveFunctionsAllkPointsSpins(
            d_dftParamsPtr->spinPolarized + 1,
            std::vector<std::vector<double>>(
              d_kPointWeights.size(), std::vector<double>(d_numEigenValues)));

        updateAuxDensityXCMatrix(d_densityInQuadValues,
                                 d_gradDensityInQuadValues,
                                 d_tauInQuadValues,
                                 d_rhoCore,
                                 d_gradRhoCore,
                                 getEigenVectors(),
                                 eigenValues,
                                 fermiEnergy,
                                 fermiEnergyUp,
                                 fermiEnergyDown,
                                 d_auxDensityMatrixXCInPtr);

        dftfe::uInt       count = 0;
        const dftfe::uInt maxPasses =
          !scfConverged &&
              (scfIter == 0 ||
               d_dftParamsPtr->allowMultipleFilteringPassesAfterFirstScf) ?
            100 :
            1;
        // maximum of the residual norm of the state closest to and
        // below the Fermi level among all k points, and also the
        // maximum between the two spins
        std::vector<std::vector<double>> maxResidualsAllkPoints(
          d_dftParamsPtr->spinPolarized + 1);
        std::vector<double> maxResSpins(d_dftParamsPtr->spinPolarized + 1, 0.0);
        double              maxRes = 1.0;

        // if the residual norm is greater than
        // adaptiveChebysevFilterPassesTol (a heuristic value)
        // do more passes of chebysev filter till the check passes.
        // This improves the scf convergence performance.

        const double filterPassTol =
          (scfIter == 0 && isRestartGroundStateCalcFromChk) ?
            1.0e-8 :
            ((scfIter == 0 &&
              adaptiveChebysevFilterPassesTol > firstScfChebyTol) ?
               firstScfChebyTol :
               adaptiveChebysevFilterPassesTol);
        while (maxRes > filterPassTol && count < maxPasses)
          {
            for (dftfe::uInt s = 0; s < d_dftParamsPtr->spinPolarized + 1; ++s)
              {
                if ((d_dftParamsPtr->memOptMode &&
                     d_dftParamsPtr->spinPolarized == 1) ||
                    count == 0)
                  {
                    computing_timer.enter_subsection("VEff Computation");
                    kohnShamDFTEigenOperator.computeVEff(
                      d_auxDensityMatrixXCInPtr, d_phiInQuadValues, s);
                    computing_timer.leave_subsection("VEff Computation");
                  }
                for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size();
                     ++kPoint)
                  {
                    if (d_dftParamsPtr->verbosity >= 4 && count > 0)
                      pcout
                        << "Maximum residual norm of the state closest to and below Fermi level for kpoint "
                        << kPoint << " spin " << s << ":"
                        << maxResidualsAllkPoints[s][kPoint] << std::endl;
                    if (count == 0 ||
                        maxResidualsAllkPoints[s][kPoint] > filterPassTol)
                      {
                        if (d_dftParamsPtr->verbosity >= 2)
                          pcout << "Beginning Chebyshev filter pass "
                                << 1 + count << " for spin " << s + 1
                                << std::endl;

                        kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,
                                                                       s);
                        if (d_dftParamsPtr->memOptMode || count == 0)
                          {
                            computing_timer.enter_subsection(
                              "Hamiltonian Matrix Computation");
                            kohnShamDFTEigenOperator
                              .computeCellHamiltonianMatrix();
                            computing_timer.leave_subsection(
                              "Hamiltonian Matrix Computation");
                          }

#ifdef DFTFE_WITH_DEVICE
                        if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                      memorySpace)
                          kohnShamEigenSpaceCompute(
                            s,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolverDevice,
                            residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                            maxPasses > 1,
                            0,
                            true,
                            scfIter == 0);
#endif
                        if constexpr (dftfe::utils::MemorySpace::HOST ==
                                      memorySpace)
                          kohnShamEigenSpaceCompute(
                            s,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolver,
                            residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                            maxPasses > 1,
                            true,
                            scfIter == 0);
                      }
                  }
              }
            for (dftfe::uInt s = 0; s < d_dftParamsPtr->spinPolarized + 1; ++s)
              for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size();
                   ++kPoint)
                {
                  for (dftfe::uInt i = 0; i < d_numEigenValues; ++i)
                    eigenValuesSpins[s][kPoint][i] =
                      eigenValues[kPoint][d_numEigenValues * s + i];
                }
            //
            if (d_dftParamsPtr->constraintMagnetization)
              {
                if (d_dftParamsPtr->pureState)
                  compute_fermienergy_constraintMagnetization_purestate(
                    eigenValues);
                else
                  compute_fermienergy_constraintMagnetization(eigenValues);
              }
            else
              {
                if (d_dftParamsPtr->pureState)
                  compute_fermienergy_purestate(eigenValues, numElectrons);
                else
                  compute_fermienergy(eigenValues, numElectrons);
              }

            for (dftfe::uInt s = 0; s < d_dftParamsPtr->spinPolarized + 1; ++s)
              {
                maxResSpins[s] = computeMaximumHighestOccupiedStateResidualNorm(
                  residualNormWaveFunctionsAllkPointsSpins[s],
                  eigenValuesSpins[s],
                  fermiEnergy,
                  maxResidualsAllkPoints[s]);
              }
            maxRes = *std::max_element(maxResSpins.begin(), maxResSpins.end());
            if (d_dftParamsPtr->verbosity >= 2)
              pcout
                << "Maximum residual norm among all states with occupation number greater than 1e-3: "
                << maxRes << std::endl;
            count++;
          }

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
          }

        numberChebyshevSolvePasses = count;
        computing_timer.enter_subsection("compute rho");
        if (d_dftParamsPtr->useSymm)
          {
#ifdef USE_COMPLEX
            symmetryPtr->computeLocalrhoOut();
            symmetryPtr->computeAndSymmetrize_rhoOut();

            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_densityOutQuadValues[0],
                                    d_densityOutNodalValues[0]);

            d_basisOperationsPtrElectroHost->interpolate(
              d_densityOutNodalValues[0],
              d_densityDofHandlerIndexElectro,
              d_lpspQuadratureIdElectro,
              d_densityTotalOutValuesLpspQuad,
              d_gradDensityTotalOutValuesLpspQuad,
              d_gradDensityTotalOutValuesLpspQuad,
              true);
#endif
          }
        else
          {
            compute_rhoOut(scfConverged ||
                           (scfIter == (d_dftParamsPtr->numSCFIterations - 1)));
          }
        computing_timer.leave_subsection("compute rho");

        //
        // compute integral rhoOut
        //
        const double integralRhoValue =
          totalCharge(d_dofHandlerPRefined, d_densityOutQuadValues[0]);

        if (d_dftParamsPtr->verbosity >= 2)
          {
            pcout << std::endl
                  << "number of electrons: " << integralRhoValue << std::endl;
          }
        if (d_dftParamsPtr->verbosity > 0 && d_dftParamsPtr->spinPolarized == 1)
          totalMagnetization(d_densityOutQuadValues[1]);

        //
        // phiTot with rhoOut
        //
        if (d_dftParamsPtr->computeEnergyEverySCF ||
            d_dftParamsPtr->useEnergyResidualTolerance)
          {
            if (d_dftParamsPtr->verbosity >= 2)
              pcout
                << std::endl
                << "Poisson solve for total electrostatic potential (rhoOut+b): ";

            computing_timer.enter_subsection("phiTot solve");

            if (d_dftParamsPtr->multipoleBoundaryConditions)
              {
                computing_timer.enter_subsection("Update inhomogenous BC");
                computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                        d_densityQuadratureIdElectro,
                                        d_densityOutQuadValues[0],
                                        &d_bQuadValuesAllAtoms);
                updatePRefinedConstraints();
                computing_timer.leave_subsection("Update inhomogenous BC");
              }

            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              densityOutQuadValuesCopy = d_densityOutQuadValues[0];
            if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
                (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
                 d_dftParamsPtr->periodicZ))
              {
                double *tempvec = densityOutQuadValuesCopy.data();
                for (dftfe::uInt iquad = 0;
                     iquad < densityOutQuadValuesCopy.size();
                     iquad++)
                  tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
              }

            if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
                d_dftParamsPtr->floatingNuclearCharges and
                not d_dftParamsPtr->pinnedNodeForPBC)
              {
#ifdef DFTFE_WITH_DEVICE
                d_phiTotalSolverProblemDevice.reinit(
                  d_basisOperationsPtrElectroHost,
                  d_phiTotRhoOut,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  densityOutQuadValuesCopy,
                  d_BLASWrapperPtr,
                  false,
                  false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  false,
                  true,
                  d_dftParamsPtr->multipoleBoundaryConditions);

                CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                                     d_dftParamsPtr->absLinearSolverTolerance,
                                     d_dftParamsPtr->maxLinearSolverIterations,
                                     d_dftParamsPtr->verbosity);
#endif
              }
            else
              {
                d_phiTotalSolverProblem.reinit(
                  d_basisOperationsPtrElectroHost,
                  d_phiTotRhoOut,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  densityOutQuadValuesCopy,
                  false,
                  false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  false,
                  true,
                  d_dftParamsPtr->multipoleBoundaryConditions);

                CGSolver.solve(d_phiTotalSolverProblem,
                               d_dftParamsPtr->absLinearSolverTolerance,
                               d_dftParamsPtr->maxLinearSolverIterations,
                               d_dftParamsPtr->verbosity);
              }

            d_basisOperationsPtrElectroHost->interpolate(
              d_phiTotRhoOut,
              d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiOutQuadValues,
              dummy,
              dummy);

            computing_timer.leave_subsection("phiTot solve");
          }

        updateAuxDensityXCMatrix(d_densityOutQuadValues,
                                 d_gradDensityOutQuadValues,
                                 d_tauOutQuadValues,
                                 d_rhoCore,
                                 d_gradRhoCore,
                                 getEigenVectors(),
                                 eigenValues,
                                 fermiEnergy,
                                 fermiEnergyUp,
                                 fermiEnergyDown,
                                 d_auxDensityMatrixXCOutPtr);

        if (d_dftParamsPtr->useEnergyResidualTolerance)
          {
            computing_timer.enter_subsection("Energy residual computation");
            energyResidual = energyCalc.computeEnergyResidual(
              d_basisOperationsPtrHost,
              d_basisOperationsPtrElectroHost,
              d_densityQuadratureId,
              d_densityQuadratureIdElectro,
              d_smearedChargeQuadratureIdElectro,
              d_lpspQuadratureIdElectro,
              d_excManagerPtr,
              d_phiInQuadValues,
              d_phiOutQuadValues,
              d_phiTotRhoIn,
              d_phiTotRhoOut,
              d_densityInQuadValues,
              d_densityOutQuadValues,
              d_gradDensityInQuadValues,
              d_gradDensityOutQuadValues,
              d_tauInQuadValues,
              d_tauOutQuadValues,
              d_auxDensityMatrixXCInPtr,
              d_auxDensityMatrixXCOutPtr,
              d_bQuadValuesAllAtoms,
              d_bCellNonTrivialAtomIds,
              d_localVselfs,
              d_atomNodeIdToChargeMap,
              d_dftParamsPtr->smearedNuclearCharges);
            if (d_dftParamsPtr->verbosity >= 1)
              pcout << "Energy residual  : " << energyResidual << std::endl;
            if (d_dftParamsPtr->reproducible_output)
              pcout << "Energy residual  : " << std::setprecision(4)
                    << energyResidual << std::endl;
            computing_timer.leave_subsection("Energy residual computation");
          }
        if (d_dftParamsPtr->computeEnergyEverySCF)
          {
            d_dispersionCorr.computeDispresionCorrection(
              atomLocations, d_domainBoundingVectors);
            const double totalEnergy = energyCalc.computeEnergy(
              d_basisOperationsPtrHost,
              d_basisOperationsPtrElectroHost,
              d_densityQuadratureId,
              d_densityQuadratureIdElectro,
              d_smearedChargeQuadratureIdElectro,
              d_lpspQuadratureIdElectro,
              eigenValues,
              d_partialOccupancies,
              d_kPointWeights,
              fermiEnergy,
              d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyUp,
              d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy :
                                                   fermiEnergyDown,
              d_excManagerPtr,
              d_dispersionCorr,
              d_phiInQuadValues,
              d_phiOutQuadValues,
              d_phiTotRhoOut,
              dummy,
              d_densityInQuadValues,
              d_densityOutQuadValues,
              d_gradDensityOutQuadValues,
              d_tauInQuadValues,
              d_tauOutQuadValues,
              d_densityTotalOutValuesLpspQuad,
              d_auxDensityMatrixXCInPtr,
              d_auxDensityMatrixXCOutPtr,
              d_bQuadValuesAllAtoms,
              d_bCellNonTrivialAtomIds,
              d_localVselfs,
              d_pseudoVLoc,
              d_atomNodeIdToChargeMap,
              atomLocations.size(),
              lowerBoundKindex,
              0,
              d_dftParamsPtr->verbosity >= 0 ? true : false,
              d_uExtQuadValuesNuclear,
              d_uExtQuadValuesRho,
              d_dftParamsPtr->smearedNuclearCharges,
              d_dftParamsPtr->applyExternalPotential &&
                (d_dftParamsPtr->externalPotentialType == "CEF"));
            if (d_dftParamsPtr->verbosity == 1)
              pcout << "Total energy  : " << totalEnergy << std::endl;
          }



        d_excManagerPtr->getExcSSDFunctionalObj()
          ->updateWaveFunctionDependentFuncDerWrtPsi(d_auxDensityMatrixXCOutPtr,
                                                     d_kPointWeights);


        d_excManagerPtr->getExcSSDFunctionalObj()
          ->computeWaveFunctionDependentExcEnergy(d_auxDensityMatrixXCOutPtr,
                                                  d_kPointWeights);

        if (d_dftParamsPtr->verbosity >= 1)
          pcout << "***********************Self-Consistent-Field Iteration: "
                << std::setw(2) << scfIter + 1
                << " complete**********************" << std::endl;

        local_timer.stop();
        if (d_dftParamsPtr->verbosity >= 1)
          pcout << "Wall time for the above scf iteration: "
                << local_timer.wall_time() << " seconds\n"
                << "Number of Chebyshev filtered subspace iterations: "
                << numberChebyshevSolvePasses << std::endl
                << std::endl;
        //
        scfIter++;

        // if (d_dftParamsPtr->saveRhoData && scfIter % 10 == 0 &&
        //     d_dftParamsPtr->solverMode == "GS")
        //   {
        //     saveTriaInfoAndRhoNodalData();
        //     if (d_useHubbard)
        //       {
        //         d_hubbardClassPtr->writeHubbOccToFile();
        //       }
        //   }
        if (d_dftParamsPtr->saveQuadData && scfIter % 10 == 0 &&
            d_dftParamsPtr->solverMode == "GS")
          {
            std::vector<std::string> field     = {"RHO", "MAG_Z"};
            std::vector<std::string> Gradfield = {"gradRHO", "gradMAG_Z"};
            std::vector<std::string> field2    = {"TAU", "TAUMAG_Z"};
            for (dftfe::Int i = 0; i < d_densityOutQuadValues.size(); i++)
              {
                saveQuadratureData(d_basisOperationsPtrHost,
                                   d_densityQuadratureId,
                                   d_densityOutQuadValues[i],
                                   1,
                                   field[i],
                                   d_dftParamsPtr->restartFolder,
                                   d_mpiCommParent,
                                   mpi_communicator,
                                   interpoolcomm,
                                   interBandGroupComm);
                bool isGradDensityDataDependent =
                  (d_excManagerPtr->getExcSSDFunctionalObj()
                     ->getDensityBasedFamilyType() == densityFamilyType::GGA);
                if (isGradDensityDataDependent)
                  {
                    saveQuadratureData(d_basisOperationsPtrHost,
                                       d_densityQuadratureId,
                                       d_gradDensityOutQuadValues[i],
                                       3,
                                       Gradfield[i],
                                       d_dftParamsPtr->restartFolder,
                                       d_mpiCommParent,
                                       mpi_communicator,
                                       interpoolcomm,
                                       interBandGroupComm);
                  }
              }
            if (isTauMGGA)
              for (dftfe::Int i = 0; i < d_tauOutQuadValues.size(); i++)
                {
                  saveQuadratureData(d_basisOperationsPtrHost,
                                     d_densityQuadratureId,
                                     d_tauOutQuadValues[i],
                                     1,
                                     field2[i],
                                     d_dftParamsPtr->restartFolder,
                                     d_mpiCommParent,
                                     mpi_communicator,
                                     interpoolcomm,
                                     interBandGroupComm);
                }
            if (d_useHubbard)
              {
                d_hubbardClassPtr->writeHubbOccToFile();
              }
          }
      }

    // if (d_dftParamsPtr->saveRhoData &&
    //     !(d_dftParamsPtr->solverMode == "GS" && scfIter % 10 == 0))
    //   {
    //     saveTriaInfoAndRhoNodalData();
    //     if (d_useHubbard)
    //       {
    //         d_hubbardClassPtr->writeHubbOccToFile();
    //       }
    //   }
    if (d_dftParamsPtr->saveQuadData &&
        !(d_dftParamsPtr->solverMode == "GS" && scfIter % 10 == 0))
      {
        std::vector<std::string> field     = {"RHO", "MAG_Z"};
        std::vector<std::string> Gradfield = {"gradRHO", "gradMAG_Z"};
        std::vector<std::string> field2    = {"TAU", "TAUMAG_Z"};
        for (dftfe::Int i = 0; i < d_densityOutQuadValues.size(); i++)
          {
            saveQuadratureData(d_basisOperationsPtrHost,
                               d_densityQuadratureId,
                               d_densityOutQuadValues[i],
                               1,
                               field[i],
                               d_dftParamsPtr->restartFolder,
                               d_mpiCommParent,
                               mpi_communicator,
                               interpoolcomm,
                               interBandGroupComm);
            bool isGradDensityDataDependent =
              (d_excManagerPtr->getExcSSDFunctionalObj()
                 ->getDensityBasedFamilyType() == densityFamilyType::GGA);
            if (isGradDensityDataDependent)
              {
                saveQuadratureData(d_basisOperationsPtrHost,
                                   d_densityQuadratureId,
                                   d_gradDensityOutQuadValues[i],
                                   3,
                                   Gradfield[i],
                                   d_dftParamsPtr->restartFolder,
                                   d_mpiCommParent,
                                   mpi_communicator,
                                   interpoolcomm,
                                   interBandGroupComm);
              }
          }
        if (isTauMGGA)
          for (int i = 0; i < d_tauOutQuadValues.size(); i++)
            {
              saveQuadratureData(d_basisOperationsPtrHost,
                                 d_densityQuadratureId,
                                 d_tauOutQuadValues[i],
                                 1,
                                 field2[i],
                                 d_dftParamsPtr->restartFolder,
                                 d_mpiCommParent,
                                 mpi_communicator,
                                 interpoolcomm,
                                 interBandGroupComm);
            }
        if (d_useHubbard)
          {
            d_hubbardClassPtr->writeHubbOccToFile();
          }
      }



    if (scfIter == d_dftParamsPtr->numSCFIterations)
      {
        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          std::cout
            << "DFT-FE Warning: SCF iterations did not converge to the specified tolerance after: "
            << scfIter << " iterations." << std::endl;
      }
    else
      {
        pcout << "SCF iterations converged to the specified tolerance after: "
              << scfIter << " iterations." << std::endl;
        if (d_dftParamsPtr->verbosity >= 1)
          {
            if (d_dftParamsPtr->spinPolarized &&
                d_dftParamsPtr->constraintMagnetization)
              {
                pcout << "GS Fermi energy spin up: " << fermiEnergyUp
                      << std::endl;
                pcout << "GS Fermi energy spin down: " << fermiEnergyDown
                      << std::endl;
              }
            else
              pcout << "GS Fermi energy spin up: " << fermiEnergy << std::endl;
          }

        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          {
            if (d_dftParamsPtr->solverMode == "GS" &&
                (d_dftParamsPtr->saveQuadData))
              {
                FILE *fermiFile;
                fermiFile = fopen("fermiEnergy.out", "w");
                if (d_dftParamsPtr->spinPolarized)
                  {
                    fprintf(fermiFile,
                            "%.14g\n%.14g\n%.14g\n ",
                            fermiEnergy,
                            fermiEnergyUp,
                            fermiEnergyDown);
                  }
                else
                  {
                    fprintf(fermiFile, "%.14g\n", fermiEnergy);
                  }
                fclose(fermiFile);
              }
          }
      }

    updateAuxDensityXCMatrix(d_densityOutQuadValues,
                             d_gradDensityOutQuadValues,
                             d_tauOutQuadValues,
                             d_rhoCore,
                             d_gradRhoCore,
                             getEigenVectors(),
                             eigenValues,
                             fermiEnergy,
                             fermiEnergyUp,
                             fermiEnergyDown,
                             d_auxDensityMatrixXCOutPtr);

    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

    const dftfe::uInt localVectorSize =
      matrix_free_data.get_vector_partitioner()->locally_owned_size();
    if (numberBandGroups > 1 && !d_dftParamsPtr->useDevice)
      {
        MPI_Barrier(interBandGroupComm);
        const dftfe::uInt blockSize =
          d_dftParamsPtr->mpiAllReduceMessageBlockSizeMB * 1e+6 /
          sizeof(dataTypes::number);
        for (dftfe::uInt kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          for (dftfe::uInt i = 0; i < d_numEigenValues * localVectorSize;
               i += blockSize)
            {
              const dftfe::uInt currentBlockSize =
                std::min(blockSize, d_numEigenValues * localVectorSize - i);
              MPI_Allreduce(
                MPI_IN_PLACE,
                &d_eigenVectorsFlattenedHost[kPoint * d_numEigenValues *
                                             localVectorSize] +
                  i,
                currentBlockSize,
                dataTypes::mpi_type_id(
                  &d_eigenVectorsFlattenedHost[kPoint * d_numEigenValues *
                                               localVectorSize]),
                MPI_SUM,
                interBandGroupComm);
            }
      }

    if ((!d_dftParamsPtr->computeEnergyEverySCF))
      {
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << std::endl
            << "Poisson solve for total electrostatic potential (rhoOut+b): ";

        computing_timer.enter_subsection("phiTot solve");

        if (d_dftParamsPtr->multipoleBoundaryConditions)
          {
            computing_timer.enter_subsection("Update inhomogenous BC");
            computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                    d_densityQuadratureIdElectro,
                                    d_densityOutQuadValues[0],
                                    &d_bQuadValuesAllAtoms);
            updatePRefinedConstraints();
            computing_timer.leave_subsection("Update inhomogenous BC");
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          densityOutQuadValuesCopy = d_densityOutQuadValues[0];
        if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
            (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
             d_dftParamsPtr->periodicZ))
          {
            double *tempvec = densityOutQuadValuesCopy.data();
            for (dftfe::uInt iquad = 0; iquad < densityOutQuadValuesCopy.size();
                 iquad++)
              tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
          }

        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
          {
#ifdef DFTFE_WITH_DEVICE
            d_phiTotalSolverProblemDevice.reinit(
              d_basisOperationsPtrElectroHost,
              d_phiTotRhoOut,
              *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
              d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
              d_atomNodeIdToChargeMap,
              d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
              densityOutQuadValuesCopy,
              d_BLASWrapperPtr,
              false,
              false,
              d_dftParamsPtr->smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true,
              d_dftParamsPtr->multipoleBoundaryConditions);

            CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                                 d_dftParamsPtr->absLinearSolverTolerance,
                                 d_dftParamsPtr->maxLinearSolverIterations,
                                 d_dftParamsPtr->verbosity);
#endif
          }
        else
          {
            d_phiTotalSolverProblem.reinit(
              d_basisOperationsPtrElectroHost,
              d_phiTotRhoOut,
              *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
              d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
              d_atomNodeIdToChargeMap,
              d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
              densityOutQuadValuesCopy,
              false,
              false,
              d_dftParamsPtr->smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true,
              d_dftParamsPtr->multipoleBoundaryConditions);

            CGSolver.solve(d_phiTotalSolverProblem,
                           d_dftParamsPtr->absLinearSolverTolerance,
                           d_dftParamsPtr->maxLinearSolverIterations,
                           d_dftParamsPtr->verbosity);
          }

        computing_timer.leave_subsection("phiTot solve");
      }
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      phiOutGradValues, dummy;


    d_basisOperationsPtrElectroHost->interpolate(d_phiTotRhoOut,
                                                 d_phiTotDofHandlerIndexElectro,
                                                 d_densityQuadratureIdElectro,
                                                 d_phiOutQuadValues,
                                                 phiOutGradValues,
                                                 dummy,
                                                 true);


    //
    // compute and print ground state energy or energy after max scf
    // iterations
    //
    d_dispersionCorr.computeDispresionCorrection(atomLocations,
                                                 d_domainBoundingVectors);
    const double totalEnergy = energyCalc.computeEnergy(
      d_basisOperationsPtrHost,
      d_basisOperationsPtrElectroHost,
      d_densityQuadratureId,
      d_densityQuadratureIdElectro,
      d_smearedChargeQuadratureIdElectro,
      d_lpspQuadratureIdElectro,
      eigenValues,
      d_partialOccupancies,
      d_kPointWeights,
      fermiEnergy,
      d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyUp,
      d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyDown,
      d_excManagerPtr,
      d_dispersionCorr,
      d_phiInQuadValues,
      d_phiOutQuadValues,
      d_phiTotRhoOut,
      phiOutGradValues,
      d_densityInQuadValues,
      d_densityOutQuadValues,
      d_gradDensityOutQuadValues,
      d_tauInQuadValues,
      d_tauOutQuadValues,
      d_densityTotalOutValuesLpspQuad,
      d_auxDensityMatrixXCInPtr,
      d_auxDensityMatrixXCOutPtr,
      d_bQuadValuesAllAtoms,
      d_bCellNonTrivialAtomIds,
      d_localVselfs,
      d_pseudoVLoc,
      d_atomNodeIdToChargeMap,
      atomLocations.size(),
      lowerBoundKindex,
      1,
      d_dftParamsPtr->verbosity >= 0 ? true : false,
      d_uExtQuadValuesNuclear,
      d_uExtQuadValuesRho,
      d_dftParamsPtr->smearedNuclearCharges,
      d_dftParamsPtr->applyExternalPotential &&
        (d_dftParamsPtr->externalPotentialType == "CEF"));

    d_groundStateEnergy = totalEnergy;

    MPI_Barrier(interpoolcomm);

    d_entropicEnergy =
      energyCalc.computeEntropicEnergy(eigenValues,
                                       d_partialOccupancies,
                                       d_kPointWeights,
                                       fermiEnergy,
                                       fermiEnergyUp,
                                       fermiEnergyDown,
                                       d_dftParamsPtr->spinPolarized == 1,
                                       d_dftParamsPtr->constraintMagnetization,
                                       d_dftParamsPtr->TVal);

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total entropic energy: " << d_entropicEnergy << std::endl;


    d_freeEnergy = d_groundStateEnergy - d_entropicEnergy;

    if (d_dftParamsPtr->verbosity >= 1)
      {
        if ((d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
             ExcFamilyType::DFTPlusU) ||
            (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
             ExcFamilyType::HYBRID) ||
            (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
             ExcFamilyType::MGGA))
          {
            pcout << " Non local part of Exc energy = "
                  << d_excManagerPtr->getExcSSDFunctionalObj()
                       ->getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
                  << "\n";
          }
      }
    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total free energy: " << d_freeEnergy << std::endl;

    if (d_dftParamsPtr->verbosity >= 0 && d_dftParamsPtr->spinPolarized == 1)
      totalMagnetization(d_densityOutQuadValues[1]);

    // This step is required for interpolating rho from current mesh to
    // the new mesh in case of atomic relaxation
    // computeNodalRhoFromQuadData();

    computing_timer.leave_subsection("scf solve");
    computingTimerStandard.leave_subsection("Total scf solve");


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice &&
        (d_dftParamsPtr->writeWfcSolutionFields ||
         d_dftParamsPtr->writeLdosFile || d_dftParamsPtr->writePdosFile))
      d_eigenVectorsFlattenedDevice.copyTo(d_eigenVectorsFlattenedHost);
#endif


    if (d_dftParamsPtr->isIonForce)
      {
        if (d_dftParamsPtr->selfConsistentSolverTolerance > 1e-4 &&
            d_dftParamsPtr->verbosity >= 1)
          pcout
            << "DFT-FE Warning: Ion force accuracy may be affected for the given scf iteration solve tolerance: "
            << d_dftParamsPtr->selfConsistentSolverTolerance
            << ", recommended to use TOLERANCE below 1e-4." << std::endl;

        if (computeForces)
          {
            computing_timer.enter_subsection("Ion force computation");
            computingTimerStandard.enter_subsection("Ion force computation");
            forcePtr->computeAtomsForces(
              matrix_free_data,
              d_dispersionCorr,
              d_eigenDofHandlerIndex,
              d_smearedChargeQuadratureIdElectro,
              d_lpspQuadratureIdElectro,
              d_matrixFreeDataPRefined,
              d_phiTotDofHandlerIndexElectro,
              d_phiTotRhoOut,
              d_densityOutQuadValues,
              d_gradDensityOutQuadValues,
              d_densityTotalOutValuesLpspQuad,
              d_gradDensityTotalOutValuesLpspQuad,
              d_rhoCore,
              d_gradRhoCore,
              d_hessianRhoCore,
              d_gradRhoCoreAtoms,
              d_hessianRhoCoreAtoms,
              d_pseudoVLoc,
              d_pseudoVLocAtoms,
              d_constraintsPRefined,
              d_vselfBinsManager,
              atomLocations,
              d_dftParamsPtr->applyExternalPotential &&
                (d_dftParamsPtr->externalPotentialType == "CEF"),
              d_dftParamsPtr->externalPotentialSlope);
            if (d_dftParamsPtr->verbosity >= 0)
              forcePtr->printAtomsForces();
            computingTimerStandard.leave_subsection("Ion force computation");
            computing_timer.leave_subsection("Ion force computation");
          }
      }

    if (d_dftParamsPtr->isCellStress)
      {
        if (d_dftParamsPtr->selfConsistentSolverTolerance > 1e-4 &&
            d_dftParamsPtr->verbosity >= 1)
          pcout
            << "DFT-FE Warning: Cell stress accuracy may be affected for the given scf iteration solve tolerance: "
            << d_dftParamsPtr->selfConsistentSolverTolerance
            << ", recommended to use TOLERANCE below 1e-4." << std::endl;

        if (computestress)
          {
            computing_timer.enter_subsection("Cell stress computation");
            computingTimerStandard.enter_subsection("Cell stress computation");
            computeStress();
            computingTimerStandard.leave_subsection("Cell stress computation");
            computing_timer.leave_subsection("Cell stress computation");
          }
      }
    return std::make_tuple(scfConverged, norm);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeStress()
  {
    KohnShamDFTBaseOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    if (d_dftParamsPtr->isPseudopotential ||
        d_dftParamsPtr->smearedNuclearCharges)
      {
        computeVselfFieldGateauxDerFD();
      }

    forcePtr->computeStress(matrix_free_data,
                            d_dispersionCorr,
                            d_eigenDofHandlerIndex,
                            d_smearedChargeQuadratureIdElectro,
                            d_lpspQuadratureIdElectro,
                            d_matrixFreeDataPRefined,
                            d_phiTotDofHandlerIndexElectro,
                            d_phiTotRhoOut,
                            d_densityOutQuadValues,
                            d_gradDensityOutQuadValues,
                            d_densityTotalOutValuesLpspQuad,
                            d_gradDensityTotalOutValuesLpspQuad,
                            d_pseudoVLoc,
                            d_pseudoVLocAtoms,
                            d_rhoCore,
                            d_gradRhoCore,
                            d_hessianRhoCore,
                            d_gradRhoCoreAtoms,
                            d_hessianRhoCoreAtoms,
                            d_constraintsPRefined,
                            d_vselfBinsManager);
    if (d_dftParamsPtr->verbosity >= 0)
      forcePtr->printStress();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeVselfFieldGateauxDerFD()
  {
    d_vselfFieldGateauxDerStrainFDBins.clear();
    d_vselfFieldGateauxDerStrainFDBins.resize(
      (d_vselfBinsManager.getVselfFieldBins()).size() * 6);

    dealii::Tensor<2, 3, double> identityTensor;
    dealii::Tensor<2, 3, double> deformationGradientPerturb1;
    dealii::Tensor<2, 3, double> deformationGradientPerturb2;

    // initialize to indentity tensors
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
        {
          if (idim == jdim)
            {
              identityTensor[idim][jdim]              = 1.0;
              deformationGradientPerturb1[idim][jdim] = 1.0;
              deformationGradientPerturb2[idim][jdim] = 1.0;
            }
          else
            {
              identityTensor[idim][jdim]              = 0.0;
              deformationGradientPerturb1[idim][jdim] = 0.0;
              deformationGradientPerturb2[idim][jdim] = 0.0;
            }
        }

    const double fdparam          = 1e-5;
    dftfe::uInt  flattenedIdCount = 0;
    for (dftfe::uInt idim = 0; idim < 3; ++idim)
      for (dftfe::uInt jdim = 0; jdim <= idim; jdim++)
        {
          deformationGradientPerturb1 = identityTensor;
          if (idim == jdim)
            {
              deformationGradientPerturb1[idim][jdim] = 1.0 + fdparam;
            }
          else
            {
              deformationGradientPerturb1[idim][jdim] = fdparam;
              deformationGradientPerturb1[jdim][idim] = fdparam;
            }

          deformDomain(deformationGradientPerturb1 *
                         dealii::invert(deformationGradientPerturb2),
                       true,
                       false,
                       d_dftParamsPtr->verbosity >= 4 ? true : false);

          computing_timer.enter_subsection(
            "Nuclear self-potential perturbation solve");

          d_vselfBinsManager.solveVselfInBinsPerturbedDomain(
            d_basisOperationsPtrElectroHost,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
#ifdef DFTFE_WITH_DEVICE
            d_dftParamsPtr->finiteElementPolynomialOrder ==
                d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics ?
              d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
              d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
            d_BLASWrapperPtr,
#endif
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_smearedChargeWidths,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);

          computing_timer.leave_subsection(
            "Nuclear self-potential perturbation solve");

          for (dftfe::uInt ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] =
              (d_vselfBinsManager.getPerturbedVselfFieldBins())[ibin];

          deformationGradientPerturb2 = identityTensor;
          if (idim == jdim)
            {
              deformationGradientPerturb2[idim][jdim] = 1.0 - fdparam;
            }
          else
            {
              deformationGradientPerturb2[idim][jdim] = -fdparam;
              deformationGradientPerturb2[jdim][idim] = -fdparam;
            }

          deformDomain(deformationGradientPerturb2 *
                         dealii::invert(deformationGradientPerturb1),
                       true,
                       false,
                       d_dftParamsPtr->verbosity >= 4 ? true : false);

          computing_timer.enter_subsection(
            "Nuclear self-potential perturbation solve");

          d_vselfBinsManager.solveVselfInBinsPerturbedDomain(
            d_basisOperationsPtrElectroHost,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
#ifdef DFTFE_WITH_DEVICE
            d_dftParamsPtr->finiteElementPolynomialOrder ==
                d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics ?
              d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
              d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
            d_BLASWrapperPtr,
#endif
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_smearedChargeWidths,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);

          computing_timer.leave_subsection(
            "Nuclear self-potential perturbation solve");

          for (dftfe::uInt ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] -=
              (d_vselfBinsManager.getPerturbedVselfFieldBins())[ibin];

          const double fac =
            (idim == jdim) ? (1.0 / 2.0 / fdparam) : (1.0 / 4.0 / fdparam);
          for (dftfe::uInt ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] *=
              fac;

          flattenedIdCount++;
        }

    // reset
    deformDomain(dealii::invert(deformationGradientPerturb2),
                 true,
                 false,
                 d_dftParamsPtr->verbosity >= 4 ? true : false);
  }

  // Output wfc
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::outputWfc(const std::string outputFileName)
  {
    //
    // identify the index which is close to Fermi Energy
    //
    dftfe::Int indexFermiEnergy = -1.0;
    for (dftfe::Int spinType = 0; spinType < 1 + d_dftParamsPtr->spinPolarized;
         ++spinType)
      {
        for (dftfe::Int i = 0; i < d_numEigenValues; ++i)
          {
            if (eigenValues[0][spinType * d_numEigenValues + i] >= fermiEnergy)
              {
                if (i > indexFermiEnergy)
                  {
                    indexFermiEnergy = i;
                    break;
                  }
              }
          }
      }

    //
    // create a range of wavefunctions to output the wavefunction files
    //
    dftfe::Int startingRange = 0;
    dftfe::Int endingRange   = d_numEigenValues;

    /*
    dftfe::Int startingRange = indexFermiEnergy - 4;
    dftfe::Int endingRange   = indexFermiEnergy + 4;

    dftfe::Int startingRangeSpin = startingRange;

    for (dftfe::Int spinType = 0; spinType < 1 + d_dftParamsPtr->spinPolarized;
         ++spinType)
      {
        for (dftfe::Int i = indexFermiEnergy - 5; i > 0; --i)
          {
            if (std::abs(eigenValues[0][spinType * d_numEigenValues +
                                        (indexFermiEnergy - 4)] -
                         eigenValues[0][spinType * d_numEigenValues + i])
    <= 5e-04)
              {
                if (spinType == 0)
                  startingRange -= 1;
                else
                  startingRangeSpin -= 1;
              }
          }
      }


    if (startingRangeSpin < startingRange)
      startingRange = startingRangeSpin;
    */
    dftfe::Int numStatesOutput = (endingRange - startingRange) + 1;


    dealii::DataOut<3> data_outEigen;
    data_outEigen.attach_dof_handler(dofHandlerEigen);

    std::vector<distributedCPUVec<double>> tempVec(1);
    tempVec[0].reinit(d_tempEigenVec);

    std::vector<distributedCPUVec<double>> visualizeWaveFunctions(
      d_kPointWeights.size() * (1 + d_dftParamsPtr->spinPolarized) *
      numStatesOutput);

    dftfe::uInt count = 0;
    for (dftfe::uInt s = 0; s < 1 + d_dftParamsPtr->spinPolarized; ++s)
      for (dftfe::uInt k = 0; k < d_kPointWeights.size(); ++k)
        for (dftfe::uInt i = startingRange; i < endingRange; ++i)
          {
#ifdef USE_COMPLEX
            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedHost.data() +
                (k * (1 + d_dftParamsPtr->spinPolarized) + s) *
                  d_numEigenValues *
                  matrix_free_data.get_vector_partitioner()
                    ->locally_owned_size(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              std::make_pair(i, i + 1),
              localProc_dof_indicesReal,
              localProc_dof_indicesImag,
              tempVec);
#else
            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedHost.data() +
                (k * (1 + d_dftParamsPtr->spinPolarized) + s) *
                  d_numEigenValues *
                  matrix_free_data.get_vector_partitioner()
                    ->locally_owned_size(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              std::make_pair(i, i + 1),
              tempVec);
#endif
            tempVec[0].update_ghost_values();
            constraintsNoneEigenDataInfo.distribute(tempVec[0]);
            visualizeWaveFunctions[count] = tempVec[0];

            if (d_dftParamsPtr->spinPolarized == 1)
              data_outEigen.add_data_vector(visualizeWaveFunctions[count],
                                            "wfc_spin" + std::to_string(s) +
                                              "_kpoint" + std::to_string(k) +
                                              "_" + std::to_string(i));
            else
              data_outEigen.add_data_vector(visualizeWaveFunctions[count],
                                            "wfc_kpoint" + std::to_string(k) +
                                              "_" + std::to_string(i));

            count += 1;
          }

    data_outEigen.set_flags(dealii::DataOutBase::VtkFlags(
      std::numeric_limits<double>::min(),
      std::numeric_limits<dftfe::uInt>::min(),
      true,
      dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::
        best_speed, // This flag is version dependent for dealII 9.5.0 it
                    // is
                    // dealii::DataOutBase::CompressionLevel::best_speed
      true));       // higher order cells set to true
    data_outEigen.build_patches(d_dftParamsPtr->finiteElementPolynomialOrder);

    std::string tempFolder = "waveFunctionOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(dofHandlerEigen,
                                               data_outEigen,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               outputFileName);
    //"wfcOutput_"+std::to_string(k)+"_"+std::to_string(i));
  }


  // Output density
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::outputDensity()
  {
    //
    // compute nodal electron-density from quad data
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                            d_constraintsRhoNodal,
                            d_densityDofHandlerIndexElectro,
                            d_densityQuadratureIdElectro,
                            d_densityOutQuadValues[0],
                            rhoNodalField);
    d_constraintsRhoNodal.distribute(rhoNodalField);
    rhoNodalField.update_ghost_values();
    distributedCPUVec<double> magNodalField;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        magNodalField.reinit(rhoNodalField);
        magNodalField = 0;
        l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                d_constraintsRhoNodal,
                                d_densityDofHandlerIndexElectro,
                                d_densityQuadratureIdElectro,
                                d_densityOutQuadValues[1],
                                magNodalField);
      }

    //
    // only generate output for electron-density
    //
    dealii::DataOut<3> dataOutRho;
    dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
    dataOutRho.add_data_vector(rhoNodalField, std::string("chargeDensity"));
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        dataOutRho.add_data_vector(magNodalField, std::string("magDensity"));
      }
    dataOutRho.set_flags(dealii::DataOutBase::VtkFlags(
      std::numeric_limits<double>::min(),
      std::numeric_limits<dftfe::uInt>::min(),
      true,
      dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::
        best_speed, // This flag is version dependent for dealII 9.5.0 it
                    // is
                    // dealii::DataOutBase::CompressionLevel::best_speed
      true));       // higher order cells set to true
    dataOutRho.build_patches(d_dftParamsPtr->finiteElementPolynomialOrder);

    std::string tempFolder = "densityOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
                                               dataOutRho,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "densityOutput");
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::writeBands()
  {
    int                 numkPoints = d_kPointWeights.size();
    std::vector<double> eigenValuesFlattened;
    //
    for (dftfe::uInt kPoint = 0; kPoint < numkPoints; ++kPoint)
      for (dftfe::uInt iWave = 0;
           iWave < d_numEigenValues * (1 + d_dftParamsPtr->spinPolarized);
           ++iWave)
        eigenValuesFlattened.push_back(eigenValues[kPoint][iWave]);
    //
    //
    //
    dftfe::Int totkPoints =
      dealii::Utilities::MPI::sum(numkPoints, interpoolcomm);
    std::vector<int> numkPointsArray(d_dftParamsPtr->npool),
      mpi_offsets(d_dftParamsPtr->npool, 0);
    std::vector<double> eigenValuesFlattenedGlobal(
      totkPoints * d_numEigenValues * (1 + d_dftParamsPtr->spinPolarized), 0.0);
    //
    MPI_Gather(&numkPoints,
               1,
               dftfe::dataTypes::mpi_type_id(&numkPoints),
               &(numkPointsArray[0]),
               1,
               dftfe::dataTypes::mpi_type_id(numkPointsArray.data()),
               0,
               interpoolcomm);
    //
    numkPointsArray[0] = d_numEigenValues *
                         (1 + d_dftParamsPtr->spinPolarized) *
                         numkPointsArray[0];
    for (dftfe::uInt ipool = 1; ipool < d_dftParamsPtr->npool; ++ipool)
      {
        numkPointsArray[ipool] = d_numEigenValues *
                                 (1 + d_dftParamsPtr->spinPolarized) *
                                 numkPointsArray[ipool];
        mpi_offsets[ipool] =
          mpi_offsets[ipool - 1] + numkPointsArray[ipool - 1];
      }
    //
    MPI_Gatherv(&(eigenValuesFlattened[0]),
                numkPoints * d_numEigenValues *
                  (1 + d_dftParamsPtr->spinPolarized),
                MPI_DOUBLE,
                &(eigenValuesFlattenedGlobal[0]),
                &(numkPointsArray[0]),
                &(mpi_offsets[0]),
                MPI_DOUBLE,
                0,
                interpoolcomm);
    //
    if (d_dftParamsPtr->reproducible_output && d_dftParamsPtr->verbosity == 0)
      {
        pcout << "Writing Bands File..." << std::endl;
        pcout << "K-Point   WaveNo.  ";
        if (d_dftParamsPtr->spinPolarized)
          pcout << "SpinUpEigenValue          SpinDownEigenValue" << std::endl;
        else
          pcout << "EigenValue" << std::endl;
      }

    std::ifstream file("fermiEnergy.out");
    std::string   line;

    if (file.is_open())
      {
        if (d_dftParamsPtr->constraintMagnetization)
          {
            std::vector<double> temp;
            while (getline(file, line))
              {
                if (!line.empty())
                  {
                    std::istringstream iss(line);
                    double             temp1;
                    while (iss >> temp1)
                      {
                        temp.push_back(temp1);
                      }
                  }
              }
            fermiEnergy     = temp[0];
            fermiEnergyUp   = temp[1];
            fermiEnergyDown = temp[2];
          }
        else
          {
            getline(file, line);
            std::istringstream iss(line);
            iss >> fermiEnergy;
          }
      }
    else
      {
        pcout << "Unable to open file fermiEnergy.out. Check if it is present.";
      }
    double FE = fermiEnergy;
    pcout << "Fermi Energy: " << FE << std::endl;
    dftfe::uInt         maxeigenIndex = d_numEigenValues;
    std::vector<double> occupationVector(totkPoints *
                                           (1 + d_dftParamsPtr->spinPolarized),
                                         0.0);

    for (dftfe::Int iWave = 1; iWave < d_numEigenValues; iWave++)
      {
        double maxOcc = -1.0;
        for (dftfe::uInt kPoint = 0; kPoint < totkPoints; ++kPoint)
          {
            if (d_dftParamsPtr->spinPolarized)
              {
                occupationVector[2 * kPoint] = dftUtils::getPartialOccupancy(
                  eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                             iWave],
                  FE,
                  C_kb,
                  d_dftParamsPtr->TVal);
                occupationVector[2 * kPoint + 1] =
                  dftUtils::getPartialOccupancy(
                    eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                 d_numEigenValues +
                                               iWave],
                    FE,
                    C_kb,
                    d_dftParamsPtr->TVal);
                maxOcc = std::max(maxOcc,
                                  std::max(occupationVector[2 * kPoint + 1],
                                           occupationVector[2 * kPoint]));
              }
            else
              {
                occupationVector[kPoint] = dftUtils::getPartialOccupancy(
                  eigenValuesFlattenedGlobal[kPoint * d_numEigenValues + iWave],
                  FE,
                  C_kb,
                  d_dftParamsPtr->TVal);
                maxOcc = std::max(maxOcc, occupationVector[kPoint]);
              }
          }

        if (maxOcc < 1E-5)
          {
            maxeigenIndex = iWave;
            break;
          }
      }

    dftfe::uInt numberEigenValues =
      d_dftParamsPtr->highestStateOfInterestForChebFiltering;
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        FILE *pFile;
        pFile = fopen("bands.out", "w");
        fprintf(pFile, "%d %d \n", totkPoints, numberEigenValues);
        for (dftfe::uInt kPoint = 0; kPoint < totkPoints; ++kPoint)
          {
            for (dftfe::uInt iWave = 0; iWave < numberEigenValues; ++iWave)
              {
                if (d_dftParamsPtr->spinPolarized)
                  {
                    double occupancyUp = dftUtils::getPartialOccupancy(
                      eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                                 iWave],
                      FE,
                      C_kb,
                      d_dftParamsPtr->TVal);

                    double occupancyDown = dftUtils::getPartialOccupancy(
                      eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                   d_numEigenValues +
                                                 iWave],
                      FE,
                      C_kb,
                      d_dftParamsPtr->TVal);

                    fprintf(
                      pFile,
                      "%d  %d   %.14g   %.14g   %.14g   %.14g\n",
                      kPoint,
                      iWave,
                      eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                                 iWave],
                      eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                   d_numEigenValues +
                                                 iWave],
                      occupancyUp,
                      occupancyDown);
                    if (d_dftParamsPtr->reproducible_output &&
                        d_dftParamsPtr->verbosity == 0)
                      {
                        double eigenUpTrunc =
                          std::floor(
                            1000000000 *
                            (eigenValuesFlattenedGlobal
                               [2 * kPoint * d_numEigenValues + iWave])) /
                          1000000000.0;
                        double eigenDownTrunc =
                          std::floor(
                            1000000000 *
                            (eigenValuesFlattenedGlobal
                               [(2 * kPoint + 1) * d_numEigenValues + iWave])) /
                          1000000000.0;
                        double occupancyUpTrunc =
                          std::floor(1000000000 * (occupancyUp)) / 1000000000.0;
                        double occupancyDownTrunc =
                          std::floor(1000000000 * (occupancyDown)) /
                          1000000000.0;
                        pcout << kPoint << "  " << iWave << "  " << std::fixed
                              << std::setprecision(8) << eigenUpTrunc << "  "
                              << eigenDownTrunc << "  " << occupancyUpTrunc
                              << "  " << occupancyDownTrunc << std::endl;
                      }
                  }
                else
                  {
                    double occupancy = dftUtils::getPartialOccupancy(
                      eigenValuesFlattenedGlobal[kPoint * d_numEigenValues +
                                                 iWave],
                      FE,
                      C_kb,
                      d_dftParamsPtr->TVal);
                    fprintf(
                      pFile,
                      "%d  %d %.14g %.14g\n",
                      kPoint,
                      iWave,
                      eigenValuesFlattenedGlobal[kPoint * d_numEigenValues +
                                                 iWave],
                      occupancy);
                    if (d_dftParamsPtr->reproducible_output &&
                        d_dftParamsPtr->verbosity == 0)
                      {
                        double eigenTrunc =
                          std::floor(1000000000 *
                                     (eigenValuesFlattenedGlobal
                                        [kPoint * d_numEigenValues + iWave])) /
                          1000000000.0;
                        double occupancyTrunc =
                          std::floor(1000000000 * (occupancy)) / 1000000000.0;
                        pcout << kPoint << "  " << iWave << "  " << std::fixed
                              << std::setprecision(8) << eigenTrunc << " "
                              << occupancyTrunc << std::endl;
                      }
                  }
              }
          }
      }
    MPI_Barrier(d_mpiCommParent);
    //
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<memorySpace>::getAtomLocationsCart() const
  {
    return atomLocations;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<memorySpace>::getImageAtomLocationsCart() const
  {
    return d_imagePositionsTrunc;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<dftfe::Int> &
  dftClass<memorySpace>::getImageAtomIDs() const
  {
    return d_imageIdsTrunc;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<memorySpace>::getAtomLocationsFrac() const
  {
    return atomLocationsFractional;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<memorySpace>::getCell() const
  {
    return d_domainBoundingVectors;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getCellVolume() const
  {
    return d_domainVolume;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  const std::set<dftfe::uInt> &
  dftClass<memorySpace>::getAtomTypes() const
  {
    return atomTypes;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<double> &
  dftClass<memorySpace>::getForceonAtoms() const
  {
    return (forcePtr->getAtomsForces());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dealii::Tensor<2, 3, double> &
  dftClass<memorySpace>::getCellStress() const
  {
    return (forcePtr->getStress());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftParameters &
  dftClass<memorySpace>::getParametersObject() const
  {
    return (*d_dftParamsPtr);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getInternalEnergy() const
  {
    return d_groundStateEnergy;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getEntropicEnergy() const
  {
    return d_entropicEnergy;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getFreeEnergy() const
  {
    return d_freeEnergy;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const distributedCPUVec<double> &
  dftClass<memorySpace>::getRhoNodalOut() const
  {
    return d_densityOutNodalValues[0];
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const distributedCPUVec<double> &
  dftClass<memorySpace>::getRhoNodalSplitOut() const
  {
    return d_rhoOutNodalValuesSplit;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getTotalChargeforRhoSplit()
  {
    double temp =
      (-totalCharge(d_matrixFreeDataPRefined, d_rhoOutNodalValuesSplit) /
       d_domainVolume);
    return (temp);
  }



  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::resetRhoNodalIn(distributedCPUVec<double> &OutDensity)
  {
    d_densityOutNodalValues[0] = OutDensity;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::resetRhoNodalSplitIn(
    distributedCPUVec<double> &OutDensity)
  {
    d_rhoOutNodalValuesSplit = OutDensity;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::writeGSElectronDensity(const std::string Path) const
  {
    const dftfe::uInt poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const dftfe::uInt bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);

    if (poolId == 0 && bandGroupId == 0)
      {
        std::vector<std::shared_ptr<dftUtils::CompositeData>> data(0);

        const dealii::Quadrature<3> &quadrature_formula =
          matrix_free_data.get_quadrature(d_densityQuadratureId);
        dealii::FEValues<3> fe_values(FE,
                                      quadrature_formula,
                                      dealii::update_quadrature_points |
                                        dealii::update_JxW_values);
        const dftfe::uInt   n_q_points = quadrature_formula.size();

        const dftfe::uInt totalLocallyOwnedCells =
          d_basisOperationsPtrHost->nCells();

        const dftfe::uInt totalQuadPoints = totalLocallyOwnedCells * n_q_points;

        std::vector<dftfe::uInt> numberOfPointsInEachProc;
        numberOfPointsInEachProc.resize(n_mpi_processes);
        std::fill(numberOfPointsInEachProc.begin(),
                  numberOfPointsInEachProc.end(),
                  0);

        numberOfPointsInEachProc[this_mpi_process] = totalQuadPoints;


        MPI_Allreduce(MPI_IN_PLACE,
                      &numberOfPointsInEachProc[0],
                      n_mpi_processes,
                      dataTypes::mpi_type_id(&numberOfPointsInEachProc[0]),
                      MPI_SUM,
                      mpi_communicator);

        dftfe::uInt quadIdStartIndex = 0;

        for (dftfe::uInt iProc = 0; iProc < this_mpi_process; iProc++)
          {
            quadIdStartIndex += numberOfPointsInEachProc[iProc];
          }



        // loop over elements
        typename dealii::DoFHandler<3>::active_cell_iterator
          cell = dofHandler.begin_active(),
          endc = dofHandler.end();
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                const dftfe::uInt cellIndex =
                  d_basisOperationsPtrHost->cellIndex(cell->id());
                const double *rhoValues =
                  d_densityOutQuadValues[0].data() + cellIndex * n_q_points;
                const double *magValues =
                  d_dftParamsPtr->spinPolarized == 1 ?
                    d_densityOutQuadValues[1].data() + cellIndex * n_q_points :
                    NULL;

                for (dftfe::uInt q_point = 0; q_point < n_q_points; ++q_point)
                  {
                    std::vector<double> quadVals(0);

                    const dealii::Point<3> &quadPoint =
                      fe_values.quadrature_point(q_point);
                    const double jxw = fe_values.JxW(q_point);

                    quadVals.push_back(quadIdStartIndex +
                                       cellIndex * n_q_points + q_point);
                    quadVals.push_back(quadPoint[0]);
                    quadVals.push_back(quadPoint[1]);
                    quadVals.push_back(quadPoint[2]);
                    quadVals.push_back(jxw);

                    if (d_dftParamsPtr->spinPolarized == 1)
                      {
                        quadVals.push_back(rhoValues[q_point]);
                        quadVals.push_back(magValues[q_point]);
                      }
                    else
                      {
                        quadVals.push_back(rhoValues[q_point]);
                      }

                    data.push_back(
                      std::make_shared<dftUtils::QuadDataCompositeWrite>(
                        quadVals));
                  }
              }
          }

        std::vector<dftUtils::CompositeData *> dataRawPtrs(data.size());
        for (dftfe::uInt i = 0; i < data.size(); ++i)
          dataRawPtrs[i] = data[i].get();
        dftUtils::MPIWriteOnFile().writeData(dataRawPtrs,
                                             Path,
                                             mpi_communicator);
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::writeMesh()
  {
    //
    // compute nodal electron-density from quad data
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                            d_constraintsRhoNodal,
                            d_densityDofHandlerIndexElectro,
                            d_densityQuadratureIdElectro,
                            d_densityInQuadValues[0],
                            rhoNodalField);

    //
    // only generate output for electron-density
    //
    dealii::DataOut<3> dataOutRho;
    dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
    dataOutRho.add_data_vector(rhoNodalField, std::string("density"));
    dataOutRho.set_flags(dealii::DataOutBase::VtkFlags(
      std::numeric_limits<double>::min(),
      std::numeric_limits<dftfe::uInt>::min(),
      true,
      dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::
        best_speed, // This flag is version dependent for dealII 9.5.0 it
                    // is
                    // dealii::DataOutBase::CompressionLevel::best_speed
      true));       // higher order cells set to true
    dataOutRho.build_patches(d_dftParamsPtr->finiteElementPolynomialOrder);

    std::string tempFolder = "meshOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
                                               dataOutRho,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "intialDensityOutput");



    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << std::endl
        << "------------------DFT-FE mesh file creation completed---------------------------"
        << std::endl;
  }
  // Output Potential
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::outputPotential()
  {
    //
    // compute nodal electron-density from quad data
    //
    dealii::DataOut<3> dataOutPot;
    dataOutPot.attach_dof_handler(d_dofHandlerPRefined);
    printNodalFieldAtCenterLine(d_phiTotRhoOut, d_dofHandlerPRefined);
    computeOutputPotential();
    MPI_Barrier(d_mpiCommParent);
    const dftfe::uInt poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const dftfe::uInt bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    const dftfe::uInt minPoolId =
      dealii::Utilities::MPI::min(poolId, interpoolcomm);
    const dftfe::uInt minBandGroupId =
      dealii::Utilities::MPI::min(bandGroupId, interBandGroupComm);
    if (poolId == minPoolId && bandGroupId == minBandGroupId)
      {
        // Add the new thing to print
        // printAppliedPotentialAtConstraintNodes(d_phiTotRhoOut,
        //                                        d_constraintNodesL,
        //                                        d_constraintNodesR,
        //                                        d_dofHandlerPRefined);
        // printAppliedPotentialAtConstraintNodes(d_potentialOut,
        //                                        d_constraintNodesL,
        //                                        d_constraintNodesR,
        //                                        d_dofHandlerPRefined);
      }
    dataOutPot.add_data_vector(d_potentialOut, std::string("potential"));
    dataOutPot.build_patches(
      d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics);
    std::string tempFolder = "PotentialOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);
    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerPRefined,
                                               dataOutPot,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "PotentialOutput");
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::outputExternalPotential()
  {
    //
    // compute nodal electron-density from quad data
    //
    dealii::DataOut<3> dataOutPot;
    dataOutPot.attach_dof_handler(d_dofHandlerPRefined);
    dataOutPot.build_patches(
      d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics);
    std::string tempFolder = "ExternalPotentialOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);
    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerPRefined,
                                               dataOutPot,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "ExternalPotentialOutput");
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeOutputPotential()
  {
    // Interpolate d_phiTotRhoOut to quad points d_lpspQuadratureIdElectro
    // compute at quad points d_phiTotRhoOut + d_pseudoVLoc
    pcout << "Starting to compute Output Potential: " << std::endl;

    const dealii::Quadrature<3> &quadratureHigh =
      d_matrixFreeDataPRefined.get_quadrature(d_lpspQuadratureIdElectro);
    const dftfe::uInt numberQuadraturePoints = quadratureHigh.size();
    // pcout << "FE rule" << std::endl;
    dealii::FEValues<3> fe_values(d_dofHandlerPRefined.get_fe(),
                                  quadratureHigh,
                                  dealii::update_values);
    std::map<dealii::CellId, std::vector<double>> potentialQuadVals;
    dealii::DoFHandler<3>::active_cell_iterator   cell = d_dofHandlerPRefined
                                                         .begin_active(),
                                                endc =
                                                  d_dofHandlerPRefined.end();
    const dftfe::uInt numberNodesPerElement =
      d_dofHandlerPRefined.get_fe().dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      numberNodesPerElement);

    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(cellGlobalDofIndices);
            std::vector<double> &Values     = potentialQuadVals[cell->id()];
            std::vector<double> &pseudoVLoc = d_pseudoVLoc[cell->id()];
            Values.resize(numberQuadraturePoints);
            for (int qpoint = 0; qpoint < numberQuadraturePoints; qpoint++)
              {
                for (dftfe::uInt inode = 0; inode < numberNodesPerElement;
                     ++inode)
                  {
                    dealii::types::global_dof_index localDoFIdi =
                      d_matrixFreeDataPRefined
                        .get_vector_partitioner(d_phiTotDofHandlerIndexElectro)
                        ->global_to_local(cellGlobalDofIndices[inode]);

                    Values[qpoint] += fe_values.shape_value(inode, qpoint) *
                                      d_phiTotRhoOut.local_element(localDoFIdi);
                  }
                Values[qpoint] += pseudoVLoc[qpoint];
              }
          }
      }
    // std::cout<<"DEBUG: Line 5099 "<<this_mpi_process<<std::endl;
    // Interpolate From quad points to nodal points
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_potentialOut, d_phiTotDofHandlerIndexElectro);
    l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                            d_constraintsPRefined,
                            d_phiExtDofHandlerIndexElectro,
                            d_lpspQuadratureIdElectro,
                            potentialQuadVals,
                            d_potentialOut);
    d_potentialOut.update_ghost_values();
    d_constraintsPRefinedOnlyHanging.distribute(d_potentialOut);
  }


  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::computeResidualQuadData(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &outValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &inValues,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &residualValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &JxW,
    const bool computeNorm)
  {
    std::transform(outValues.begin(),
                   outValues.end(),
                   inValues.begin(),
                   residualValues.begin(),
                   std::minus<>{});
    double normValue = 0.0;
    if (computeNorm)
      {
        for (dftfe::uInt iQuad = 0; iQuad < residualValues.size(); ++iQuad)
          normValue +=
            residualValues[iQuad] * residualValues[iQuad] * JxW[iQuad];
        MPI_Allreduce(
          MPI_IN_PLACE, &normValue, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      }
    return std::sqrt(normValue);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::computeResidualNodalData(
    const distributedCPUVec<double> &outValues,
    const distributedCPUVec<double> &inValues,
    distributedCPUVec<double>       &residualValues)
  {
    residualValues.reinit(inValues);

    residualValues = 0.0;

    // compute residual = rhoOut - rhoIn
    residualValues.add(1.0, outValues, -1.0, inValues);

    // compute l2 norm of the field residual
    double normValue = rhofieldl2Norm(d_matrixFreeDataPRefined,
                                      residualValues,
                                      d_densityDofHandlerIndexElectro,
                                      d_densityQuadratureIdElectro);
    return normValue;
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::determineAtomsOfInterstPseudopotential(
    const std::vector<std::vector<double>> &atomCoordinates)
  {
    d_atomLocationsInterestPseudopotential.clear();
    d_atomIdPseudopotentialInterestToGlobalId.clear();
    dftfe::uInt atomIdPseudo = 0;
    // pcout<<"Atoms of interest: "<<std::endl;
    for (dftfe::uInt iAtom = 0; iAtom < atomCoordinates.size(); iAtom++)
      {
        if (true)
          {
            d_atomLocationsInterestPseudopotential.push_back(
              atomCoordinates[iAtom]);
            d_atomIdPseudopotentialInterestToGlobalId[atomIdPseudo] = iAtom;
            // pcout<<iAtom<<" "<<atomIdPseudo<<" ";
            // for(dftfe::Int i = 0; i <
            // d_atomLocationsInterestPseudopotential[atomIdPseudo].size();
            // i++)
            //   pcout<<d_atomLocationsInterestPseudopotential[atomIdPseudo][i]<<"
            //   ";
            // pcout<<std::endl;
            atomIdPseudo++;
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST> &
  dftClass<memorySpace>::getEigenVectorsHost() const
  {
    return d_eigenVectorsFlattenedHost;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
  dftClass<memorySpace>::getEigenVectors() const
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      return d_eigenVectorsFlattenedHost;
#ifdef DFTFE_WITH_DEVICE
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      return d_eigenVectorsFlattenedDevice;
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getFermiEnergy() const
  {
    return fermiEnergy;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::getNumElectrons() const
  {
    return numElectrons;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::setNumElectrons(dftfe::uInt inputNumElectrons)
  {
    this->numElectrons = inputNumElectrons;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<memorySpace>::getEigenValues() const
  {
    return eigenValues;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  elpaScalaManager *
  dftClass<memorySpace>::getElpaScalaManager() const
  {
    return d_elpaScala;
  }

#ifdef DFTFE_WITH_DEVICE

  template <dftfe::utils::MemorySpace memorySpace>
  chebyshevOrthogonalizedSubspaceIterationSolverDevice *
  dftClass<memorySpace>::getSubspaceIterationSolverDevice()
  {
    return &d_subspaceIterationSolverDevice;
  }

#endif

  template <dftfe::utils::MemorySpace memorySpace>
  chebyshevOrthogonalizedSubspaceIterationSolver *
  dftClass<memorySpace>::getSubspaceIterationSolverHost()
  {
    return &d_subspaceIterationSolver;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  KohnShamDFTBaseOperator<memorySpace> *
  dftClass<memorySpace>::getKohnShamDFTBaseOperatorClass()
  {
    return d_kohnShamDFTOperatorPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getDensityDofHandlerIndex()
  {
    return d_densityDofHandlerIndex;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getDensityQuadratureId()
  {
    return d_densityQuadratureId;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<double> &
  dftClass<memorySpace>::getKPointWeights() const
  {
    return d_kPointWeights;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getNumEigenValues() const
  {
    return d_numEigenValues;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  triangulationManager *
  dftClass<memorySpace>::getTriangulationManager()
  {
    return &d_mesh;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dealii::AffineConstraints<double> *
  dftClass<memorySpace>::getDensityConstraint()
  {
    return &constraintsNone;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dealii::MatrixFree<3, double> &
  dftClass<memorySpace>::getMatrixFreeDataElectro() const
  {
    return d_matrixFreeDataPRefined;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getElectroDofHandlerIndex() const
  {
    return d_phiTotDofHandlerIndexElectro;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getElectroQuadratureRhsId() const
  {
    return d_densityQuadratureIdElectro;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getElectroQuadratureAxId() const
  {
    return d_phiTotAXQuadratureIdElectro;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<
    dftfe::basis::FEBasisOperations<dataTypes::number,
                                    double,
                                    dftfe::utils::MemorySpace::HOST>>
  dftClass<memorySpace>::getBasisOperationsHost()
  {
    return d_basisOperationsPtrHost;
  }

  template <dftfe::utils::MemorySpace memorySpace>

  std::shared_ptr<
    dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
  dftClass<memorySpace>::getBasisOperationsMemSpace()
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      return d_basisOperationsPtrHost;
#ifdef DFTFE_WITH_DEVICE
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      return d_basisOperationsPtrDevice;
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<
    dftfe::basis::
      FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
  dftClass<memorySpace>::getBasisOperationsElectroHost()
  {
    return d_basisOperationsPtrElectroHost;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<dftfe::basis::FEBasisOperations<double, double, memorySpace>>
  dftClass<memorySpace>::getBasisOperationsElectroMemSpace()
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      return d_basisOperationsPtrElectroHost;
#ifdef DFTFE_WITH_DEVICE
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      return d_basisOperationsPtrElectroDevice;
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
  dftClass<memorySpace>::getBLASWrapperMemSpace()
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      return d_BLASWrapperPtrHost;
#ifdef DFTFE_WITH_DEVICE
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      return d_BLASWrapperPtr;
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<
    dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
  dftClass<memorySpace>::getBLASWrapperHost()
  {
    return d_BLASWrapperPtrHost;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &
  dftClass<memorySpace>::getDensityInValues()
  {
    return d_densityInQuadValues;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::vector<
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> &
  dftClass<memorySpace>::getDensityOutValues()
  {
    return d_densityOutQuadValues;
  }

  /// map of atom node number and atomic weight
  template <dftfe::utils::MemorySpace memorySpace>
  std::map<dealii::types::global_dof_index, double> &
  dftClass<memorySpace>::getAtomNodeToChargeMap()
  {
    return d_atomNodeIdToChargeMap;
  }

  /// non-intersecting smeared charges of all atoms at quad points
  template <dftfe::utils::MemorySpace memorySpace>
  std::map<dealii::CellId, std::vector<double>> &
  dftClass<memorySpace>::getBQuadValuesAllAtoms()
  {
    return d_bQuadValuesAllAtoms;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  dftClass<memorySpace>::getSmearedChargeQuadratureIdElectro()
  {
    return d_smearedChargeQuadratureIdElectro;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const dealii::AffineConstraints<double> *
  dftClass<memorySpace>::getConstraintsVectorElectro()
  {
    return d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro];
  }



  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  dftClass<memorySpace>::getMPIDomain() const
  {
    return mpi_communicator;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  dftClass<memorySpace>::getMPIParent() const
  {
    return d_mpiCommParent;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  dftClass<memorySpace>::getMPIInterPool() const
  {
    return interpoolcomm;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const MPI_Comm &
  dftClass<memorySpace>::getMPIInterBand() const
  {
    return interBandGroupComm;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const expConfiningPotential &
  dftClass<memorySpace>::getConfiningPotential() const
  {
    return d_expConfiningPot;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<double> &
  dftClass<memorySpace>::getNearestAtomDistance() const
  {
    return d_nearestAtomDistances;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<memorySpace>::getLocalVselfs() const
  {
    return d_localVselfs;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<dftfe::uInt>> &
  dftClass<memorySpace>::getbCellNonTrivialAtomIds() const
  {
    return d_bCellNonTrivialAtomIds;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<hubbard<dataTypes::number, memorySpace>>
  dftClass<memorySpace>::getHubbardClassPtr()
  {
    return d_hubbardClassPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  bool
  dftClass<memorySpace>::isHubbardCorrectionsUsed()
  {
    return d_useHubbard;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const std::map<dealii::CellId, std::vector<double>> &
  dftClass<memorySpace>::getPseudoVLoc() const
  {
    return d_pseudoVLoc;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computePlanarAverageField(
    const distributedCPUVec<double>     &field,
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const dftfe::uInt                    matrixFreeDofhandlerIndex,
    const MPI_Comm                      &mpiCommDomain,
    const MPI_Comm                      &interpoolcomm,
    std::vector<std::vector<double>>    &outputValues)
  {
    outputValues.clear();

    distributedCPUMultiVec<double> xField;
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      field.get_partitioner(), 1, xField);

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::HOST>::copy(xField.localSize() *
                                               xField.numVectors(),
                                             xField.begin(),
                                             field.begin());



    const dealii::DoFHandler<3> *dofHandlerMesh1 =
      &matrixFreeDataObject.get_dof_handler(matrixFreeDofhandlerIndex);
    dftfe::uInt totallyOwnedCellsMesh1 =
      matrixFreeDataObject.n_physical_cells();

    const dftfe::uInt noZGridPoints = d_dftParamsPtr->noOfSamplingPointsZ;
    // Fixed
    const dftfe::uInt noXGridPoints = d_dftParamsPtr->noOfSamplingPointsX;
    const dftfe::uInt noYGridPoints = d_dftParamsPtr->noOfSamplingPointsY;

    dftfe::uInt totalGridPoints = noYGridPoints * noXGridPoints * noZGridPoints;

    double minVal = -0.4999;
    double maxVal = 0.4999;

    double deltaX = 2 * maxVal / (noXGridPoints - 1);
    double deltaY = 2 * maxVal / (noYGridPoints - 1);
    double deltaZ = 2 * maxVal / (noZGridPoints - 1);



    std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> srcCellsMesh1(0);
    std::vector<std::shared_ptr<
      InterpolateFromCellToLocalPoints<double,
                                       dftfe::utils::MemorySpace::HOST>>>
                                    interpolateLocalMesh1(0);
    const dealii::FiniteElement<3> &feMesh1 = dofHandlerMesh1->get_fe();
    std::vector<dftfe::uInt>        numberDofsPerCell1;
    numberDofsPerCell1.resize(totallyOwnedCellsMesh1);
    dftfe::uInt iElemIndex = 0;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellMesh1 = dofHandlerMesh1->begin_active(),
      endcMesh1 = dofHandlerMesh1->end();
    for (; cellMesh1 != endcMesh1; cellMesh1++)
      {
        if (cellMesh1->is_locally_owned())
          {
            numberDofsPerCell1[iElemIndex] =
              dofHandlerMesh1->get_fe().dofs_per_cell;
            auto srcCellPtr =
              std::make_shared<dftfe::utils::FECell<3>>(cellMesh1, feMesh1);
            srcCellsMesh1.push_back(srcCellPtr);

            interpolateLocalMesh1.push_back(
              std::make_shared<InterpolateFromCellToLocalPoints<
                double,
                dftfe::utils::MemorySpace::HOST>>(
                srcCellPtr, numberDofsPerCell1[iElemIndex], true));

            iElemIndex++;
          }
      }

    dftfe::uInt totalGridPointsPerProc =
      (int)(totalGridPoints) / int(n_mpi_processes);
    dftfe::uInt remainderGridPointsPerPrc =
      int(totalGridPoints) % int(n_mpi_processes);
    pcout << "Number of points: " << totalGridPointsPerProc << " "
          << remainderGridPointsPerPrc << " " << totalGridPoints << std::endl;
    std::vector<dftfe::uInt> startIndex(n_mpi_processes, 0);
    std::vector<dftfe::uInt> endIndex(n_mpi_processes, 0);
    dftfe::uInt              countTot = 0;
    dftfe::uInt              countRem = 0;

    for (int iTask = 0; iTask < n_mpi_processes; iTask++)
      {
        startIndex[iTask] = countTot;
        countTot += totalGridPointsPerProc;
        if (countRem < remainderGridPointsPerPrc)
          {
            countTot++;
            countRem++;
          }
        endIndex[iTask] = countTot;
      }
    dftfe::uInt noPoints =
      endIndex[this_mpi_process] - startIndex[this_mpi_process];
    std::vector<std::vector<double>> dstCoordinatesPerProc(
      noPoints, std::vector<double>(3, 0.0));
    dftfe::uInt count = 0;
    for (int i = 0; i < noZGridPoints; i++)
      {
        double zCoordinate = minVal + double(i) * deltaZ;
        for (int j = 0; j < noXGridPoints; j++)
          {
            double xCoordinate = minVal + double(j) * deltaX;
            for (int k = 0; k < noYGridPoints; k++)
              {
                double yCoordinate = minVal + double(k) * deltaY;
                if (count >= startIndex[this_mpi_process] &&
                    count < endIndex[this_mpi_process])
                  {
                    dftfe::uInt index = count - startIndex[this_mpi_process];
                    dstCoordinatesPerProc[index][0] =
                      xCoordinate * d_domainBoundingVectors[0][0] +
                      yCoordinate * d_domainBoundingVectors[1][0] +
                      zCoordinate * d_domainBoundingVectors[2][0];
                    dstCoordinatesPerProc[index][1] =
                      xCoordinate * d_domainBoundingVectors[0][1] +
                      yCoordinate * d_domainBoundingVectors[1][1] +
                      zCoordinate * d_domainBoundingVectors[2][1];
                    dstCoordinatesPerProc[index][2] =
                      xCoordinate * d_domainBoundingVectors[0][2] +
                      yCoordinate * d_domainBoundingVectors[1][2] +
                      zCoordinate * d_domainBoundingVectors[2][2];
                  }

                count++;
              }
          }
      }



    auto mesh1toMesh2 = std::make_shared<
      InterpolateCellWiseDataToPoints<double, dftfe::utils::MemorySpace::HOST>>(
      srcCellsMesh1,
      interpolateLocalMesh1,
      dstCoordinatesPerProc,
      numberDofsPerCell1,
      4,
      mpiCommDomain);



    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      outputQuadDataPerProc(noPoints, 0.0);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapChild;
    dftfe::vectorTools::computeCellLocalIndexSetMap(
      xField.getMPIPatternP2P(),
      matrixFreeDataObject,
      matrixFreeDofhandlerIndex,
      1,
      fullFlattenedArrayCellLocalProcIndexIdMapChild);

    dftfe::utils::MemoryStorage<dftfe::uInt, dftfe::utils::MemorySpace::HOST>
      fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage;
    fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage.resize(
      fullFlattenedArrayCellLocalProcIndexIdMapChild.size());
    fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage.copyFrom(
      fullFlattenedArrayCellLocalProcIndexIdMapChild);


    mesh1toMesh2->interpolateSrcDataToTargetPoints(
      d_BLASWrapperPtrHost,
      xField,
      1,
      fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage,
      outputQuadDataPerProc,
      1,
      1,
      0,
      true);


    // computing planar Average
    std::vector<double> planarTotal(noZGridPoints, 0.0);
    outputValues.resize(noZGridPoints, std::vector<double>(2, 0.0));
    for (dftfe::uInt i = 0; i < noPoints; i++)
      {
        dftfe::uInt index = i + startIndex[this_mpi_process];

        planarTotal[index / (noXGridPoints * noYGridPoints)] +=
          outputQuadDataPerProc[i];
      }

    MPI_Allreduce(MPI_IN_PLACE,
                  &planarTotal[0],
                  noZGridPoints,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpiCommDomain);

    for (int i = 0; i < noZGridPoints; i++)
      {
        double zCoordinate = minVal + double(i) * deltaZ;
        outputValues[i][0] = zCoordinate * d_domainBoundingVectors[2][2];
        outputValues[i][1] = planarTotal[i] / (noXGridPoints * noYGridPoints);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeDeltaPlanarAverageField(
    const distributedCPUVec<double>     &field,
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const dftfe::uInt                    matrixFreeDofhandlerIndex,
    const MPI_Comm                      &mpiCommDomain,
    const MPI_Comm                      &interpoolcomm,
    std::vector<std::vector<double>>    &outputValues)
  {
    outputValues.clear();

    distributedCPUMultiVec<double> xField;
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      field.get_partitioner(), 1, xField);

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::HOST>::copy(xField.localSize() *
                                               xField.numVectors(),
                                             xField.begin(),
                                             field.begin());



    const dealii::DoFHandler<3> *dofHandlerMesh1 =
      &matrixFreeDataObject.get_dof_handler(matrixFreeDofhandlerIndex);
    dftfe::uInt totallyOwnedCellsMesh1 =
      matrixFreeDataObject.n_physical_cells();

    const dftfe::uInt   noZGridPoints = 2;
    std::vector<double> z_inputCoord(2);
    z_inputCoord[0] = d_dftParamsPtr->zCoordinateL - 0.5;
    z_inputCoord[1] = d_dftParamsPtr->zCoordinateR - 0.5;
    // Fixed
    const dftfe::uInt noXGridPoints = d_dftParamsPtr->noOfSamplingPointsX;
    const dftfe::uInt noYGridPoints = d_dftParamsPtr->noOfSamplingPointsY;

    dftfe::uInt totalGridPoints = noYGridPoints * noXGridPoints * noZGridPoints;

    double minVal = -0.4999;
    double maxVal = 0.4999;

    double deltaX = 2 * maxVal / (noXGridPoints - 1);
    double deltaY = 2 * maxVal / (noYGridPoints - 1);



    std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> srcCellsMesh1(0);
    std::vector<std::shared_ptr<
      InterpolateFromCellToLocalPoints<double,
                                       dftfe::utils::MemorySpace::HOST>>>
                                    interpolateLocalMesh1(0);
    const dealii::FiniteElement<3> &feMesh1 = dofHandlerMesh1->get_fe();
    std::vector<dftfe::uInt>        numberDofsPerCell1;
    numberDofsPerCell1.resize(totallyOwnedCellsMesh1);
    dftfe::uInt iElemIndex = 0;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellMesh1 = dofHandlerMesh1->begin_active(),
      endcMesh1 = dofHandlerMesh1->end();
    for (; cellMesh1 != endcMesh1; cellMesh1++)
      {
        if (cellMesh1->is_locally_owned())
          {
            numberDofsPerCell1[iElemIndex] =
              dofHandlerMesh1->get_fe().dofs_per_cell;
            auto srcCellPtr =
              std::make_shared<dftfe::utils::FECell<3>>(cellMesh1, feMesh1);
            srcCellsMesh1.push_back(srcCellPtr);

            interpolateLocalMesh1.push_back(
              std::make_shared<InterpolateFromCellToLocalPoints<
                double,
                dftfe::utils::MemorySpace::HOST>>(
                srcCellPtr, numberDofsPerCell1[iElemIndex], true));

            iElemIndex++;
          }
      }

    dftfe::uInt totalGridPointsPerProc =
      (int)(totalGridPoints) / int(n_mpi_processes);
    dftfe::uInt remainderGridPointsPerPrc =
      int(totalGridPoints) % int(n_mpi_processes);
    pcout << "Number of points: " << totalGridPointsPerProc << " "
          << remainderGridPointsPerPrc << " " << totalGridPoints << std::endl;
    std::vector<dftfe::uInt> startIndex(n_mpi_processes, 0);
    std::vector<dftfe::uInt> endIndex(n_mpi_processes, 0);
    dftfe::uInt              countTot = 0;
    dftfe::uInt              countRem = 0;

    for (int iTask = 0; iTask < n_mpi_processes; iTask++)
      {
        startIndex[iTask] = countTot;
        countTot += totalGridPointsPerProc;
        if (countRem < remainderGridPointsPerPrc)
          {
            countTot++;
            countRem++;
          }
        endIndex[iTask] = countTot;
        pcout << "Start Index and EndIndex: " << iTask << " "
              << startIndex[iTask] << " " << endIndex[iTask] << std::endl;
      }
    dftfe::uInt noPoints =
      endIndex[this_mpi_process] - startIndex[this_mpi_process];
    std::vector<std::vector<double>> dstCoordinatesPerProc(
      noPoints, std::vector<double>(3, 0.0));
    dftfe::uInt count = 0;
    for (int i = 0; i < noZGridPoints; i++)
      {
        double zCoordinate = z_inputCoord[i];
        for (int j = 0; j < noXGridPoints; j++)
          {
            double xCoordinate = minVal + double(j) * deltaX;
            for (int k = 0; k < noYGridPoints; k++)
              {
                double yCoordinate = minVal + double(k) * deltaY;
                if (count >= startIndex[this_mpi_process] &&
                    count < endIndex[this_mpi_process])
                  {
                    dftfe::uInt index = count - startIndex[this_mpi_process];
                    dstCoordinatesPerProc[index][0] =
                      xCoordinate * d_domainBoundingVectors[0][0] +
                      yCoordinate * d_domainBoundingVectors[1][0] +
                      zCoordinate * d_domainBoundingVectors[2][0];
                    dstCoordinatesPerProc[index][1] =
                      xCoordinate * d_domainBoundingVectors[0][1] +
                      yCoordinate * d_domainBoundingVectors[1][1] +
                      zCoordinate * d_domainBoundingVectors[2][1];
                    dstCoordinatesPerProc[index][2] =
                      xCoordinate * d_domainBoundingVectors[0][2] +
                      yCoordinate * d_domainBoundingVectors[1][2] +
                      zCoordinate * d_domainBoundingVectors[2][2];
                  }

                count++;
              }
          }
      }



    auto mesh1toMesh2 = std::make_shared<
      InterpolateCellWiseDataToPoints<double, dftfe::utils::MemorySpace::HOST>>(
      srcCellsMesh1,
      interpolateLocalMesh1,
      dstCoordinatesPerProc,
      numberDofsPerCell1,
      4,
      mpiCommDomain);



    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      outputQuadDataPerProc(noPoints, 0.0);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapChild;
    dftfe::vectorTools::computeCellLocalIndexSetMap(
      xField.getMPIPatternP2P(),
      matrixFreeDataObject,
      matrixFreeDofhandlerIndex,
      1,
      fullFlattenedArrayCellLocalProcIndexIdMapChild);

    dftfe::utils::MemoryStorage<dftfe::uInt, dftfe::utils::MemorySpace::HOST>
      fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage;
    fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage.resize(
      fullFlattenedArrayCellLocalProcIndexIdMapChild.size());
    fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage.copyFrom(
      fullFlattenedArrayCellLocalProcIndexIdMapChild);


    mesh1toMesh2->interpolateSrcDataToTargetPoints(
      d_BLASWrapperPtrHost,
      xField,
      1,
      fullFlattenedArrayCellLocalProcIndexIdMapChildMemStorage,
      outputQuadDataPerProc,
      1,
      1,
      0,
      true);


    // computing planar Average
    std::vector<double> planarTotal(noZGridPoints, 0.0);
    outputValues.resize(noZGridPoints, std::vector<double>(2, 0.0));
    for (dftfe::uInt i = 0; i < noPoints; i++)
      {
        dftfe::uInt index = i + startIndex[this_mpi_process];

        planarTotal[index / (noXGridPoints * noYGridPoints)] +=
          outputQuadDataPerProc[i];
      }

    MPI_Allreduce(MPI_IN_PLACE,
                  &planarTotal[0],
                  noZGridPoints,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpiCommDomain);

    for (int i = 0; i < noZGridPoints; i++)
      {
        outputValues[i][0] = z_inputCoord[i] * d_domainBoundingVectors[2][2];
        outputValues[i][1] = planarTotal[i] / (noXGridPoints * noYGridPoints);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::updateAuxDensityXCMatrix(
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityQuadValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityQuadValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                                                        &tauQuadValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCore,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCore,
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                           &eigenVectorsFlattenedMemSpace,
    const std::vector<std::vector<double>> &eigenValues_,
    const double                            fermiEnergy_,
    const double                            fermiEnergyUp_,
    const double                            fermiEnergyDown_,
    std::shared_ptr<AuxDensityMatrix<memorySpace>> auxDensityMatrixXCPtr)
  {
    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    const bool isGGA = isGradDensityDataDependent;

    const bool isTauMGGA =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getExcFamilyType() ==
       ExcFamilyType::TauMGGA);


    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId);
    const dftfe::uInt totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const dftfe::uInt nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();
    const dftfe::uInt spinPolarizedFactor = 1 + d_dftParamsPtr->spinPolarized;

    if (d_dftParamsPtr->auxBasisTypeXC == "FE")
      {
        std::unordered_map<std::string, std::vector<double>>
                             densityProjectionInputs;
        std::vector<double> &densityValsForXC =
          densityProjectionInputs["densityFunc"];
        densityValsForXC.resize(2 * totalLocallyOwnedCells * nQuadsPerCell, 0);

        if (spinPolarizedFactor == 1)
          {
            for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
              {
                const double *cellRhoValues =
                  densityQuadValues[0].data() + iCell * nQuadsPerCell;

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  densityValsForXC[iCell * nQuadsPerCell + iQuad] =
                    cellRhoValues[iQuad] / 2.0;

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  densityValsForXC[totalLocallyOwnedCells * nQuadsPerCell +
                                   iCell * nQuadsPerCell + iQuad] =
                    cellRhoValues[iQuad] / 2.0;
              }
          }
        else if (spinPolarizedFactor == 2)
          {
            for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
              {
                const double *cellRhoValues =
                  densityQuadValues[0].data() + iCell * nQuadsPerCell;
                const double *cellMagValues =
                  densityQuadValues[1].data() + iCell * nQuadsPerCell;

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  densityValsForXC[iCell * nQuadsPerCell + iQuad] =
                    cellRhoValues[iQuad] / 2.0 + cellMagValues[iQuad] / 2.0;

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  densityValsForXC[totalLocallyOwnedCells * nQuadsPerCell +
                                   iCell * nQuadsPerCell + iQuad] =
                    cellRhoValues[iQuad] / 2.0 - cellMagValues[iQuad] / 2.0;
              }
          }



        if (d_dftParamsPtr->nonLinearCoreCorrection)
          {
            for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
              {
                const std::vector<double> &tempRhoCore =
                  rhoCore.find(d_basisOperationsPtrHost->cellID(iCell))->second;

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  densityValsForXC[iCell * nQuadsPerCell + iQuad] +=
                    tempRhoCore[iQuad] / 2.0;

                for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                  densityValsForXC[totalLocallyOwnedCells * nQuadsPerCell +
                                   iCell * nQuadsPerCell + iQuad] +=
                    tempRhoCore[iQuad] / 2.0;
              }
          }
        if (isGGA)
          {
            std::vector<double> &gradDensityValsForXC =
              densityProjectionInputs["gradDensityFunc"];

            gradDensityValsForXC.resize(2 * totalLocallyOwnedCells *
                                          nQuadsPerCell * 3,
                                        0);


            if (spinPolarizedFactor == 1)
              {
                for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells;
                     ++iCell)
                  {
                    const double *cellGradRhoValues =
                      gradDensityQuadValues[0].data() +
                      iCell * nQuadsPerCell * 3;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt idim = 0; idim < 3; ++idim)
                        gradDensityValsForXC[iCell * nQuadsPerCell * 3 +
                                             iQuad * 3 + idim] =
                          cellGradRhoValues[3 * iQuad + idim] / 2.0;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt idim = 0; idim < 3; ++idim)
                        gradDensityValsForXC[totalLocallyOwnedCells *
                                               nQuadsPerCell * 3 +
                                             iCell * nQuadsPerCell * 3 +
                                             iQuad * 3 + idim] =
                          cellGradRhoValues[3 * iQuad + idim] / 2.0;
                  }
              }
            else if (spinPolarizedFactor == 2)
              {
                for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells;
                     ++iCell)
                  {
                    const double *cellGradRhoValues =
                      gradDensityQuadValues[0].data() +
                      iCell * nQuadsPerCell * 3;
                    const double *cellGradMagValues =
                      gradDensityQuadValues[1].data() +
                      iCell * nQuadsPerCell * 3;


                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt idim = 0; idim < 3; ++idim)
                        gradDensityValsForXC[iCell * nQuadsPerCell * 3 +
                                             iQuad * 3 + idim] =
                          cellGradRhoValues[3 * iQuad + idim] / 2.0 +
                          cellGradMagValues[3 * iQuad + idim] / 2.0;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt idim = 0; idim < 3; ++idim)
                        gradDensityValsForXC[totalLocallyOwnedCells *
                                               nQuadsPerCell * 3 +
                                             iCell * nQuadsPerCell * 3 +
                                             iQuad * 3 + idim] =
                          cellGradRhoValues[3 * iQuad + idim] / 2.0 -
                          cellGradMagValues[3 * iQuad + idim] / 2.0;
                  }
              }


            if (d_dftParamsPtr->nonLinearCoreCorrection)
              {
                for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells;
                     ++iCell)
                  {
                    const std::vector<double> &tempGradRhoCore =
                      gradRhoCore.find(d_basisOperationsPtrHost->cellID(iCell))
                        ->second;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt idim = 0; idim < 3; ++idim)
                        gradDensityValsForXC[iCell * nQuadsPerCell * 3 +
                                             iQuad * 3 + idim] +=
                          tempGradRhoCore[3 * iQuad + idim] / 2.0;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      for (dftfe::uInt idim = 0; idim < 3; ++idim)
                        gradDensityValsForXC[totalLocallyOwnedCells *
                                               nQuadsPerCell * 3 +
                                             iCell * nQuadsPerCell * 3 +
                                             iQuad * 3 + idim] +=
                          tempGradRhoCore[3 * iQuad + idim] / 2.0;
                  }
              }
          }
        if (isTauMGGA)
          {
            std::vector<double> &tauValsForXC =
              densityProjectionInputs["tauFunc"];
            tauValsForXC.resize(2 * totalLocallyOwnedCells * nQuadsPerCell, 0);
            if (spinPolarizedFactor == 1)
              {
                for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells;
                     ++iCell)
                  {
                    const double *cellTauValues =
                      tauQuadValues[0].data() + iCell * nQuadsPerCell;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      tauValsForXC[iCell * nQuadsPerCell + iQuad] =
                        cellTauValues[iQuad] / 2.0;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      tauValsForXC[totalLocallyOwnedCells * nQuadsPerCell +
                                   iCell * nQuadsPerCell + iQuad] =
                        cellTauValues[iQuad] / 2.0;
                  }
              }
            else if (spinPolarizedFactor == 2)
              {
                for (dftfe::uInt iCell = 0; iCell < totalLocallyOwnedCells;
                     ++iCell)
                  {
                    const double *cellTauValues =
                      tauQuadValues[0].data() + iCell * nQuadsPerCell;
                    const double *cellTauMagValues =
                      tauQuadValues[1].data() + iCell * nQuadsPerCell;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      tauValsForXC[iCell * nQuadsPerCell + iQuad] =
                        cellTauValues[iQuad] / 2.0 +
                        cellTauMagValues[iQuad] / 2.0;

                    for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                      tauValsForXC[totalLocallyOwnedCells * nQuadsPerCell +
                                   iCell * nQuadsPerCell + iQuad] =
                        cellTauValues[iQuad] / 2.0 -
                        cellTauMagValues[iQuad] / 2.0;
                  }
              }
            if (d_dftParamsPtr->nonLinearCoreCorrection)
              {
                // std::string errMsg = "NLCC is not completed yet for SCAN.";
                // dftfe::utils::throwException(false, errMsg);
              }
          }

        auto quadPoints = d_basisOperationsPtrHost->quadPoints();

        auto                 quadWeights = d_basisOperationsPtrHost->JxW();
        std::vector<double> &quadPointsStdVec =
          densityProjectionInputs["quadpts"];
        quadPointsStdVec.resize(quadPoints.size());
        std::vector<double> &quadWeightsStdVec =
          densityProjectionInputs["quadWt"];
        quadWeightsStdVec.resize(quadWeights.size());
        for (dftfe::uInt iQuad = 0; iQuad < quadWeightsStdVec.size(); ++iQuad)
          {
            for (dftfe::uInt idim = 0; idim < 3; ++idim)
              quadPointsStdVec[3 * iQuad + idim] = quadPoints[3 * iQuad + idim];
            quadWeightsStdVec[iQuad] = std::real(quadWeights[iQuad]);
          }


        auxDensityMatrixXCPtr->projectDensityStart(densityProjectionInputs);

        auxDensityMatrixXCPtr->projectDensityEnd(mpi_communicator);


        std::shared_ptr<AuxDensityMatrixFE<memorySpace>>
          auxDensityMatrixXCFEPtr =
            std::dynamic_pointer_cast<AuxDensityMatrixFE<memorySpace>>(
              auxDensityMatrixXCPtr);

        Assert(
          auxDensityMatrixXCFEPtr != nullptr,
          dealii::ExcMessage(
            "DFT-FE Error: unable to type cast the auxiliary matrix to FE."));

        auxDensityMatrixXCFEPtr->setDensityMatrixComponents(
          eigenVectorsFlattenedMemSpace, d_partialOccupancies);
      }
    else if (d_dftParamsPtr->auxBasisTypeXC == "SLATER")
      {
#ifndef USE_COMPLEX
        auto basisOpMemSpace     = getBasisOperationsMemSpace();
        auto blasWrapperMemSpace = getBLASWrapperMemSpace();
        computeAuxProjectedDensityMatrixFromPSI(eigenVectorsFlattenedMemSpace,
                                                d_numEigenValues,
                                                d_partialOccupancies,
                                                basisOpMemSpace,
                                                blasWrapperMemSpace,
                                                d_densityDofHandlerIndex,
                                                d_gllQuadratureId,
                                                d_kPointWeights,
                                                *auxDensityMatrixXCPtr,
                                                d_mpiCommParent,
                                                mpi_communicator,
                                                interpoolcomm,
                                                interBandGroupComm,
                                                *d_dftParamsPtr);
#endif
      }
  }

#include "dft.inst.cc"
} // namespace dftfe
