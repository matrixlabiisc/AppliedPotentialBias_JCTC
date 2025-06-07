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
// @author Kartick Ramakrishnan
//
#include <oncvClass.h>
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  oncvClass<ValueType, memorySpace>::oncvClass(
    const MPI_Comm                           &mpi_comm_parent,
    const std::string                        &scratchFolderName,
    const std::set<dftfe::uInt>              &atomTypes,
    const bool                                floatingNuclearCharges,
    const dftfe::uInt                         nOMPThreads,
    const std::map<dftfe::uInt, dftfe::uInt> &atomAttributes,
    const bool                                reproducibleOutput,
    const dftfe::Int                          verbosity,
    const bool                                useDevice,
    const bool                                memOptMode)
    : d_mpiCommParent(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_dftfeScratchFolderName = scratchFolderName;
    d_atomTypes              = atomTypes;
    d_floatingNuclearCharges = floatingNuclearCharges;
    d_nOMPThreads            = nOMPThreads;
    d_reproducible_output    = reproducibleOutput;
    d_verbosity              = verbosity;
    d_atomTypeAtributes      = atomAttributes;
    d_useDevice              = useDevice;
    d_memoryOptMode          = memOptMode;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType,
            memorySpace>::createAtomCenteredSphericalFunctionsForDensities()
  {
    d_atomicCoreDensityVector.clear();
    d_atomicCoreDensityVector.resize(d_nOMPThreads);
    d_atomicValenceDensityVector.clear();
    d_atomicValenceDensityVector.resize(d_nOMPThreads);

    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt atomicNumber = *it;
        char        valenceDataFile[256];
        strcpy(valenceDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/density.inp")
                 .c_str());
        char coreDataFile[256];
        strcpy(coreDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/coreDensity.inp")
                 .c_str());

        for (dftfe::uInt i = 0; i < d_nOMPThreads; i++)
          {
            d_atomicValenceDensityVector[i][*it] = std::make_shared<
              AtomCenteredSphericalFunctionValenceDensitySpline>(
              valenceDataFile, 1E-10, false);
            d_atomicCoreDensityVector[i][*it] =
              std::make_shared<AtomCenteredSphericalFunctionCoreDensitySpline>(
                coreDataFile, 1E-12, true);
          }
        if (d_atomicCoreDensityVector[0][atomicNumber]->isDataPresent())
          d_atomTypeCoreFlagMap[atomicNumber] = true;
        else
          d_atomTypeCoreFlagMap[atomicNumber] = false;
      } //*it loop
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::initialise(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsDevicePtr,
#endif
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtrDevice,
#endif
    dftfe::uInt                              densityQuadratureId,
    dftfe::uInt                              localContributionQuadratureId,
    dftfe::uInt                              sparsityPatternQuadratureId,
    dftfe::uInt                              nlpspQuadratureId,
    dftfe::uInt                              densityQuadratureIdElectro,
    std::shared_ptr<excManager<memorySpace>> excFunctionalPtr,
    const std::vector<std::vector<double>>  &atomLocations,
    dftfe::uInt                              numEigenValues,
    const bool                               singlePrecNonLocalOperator,
    const bool computeSphericalFnTimesXNonLocalOperator)
  {
    MPI_Barrier(d_mpiCommParent);
    d_BasisOperatorHostPtr = basisOperationsHostPtr;
    d_BLASWrapperHostPtr   = BLASWrapperPtrHost;
#if defined(DFTFE_WITH_DEVICE)
    d_BLASWrapperDevicePtr   = BLASWrapperPtrDevice;
    d_BasisOperatorDevicePtr = basisOperationsDevicePtr;
#endif


    std::vector<dftfe::uInt> atomicNumbers;
    for (dftfe::Int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
      }

    d_densityQuadratureId           = densityQuadratureId;
    d_localContributionQuadratureId = localContributionQuadratureId;
    d_densityQuadratureIdElectro    = densityQuadratureIdElectro;
    d_sparsityPatternQuadratureId   = sparsityPatternQuadratureId;
    d_nlpspQuadratureId             = nlpspQuadratureId;
    d_excManagerPtr                 = excFunctionalPtr;
    d_numEigenValues                = numEigenValues;
    d_singlePrecNonLocalOperator    = singlePrecNonLocalOperator;

    createAtomCenteredSphericalFunctionsForDensities();
    createAtomCenteredSphericalFunctionsForProjectors();
    createAtomCenteredSphericalFunctionsForLocalPotential();

    d_atomicProjectorFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicProjectorFnsContainer->init(atomicNumbers, d_atomicProjectorFnsMap);

    if (!d_useDevice)
      {
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperHostPtr,
            d_BasisOperatorHostPtr,
            d_atomicProjectorFnsContainer,
            d_mpiCommParent,
            d_memoryOptMode,
            computeSphericalFnTimesXNonLocalOperator);
        if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
          if (d_singlePrecNonLocalOperator)
            d_nonLocalOperatorSinglePrec =
              std::make_shared<AtomicCenteredNonLocalOperator<
                typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                memorySpace>>(d_BLASWrapperHostPtr,
                              d_BasisOperatorHostPtr,
                              d_atomicProjectorFnsContainer,
                              d_mpiCommParent,
                              d_memoryOptMode,
                              computeSphericalFnTimesXNonLocalOperator);
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          d_nonLocalOperator = std::make_shared<
            AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
            d_BLASWrapperDevicePtr,
            d_BasisOperatorDevicePtr,
            d_atomicProjectorFnsContainer,
            d_mpiCommParent,
            d_memoryOptMode,
            computeSphericalFnTimesXNonLocalOperator);
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          if (d_singlePrecNonLocalOperator)
            d_nonLocalOperatorSinglePrec =
              std::make_shared<AtomicCenteredNonLocalOperator<
                typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                memorySpace>>(d_BLASWrapperDevicePtr,
                              d_BasisOperatorDevicePtr,
                              d_atomicProjectorFnsContainer,
                              d_mpiCommParent,
                              d_memoryOptMode,
                              computeSphericalFnTimesXNonLocalOperator);
      }
#endif

    computeNonlocalPseudoPotentialConstants();
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double>              &kPointWeights,
    const std::vector<double>              &kPointCoordinates,
    const bool                              updateNonlocalSparsity)
  {
    std::vector<dftfe::uInt> atomicNumbers;
    std::vector<double>      atomCoords;


    for (dftfe::Int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
        for (dftfe::Int dim = 2; dim < 5; dim++)
          atomCoords.push_back(atomLocations[iAtom][dim]);
      }


    d_atomicProjectorFnsContainer->initaliseCoordinates(atomCoords,
                                                        periodicCoords,
                                                        imageIds);


    if (updateNonlocalSparsity)
      {
        d_HamiltonianCouplingMatrixEntriesUpdated           = false;
        d_HamiltonianCouplingMatrixSinglePrecEntriesUpdated = false;
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicProjectorFnsContainer->computeSparseStructure(
          d_BasisOperatorHostPtr, d_sparsityPatternQuadratureId, 1E-8, 0);
        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "ONCVclass: Time taken for computeSparseStructureNonLocalProjectors: "
            << TotalTime << std::endl;
      }
    MPI_Barrier(d_mpiCommParent);
    double InitTimeTotal = MPI_Wtime();
    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_BLASWrapperHostPtr,
      d_nlpspQuadratureId);
    if (d_singlePrecNonLocalOperator)
      d_nonLocalOperatorSinglePrec
        ->copyPartitionerKPointsAndComputeCMatrixEntries(updateNonlocalSparsity,
                                                         kPointWeights,
                                                         kPointCoordinates,
                                                         d_BasisOperatorHostPtr,
                                                         d_BLASWrapperHostPtr,
                                                         d_nlpspQuadratureId,
                                                         d_nonLocalOperator);


    MPI_Barrier(d_mpiCommParent);
    double TotalTime = MPI_Wtime() - InitTimeTotal;
    if (d_verbosity >= 2)
      pcout << "ONCVclass: Time taken for non local psp init: " << TotalTime
            << std::endl;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::initialiseNonLocalContribution(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    const std::vector<double>              &kPointWeights,
    const std::vector<double>              &kPointCoordinates,
    const bool                              updateNonlocalSparsity,
    const std::map<dftfe::uInt, std::vector<dftfe::Int>> &sparsityPattern,
    const std::vector<std::vector<dealii::CellId>>
      &elementIdsInAtomCompactSupport,
    const std::vector<std::vector<dftfe::uInt>>
                                   &elementIndexesInAtomCompactSupport,
    const std::vector<dftfe::uInt> &atomIdsInCurrentProcess,
    dftfe::uInt                     numberElements)
  {
    std::vector<dftfe::uInt> atomicNumbers;
    std::vector<double>      atomCoords;


    for (dftfe::Int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        atomicNumbers.push_back(atomLocations[iAtom][0]);
        for (dftfe::Int dim = 2; dim < 5; dim++)
          atomCoords.push_back(atomLocations[iAtom][dim]);
      }


    d_atomicProjectorFnsContainer->initaliseCoordinates(atomCoords,
                                                        periodicCoords,
                                                        imageIds);


    if (updateNonlocalSparsity)
      {
        d_HamiltonianCouplingMatrixEntriesUpdated           = false;
        d_HamiltonianCouplingMatrixSinglePrecEntriesUpdated = false;
        MPI_Barrier(d_mpiCommParent);
        double InitTime = MPI_Wtime();
        d_atomicProjectorFnsContainer->getDataForSparseStructure(
          sparsityPattern,
          elementIdsInAtomCompactSupport,
          elementIndexesInAtomCompactSupport,
          atomIdsInCurrentProcess,
          numberElements);

        MPI_Barrier(d_mpiCommParent);
        double TotalTime = MPI_Wtime() - InitTime;
        if (d_verbosity >= 2)
          pcout
            << "ONCVclass: Time taken for computeSparseStructureNonLocalProjectors: "
            << TotalTime << std::endl;
      }
    MPI_Barrier(d_mpiCommParent);
    double InitTimeTotal = MPI_Wtime();
    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      d_BLASWrapperHostPtr,
      d_nlpspQuadratureId);
    if (d_singlePrecNonLocalOperator)
      d_nonLocalOperatorSinglePrec
        ->intitialisePartitionerKPointsAndComputeCMatrixEntries(
          updateNonlocalSparsity,
          kPointWeights,
          kPointCoordinates,
          d_BasisOperatorHostPtr,
          d_BLASWrapperHostPtr,
          d_nlpspQuadratureId);

    MPI_Barrier(d_mpiCommParent);
    double TotalTime = MPI_Wtime() - InitTimeTotal;
    if (d_verbosity >= 2)
      pcout << "ONCVclass: Time taken for non local psp init: " << TotalTime
            << std::endl;
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::computeNonlocalPseudoPotentialConstants()
  {
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        const dftfe::uInt Znum = *it;
        const std::map<std::pair<dftfe::uInt, dftfe::uInt>,
                       std::shared_ptr<AtomCenteredSphericalFunctionBase>>
          sphericalFunction =
            d_atomicProjectorFnsContainer->getSphericalFunctions();
        dftfe::uInt numRadProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfRadialSphericalFunctionsPerAtom(Znum);
        dftfe::uInt numTotalProjectors =
          d_atomicProjectorFnsContainer
            ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
        char denominatorDataFileName[256];
        strcpy(denominatorDataFileName,
               (d_dftfeScratchFolderName + "/z" + std::to_string(Znum) + "/" +
                "denom.dat")
                 .c_str());
        std::vector<std::vector<double>> denominator(0);
        dftUtils::readFile(numRadProjectors,
                           denominator,
                           denominatorDataFileName);
        std::vector<double> pseudoPotentialConstants(numTotalProjectors, 0.0);
        dftfe::uInt         ProjId = 0;
        for (dftfe::uInt iProj = 0; iProj < numRadProjectors; iProj++)
          {
            std::shared_ptr<AtomCenteredSphericalFunctionBase> sphFn =
              sphericalFunction.find(std::make_pair(Znum, iProj))->second;
            dftfe::uInt lQuantumNumber = sphFn->getQuantumNumberl();
            for (dftfe::Int l = 0; l < 2 * lQuantumNumber + 1; l++)
              {
                pseudoPotentialConstants[ProjId] = denominator[iProj][iProj];
                ProjId++;
              }
          }
        d_atomicNonLocalPseudoPotentialConstants[Znum] =
          pseudoPotentialConstants;

      } //*it
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType,
            memorySpace>::createAtomCenteredSphericalFunctionsForProjectors()
  {
    d_atomicProjectorFnsVector.clear();
    std::vector<std::vector<dftfe::Int>> projectorIdDetails;
    std::vector<std::vector<dftfe::Int>> atomicFunctionIdDetails;
    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        char        pseudoAtomDataFile[256];
        dftfe::uInt cumulativeSplineId = 0;
        strcpy(pseudoAtomDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/PseudoAtomDat")
                 .c_str());

        dftfe::uInt   Znum = *it;
        std::ifstream readPseudoDataFileNames(pseudoAtomDataFile);
        dftfe::uInt   numberOfProjectors;
        readPseudoDataFileNames >> numberOfProjectors;
        readPseudoDataFileNames.ignore();
        projectorIdDetails.resize(numberOfProjectors);
        std::string          readLine;
        std::set<dftfe::Int> radFunctionIds;
        atomicFunctionIdDetails.resize(numberOfProjectors);
        for (dftfe::uInt i = 0; i < numberOfProjectors; ++i)
          {
            std::vector<dftfe::Int> &radAndAngularFunctionId =
              atomicFunctionIdDetails[i];
            radAndAngularFunctionId.resize(3, 0);
            std::getline(readPseudoDataFileNames, readLine);

            std::istringstream lineString(readLine);
            dftfe::uInt        count = 0;
            dftfe::Int         Id;
            double             mollifierRadius;
            std::string        dummyString;
            while (lineString >> dummyString)
              {
                if (count < 3)
                  {
                    Id = atoi(dummyString.c_str());

                    if (count == 1)
                      radFunctionIds.insert(Id);
                    radAndAngularFunctionId[count] = Id;
                  }

                if (count > 3)
                  {
                    std::cerr << "Invalid argument in the SingleAtomData file"
                              << std::endl;
                    exit(-1);
                  }

                count++;
              }
          }
        std::string tempProjRadialFunctionFileName;
        dftfe::uInt numProj;
        dftfe::uInt alpha = 0;
        for (std::set<dftfe::Int>::iterator i = radFunctionIds.begin();
             i != radFunctionIds.end();
             ++i)
          {
            char        projRadialFunctionFileName[512];
            dftfe::uInt lQuantumNo = *i;
            readPseudoDataFileNames >> tempProjRadialFunctionFileName;
            readPseudoDataFileNames >> numProj;
            strcpy(projRadialFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/" + tempProjRadialFunctionFileName)
                     .c_str());

            //
            // 2D vector to store the radial coordinate and its
            // corresponding function value
            std::vector<std::vector<double>> radialFunctionData(0);

            //
            // read the radial function file
            //
            dftUtils::readFile(numProj + 1,
                               radialFunctionData,
                               projRadialFunctionFileName);

            for (dftfe::Int j = 1; j < numProj + 1; j++)
              {
                d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)] =
                  std::make_shared<
                    AtomCenteredSphericalFunctionProjectorSpline>(
                    projRadialFunctionFileName,
                    lQuantumNo,
                    0,
                    j,
                    numProj + 1,
                    1E-12);
                alpha++;
              }
          } // i loop

      } // for loop *it
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::
    createAtomCenteredSphericalFunctionsForLocalPotential()
  {
    d_atomicLocalPotVector.clear();
    d_atomicLocalPotVector.resize(d_nOMPThreads);

    for (std::set<dftfe::uInt>::iterator it = d_atomTypes.begin();
         it != d_atomTypes.end();
         ++it)
      {
        dftfe::uInt atomicNumber = *it;
        char        LocalDataFile[256];
        strcpy(LocalDataFile,
               (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                "/locPot.dat")
                 .c_str());
        for (dftfe::uInt i = 0; i < d_nOMPThreads; i++)
          d_atomicLocalPotVector[i][*it] =
            std::make_shared<AtomCenteredSphericalFunctionLocalPotentialSpline>(
              LocalDataFile,
              d_atomTypeAtributes[*it],
              d_reproducible_output ? 1.0e-8 : 1.0e-7,
              d_reproducible_output ? 8.0001 : 10.0001);

      } //*it loop
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  oncvClass<ValueType, memorySpace>::getRadialValenceDensity(dftfe::uInt Znum,
                                                             double      rad)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    double      Value =
      d_atomicValenceDensityVector[threadId][Znum]->getRadialValue(rad);

    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::getRadialValenceDensity(
    dftfe::uInt          Znum,
    double               rad,
    std::vector<double> &Val)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicValenceDensityVector[threadId][Znum]->getDerivativeValue(rad);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  oncvClass<ValueType, memorySpace>::getRmaxValenceDensity(dftfe::uInt Znum)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    return (d_atomicValenceDensityVector[threadId][Znum]->getRadialCutOff());
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  oncvClass<ValueType, memorySpace>::getRmaxCoreDensity(dftfe::uInt Znum)
  {
    dftfe::uInt threadId = omp_get_thread_num();

    return (d_atomicCoreDensityVector[threadId][Znum]->getRadialCutOff());
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  oncvClass<ValueType, memorySpace>::getRadialCoreDensity(dftfe::uInt Znum,
                                                          double      rad)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    double      Value =
      d_atomicCoreDensityVector[threadId][Znum]->getRadialValue(rad);
    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::getRadialCoreDensity(
    dftfe::uInt          Znum,
    double               rad,
    std::vector<double> &Val)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    Val.clear();
    Val = d_atomicCoreDensityVector[threadId][Znum]->getDerivativeValue(rad);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  oncvClass<ValueType, memorySpace>::getRadialLocalPseudo(dftfe::uInt Znum,
                                                          double      rad)
  {
    dftfe::uInt threadId = omp_get_thread_num();
    double Value = d_atomicLocalPotVector[threadId][Znum]->getRadialValue(rad);
    return (Value);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  oncvClass<ValueType, memorySpace>::getRmaxLocalPot(dftfe::uInt Znum)
  {
    return (d_atomicLocalPotVector[0][Znum]->getRadialCutOff());
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  bool
  oncvClass<ValueType, memorySpace>::coreNuclearDensityPresent(dftfe::uInt Znum)
  {
    return (d_atomTypeCoreFlagMap[Znum]);
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  oncvClass<ValueType, memorySpace>::setImageCoordinates(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &periodicCoords,
    std::vector<dftfe::uInt>               &imageIdsTemp,
    std::vector<double>                    &imageCoordsTemp)
  {
    imageIdsTemp.clear();
    imageCoordsTemp.clear();
    imageCoordsTemp.resize(imageIds.size() * 3, 0.0);
    std::vector<dftfe::uInt> imageLoc(dftfe::Int(atomLocations.size()), 0.0);
    for (dftfe::Int jImage = 0; jImage < imageIds.size(); jImage++)
      {
        dftfe::uInt atomId = (imageIds[jImage]);
        imageIdsTemp.push_back(atomId);
        dftfe::Int startLoc = imageLoc[atomId];
        imageCoordsTemp[3 * jImage + 0] =
          periodicCoords[atomId][3 * startLoc + 0];
        imageCoordsTemp[3 * jImage + 1] =
          periodicCoords[atomId][3 * startLoc + 1];
        imageCoordsTemp[3 * jImage + 2] =
          periodicCoords[atomId][3 * startLoc + 2];
        imageLoc[atomId] += 1;
      }
  }
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
  oncvClass<ValueType, memorySpace>::getCouplingMatrix(
    CouplingType couplingtype)
  {
    std::vector<ValueType> Entries;
    if (!d_HamiltonianCouplingMatrixEntriesUpdated)
      {
        const std::vector<dftfe::uInt> atomIdsInProcessor =
          d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
        std::vector<dftfe::uInt> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        d_couplingMatrixEntries.clear();

        for (dftfe::Int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            dftfe::uInt atomId = atomIdsInProcessor[iAtom];
            dftfe::uInt Znum   = atomicNumber[atomId];
            dftfe::uInt numberSphericalFunctions =
              d_atomicProjectorFnsContainer
                ->getTotalNumberOfSphericalFunctionsPerAtom(Znum);
            for (dftfe::uInt alpha = 0; alpha < numberSphericalFunctions;
                 alpha++)
              {
                double V =
                  d_atomicNonLocalPseudoPotentialConstants[Znum][alpha];
                Entries.push_back(ValueType(V));
              }
          }
      }
    if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
      {
        if (!d_HamiltonianCouplingMatrixEntriesUpdated)
          {
            d_couplingMatrixEntries.resize(Entries.size());
            d_couplingMatrixEntries.copyFrom(Entries);
            d_HamiltonianCouplingMatrixEntriesUpdated = true;
          }

        return (d_couplingMatrixEntries);
      }
#if defined(DFTFE_WITH_DEVICE)
    else
      {
        if (!d_HamiltonianCouplingMatrixEntriesUpdated)
          {
            std::vector<ValueType> EntriesPadded;
            d_nonLocalOperator->paddingCouplingMatrix(
              Entries, EntriesPadded, CouplingStructure::diagonal);
            d_couplingMatrixEntries.resize(EntriesPadded.size());
            d_couplingMatrixEntries.copyFrom(EntriesPadded);
            d_HamiltonianCouplingMatrixEntriesUpdated = true;
          }
        return (d_couplingMatrixEntries);
      }
#endif
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<
    typename dftfe::dataTypes::singlePrecType<ValueType>::type,
    memorySpace> &
  oncvClass<ValueType, memorySpace>::getCouplingMatrixSinglePrec(
    CouplingType couplingtype)
  {
    getCouplingMatrix(couplingtype);
    if (!d_HamiltonianCouplingMatrixSinglePrecEntriesUpdated)
      {
        d_couplingMatrixEntriesSinglePrec.resize(
          d_couplingMatrixEntries.size());
        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          d_BLASWrapperHostPtr->copyValueType1ArrToValueType2Arr(
            d_couplingMatrixEntriesSinglePrec.size(),
            d_couplingMatrixEntries.data(),
            d_couplingMatrixEntriesSinglePrec.data());
#if defined(DFTFE_WITH_DEVICE)
        else
          d_BLASWrapperDevicePtr->copyValueType1ArrToValueType2Arr(
            d_couplingMatrixEntriesSinglePrec.size(),
            d_couplingMatrixEntries.data(),
            d_couplingMatrixEntriesSinglePrec.data());
#endif

        d_HamiltonianCouplingMatrixSinglePrecEntriesUpdated = true;
      }

    return (d_couplingMatrixEntriesSinglePrec);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  oncvClass<ValueType, memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<
    typename dftfe::dataTypes::singlePrecType<ValueType>::type,
    memorySpace>>
  oncvClass<ValueType, memorySpace>::getNonLocalOperatorSinglePrec()
  {
    return d_nonLocalOperatorSinglePrec;
  }



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  oncvClass<ValueType, memorySpace>::getTotalNumberOfAtomsInCurrentProcessor()
  {
    return d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess().size();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  oncvClass<ValueType, memorySpace>::getAtomIdInCurrentProcessor(
    dftfe::uInt iAtom)
  {
    std::vector<dftfe::uInt> atomIdList =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    return (atomIdList[iAtom]);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::uInt
  oncvClass<ValueType, memorySpace>::
    getTotalNumberOfSphericalFunctionsForAtomId(dftfe::uInt atomId)
  {
    std::vector<dftfe::uInt> atomicNumbers =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    return (
      d_atomicProjectorFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
        atomicNumbers[atomId]));
  }
  template class oncvClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class oncvClass<dataTypes::number,
                           dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
