// ---------------------------------------------------------------------
//
// Copyright (c) 2017-18 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das
//
#include <dft.h>
#include <force.h>
#include <eshelbyTensor.h>
#include <eshelbyTensorSpinPolarized.h>
#include <dftUtils.h>
#include <forceWfcContractions.h>

namespace dftfe
{
  //
  // compute configurational force contribution from all terms except the
  // nuclear self energy
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::
    computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const dftfe::uInt                    eigenDofHandlerIndex,
      const dftfe::uInt                    smearedChargeQuadratureId,
      const dftfe::uInt                    lpspQuadratureIdElectro,
      const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
      const dftfe::uInt                    phiTotDofHandlerIndexElectro,
      const distributedCPUVec<double>     &phiTotRhoOutElectro,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoTotalOutValuesLpsp,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradRhoTotalOutValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                                                          &hessianRhoCoreAtoms,
      const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                             &pseudoVLocAtomsElectro,
      const vselfBinsManager &vselfBinsManagerElectro)
  {
    int this_process;
    MPI_Comm_rank(d_mpiCommParent, &this_process);
    MPI_Barrier(d_mpiCommParent);
    double forcetotal_time = MPI_Wtime();

    MPI_Barrier(d_mpiCommParent);
    double init_time = MPI_Wtime();

    const dftfe::uInt numberGlobalAtoms = dftPtr->atomLocations.size();
    std::map<dftfe::uInt, std::vector<double>> forceContributionFnlGammaAtoms;
    std::map<dftfe::uInt, std::vector<double>>
      forceContributionFnlGammaAtomsHubbard;

    const bool isPseudopotential = d_dftParams.isPseudopotential;

    FEEvaluationWrapperClass<3> forceEval(1,
                                          d_dftParams.densityQuadratureRule,
                                          matrixFreeData,
                                          d_forceDofHandlerIndex,
                                          dftPtr->d_densityQuadratureId);
    FEEvaluationWrapperClass<3> forceEvalNLP(
      1,
      C_num1DQuadNLPSP(d_dftParams.finiteElementPolynomialOrder) *
        C_numCopies1DQuadNLPSP(),
      matrixFreeData,
      d_forceDofHandlerIndex,
      dftPtr->d_nlpspQuadratureId);


    std::map<dftfe::uInt, std::vector<double>>
      forceContributionShadowLocalGammaAtoms;

    const dftfe::uInt numQuadPoints    = forceEval.n_q_points;
    const dftfe::uInt numQuadPointsNLP = forceEvalNLP.n_q_points;

    const dftfe::uInt numEigenVectors = dftPtr->d_numEigenValues;
    const dftfe::uInt numKPoints      = dftPtr->d_kPointWeights.size();
    dealii::DoFHandler<3>::active_cell_iterator           subCellPtr;
    dealii::Tensor<1, 2, dealii::VectorizedArray<double>> zeroTensor1;
    zeroTensor1[0] = dealii::make_vectorized_array(0.0);
    zeroTensor1[1] = dealii::make_vectorized_array(0.0);
    dealii::Tensor<1, 2, dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                                                          zeroTensor2;
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor4;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        zeroTensor2[0][idim] = dealii::make_vectorized_array(0.0);
        zeroTensor2[1][idim] = dealii::make_vectorized_array(0.0);
        zeroTensor3[idim]    = dealii::make_vectorized_array(0.0);
      }
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
          {
            zeroTensor4[idim][jdim] = dealii::make_vectorized_array(0.0);
          }
      }

    const double spinPolarizedFactor =
      (d_dftParams.spinPolarized == 1) ? 0.5 : 1.0;
    const dealii::VectorizedArray<double> spinPolarizedFactorVect =
      (d_dftParams.spinPolarized == 1) ? dealii::make_vectorized_array(0.5) :
                                         dealii::make_vectorized_array(1.0);

    const dftfe::uInt numPhysicalCells = matrixFreeData.n_physical_cells();

    std::map<dealii::CellId, dftfe::uInt> cellIdToCellNumberMap;

    dealii::DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler
                                                         .begin_active(),
                                                endc = dftPtr->dofHandler.end();
    dftfe::uInt iElem                                = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cellIdToCellNumberMap[cell->id()] = iElem;
          iElem++;
        }



    // band group parallelization data structures
    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               numEigenVectors,
                                               bandGroupLowHighPlusOneIndices);


    const dftfe::uInt localVectorSize =
      matrixFreeData.get_vector_partitioner()->locally_owned_size();

    const dftfe::uInt numMacroCells = matrixFreeData.n_cell_batches();


    std::vector<std::vector<double>> partialOccupancies(
      numKPoints,
      std::vector<double>((1 + d_dftParams.spinPolarized) * numEigenVectors,
                          0.0));
    for (dftfe::uInt spinIndex = 0; spinIndex < (1 + d_dftParams.spinPolarized);
         ++spinIndex)
      for (dftfe::uInt kPoint = 0; kPoint < numKPoints; ++kPoint)
        for (dftfe::uInt iWave = 0; iWave < numEigenVectors; ++iWave)
          partialOccupancies[kPoint][numEigenVectors * spinIndex + iWave] =
            dftPtr->d_partialOccupancies[kPoint]
                                        [numEigenVectors * spinIndex + iWave];

    MPI_Barrier(d_mpiCommParent);
    init_time = MPI_Wtime() - init_time;

    for (dftfe::uInt spinIndex = 0; spinIndex < (1 + d_dftParams.spinPolarized);
         ++spinIndex)
      {
        std::vector<double> elocWfcEshelbyTensorQuadValuesH(
          numKPoints * numPhysicalCells * numQuadPoints * 9, 0.0);
        std::vector<dataTypes::number>
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened;

        std::vector<dataTypes::number>
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard;

#ifdef USE_COMPLEX
        std::vector<dataTypes::number>
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened;

        std::vector<dataTypes::number>
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHubbard;
#endif

        if (isPseudopotential)
          {
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
              .resize(numKPoints *
                        dftPtr->d_oncvClassPtr->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPointsNLP * 3,
                      dataTypes::number(0.0));

#ifdef USE_COMPLEX
            projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
              .resize(numKPoints *
                        dftPtr->d_oncvClassPtr->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPointsNLP,
                      dataTypes::number(0.0));
#endif
          }

        if (dftPtr->isHubbardCorrectionsUsed())
          {
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard
              .resize(numKPoints *
                        dftPtr->getHubbardClassPtr()
                          ->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPointsNLP * 3,
                      dataTypes::number(0.0));

#ifdef USE_COMPLEX
            projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHubbard
              .resize(numKPoints *
                        dftPtr->getHubbardClassPtr()
                          ->getNonLocalOperator()
                          ->getTotalNonTrivialSphericalFnsOverAllCells() *
                        numQuadPointsNLP,
                      dataTypes::number(0.0));
#endif
          }



#if defined(DFTFE_WITH_DEVICE)
        if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
          {
            MPI_Barrier(d_mpiCommParent);
            double device_time = MPI_Wtime();
            force::wfcContractionsForceKernelsAllH(
              dftPtr->d_basisOperationsPtrDevice,
              dftPtr->d_densityQuadratureId,
              dftPtr->d_nlpspQuadratureId,
              dftPtr->d_BLASWrapperPtr,
              dftPtr->d_oncvClassPtr,
              dftPtr->getHubbardClassPtr(),
              dftPtr->isHubbardCorrectionsUsed(),
              dftPtr->d_eigenVectorsFlattenedDevice.begin(),
              d_dftParams.spinPolarized,
              spinIndex,
              dftPtr->eigenValues,
              partialOccupancies,
              dftPtr->d_kPointCoordinates,
              localVectorSize,
              numEigenVectors,
              numPhysicalCells,
              numQuadPoints,
              numQuadPointsNLP,
              elocWfcEshelbyTensorQuadValuesH.data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                .data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard
                .data(),
#  ifdef USE_COMPLEX
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                .data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHubbard
                .data(),
#  endif
              d_mpiCommParent,
              dftPtr->interBandGroupComm,
              isPseudopotential,
              d_dftParams.floatingNuclearCharges,
              false,
              d_dftParams);

            MPI_Barrier(d_mpiCommParent);
            device_time = MPI_Wtime() - device_time;

            if (this_process == 0 && d_dftParams.verbosity >= 4)
              std::cout << "Time for wfc contractions in forces: "
                        << device_time << std::endl;
          }
        else
#endif
          {
            MPI_Barrier(d_mpiCommParent);
            double host_time = MPI_Wtime();

            force::wfcContractionsForceKernelsAllH(
              dftPtr->d_basisOperationsPtrHost,
              dftPtr->d_densityQuadratureId,
              dftPtr->d_nlpspQuadratureId,
              dftPtr->d_BLASWrapperPtrHost,
              dftPtr->d_oncvClassPtr,
              dftPtr->getHubbardClassPtr(),
              dftPtr->isHubbardCorrectionsUsed(),
              dftPtr->d_eigenVectorsFlattenedHost.begin(),
              d_dftParams.spinPolarized,
              spinIndex,
              dftPtr->eigenValues,
              partialOccupancies,
              dftPtr->d_kPointCoordinates,
              localVectorSize,
              numEigenVectors,
              numPhysicalCells,
              numQuadPoints,
              numQuadPointsNLP,
              elocWfcEshelbyTensorQuadValuesH.data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                .data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard
                .data(),
#ifdef USE_COMPLEX
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                .data(),
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHubbard
                .data(),
#endif
              d_mpiCommParent,
              dftPtr->interBandGroupComm,
              isPseudopotential,
              d_dftParams.floatingNuclearCharges,
              false,
              d_dftParams);

            /*
            double
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbardNorm
            = 0.0; for ( dftfe::uInt i= 0 ; i <
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard.size();
            i++)
              {
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbardNorm
            +=
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard[i]*
                                                                                                     projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard[i];
              }

            MPI_Allreduce(MPI_IN_PLACE,
                          &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbardNorm,
                          1,
                          MPI_DOUBLE,
                          MPI_SUM,
                          dftPtr->mpi_communicator);

            std::cout<<" Norm of
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbardNorm
            =
            "<<projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbardNorm<<"\n";

      */
            MPI_Barrier(d_mpiCommParent);
            host_time = MPI_Wtime() - host_time;

            if (this_process == 0 && d_dftParams.verbosity >= 4)
              std::cout << "Time for wfc contractions in forces: " << host_time
                        << std::endl;
          }


        if (!d_dftParams.floatingNuclearCharges)
          {
            dealii::AlignedVector<
              dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
              EQuad(numQuadPoints, zeroTensor4);
            for (dftfe::uInt cell = 0; cell < matrixFreeData.n_cell_batches();
                 ++cell)
              {
                forceEval.reinit(cell);

                std::fill(EQuad.begin(), EQuad.end(), zeroTensor4);

                const dftfe::uInt numSubCells =
                  matrixFreeData.n_active_entries_per_cell_batch(cell);

                for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                     ++iSubCell)
                  {
                    subCellPtr =
                      matrixFreeData.get_cell_iterator(cell, iSubCell);
                    const dftfe::uInt physicalCellId =
                      cellIdToCellNumberMap[subCellPtr->id()];
                    for (dftfe::uInt kPoint = 0; kPoint < numKPoints; ++kPoint)
                      {
                        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                          {
                            const dftfe::uInt id =
                              kPoint * numPhysicalCells * numQuadPoints * 9 +
                              physicalCellId * numQuadPoints * 9 + q * 9;
                            EQuad[q][0][0][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 0];
                            EQuad[q][0][1][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 1];
                            EQuad[q][0][2][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 2];
                            EQuad[q][1][0][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 3];
                            EQuad[q][1][1][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 4];
                            EQuad[q][1][2][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 5];
                            EQuad[q][2][0][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 6];
                            EQuad[q][2][1][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 7];
                            EQuad[q][2][2][iSubCell] +=
                              dftPtr->d_kPointWeights[kPoint] *
                              elocWfcEshelbyTensorQuadValuesH[id + 8];
                          } // quad loop
                      }     // kpoint loop
                  }         // subcell loop


                for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                  forceEval.submit_gradient(spinPolarizedFactorVect * EQuad[q],
                                            q);

                forceEval.integrate(dealii::EvaluationFlags::gradients);
#ifdef USE_COMPLEX
                forceEval.distribute_local_to_global(
                  d_configForceVectorLinFEKPoints);
#else
                forceEval.distribute_local_to_global(d_configForceVectorLinFE);
#endif
              } // cell loop
          }

        if (isPseudopotential)
          {
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              FVectQuads(numQuadPointsNLP, zeroTensor3);

            const dftfe::uInt numNonLocalAtomsCurrentProcessPsP =
              (dftPtr->d_oncvClassPtr->getNonLocalOperator()
                 ->getTotalAtomInCurrentProcessor());

            std::vector<dftfe::Int> nonLocalAtomIdPsP,
              globalChargeIdNonLocalAtomPsP;
            nonLocalAtomIdPsP.resize(numNonLocalAtomsCurrentProcessPsP);
            globalChargeIdNonLocalAtomPsP.resize(
              numNonLocalAtomsCurrentProcessPsP);

            std::vector<dftfe::uInt> numberPseudoWaveFunctionsPerAtomPsP;
            numberPseudoWaveFunctionsPerAtomPsP.resize(
              numNonLocalAtomsCurrentProcessPsP);
            for (dftfe::uInt iAtom = 0;
                 iAtom < numNonLocalAtomsCurrentProcessPsP;
                 iAtom++)
              {
                nonLocalAtomIdPsP[iAtom] =
                  dftPtr->d_oncvClassPtr->getAtomIdInCurrentProcessor(iAtom);
                globalChargeIdNonLocalAtomPsP[iAtom] =
                  dftPtr->d_atomIdPseudopotentialInterestToGlobalId
                    .find(nonLocalAtomIdPsP[iAtom])
                    ->second;
                numberPseudoWaveFunctionsPerAtomPsP[iAtom] =
                  dftPtr->d_oncvClassPtr
                    ->getTotalNumberOfSphericalFunctionsForAtomId(
                      nonLocalAtomIdPsP[iAtom]);
              }

            const std::shared_ptr<
              AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
              oncvNonLocalOp = dftPtr->d_oncvClassPtr->getNonLocalOperator();
            for (dftfe::uInt cell = 0; cell < matrixFreeData.n_cell_batches();
                 ++cell)
              {
                forceEvalNLP.reinit(cell);

                // compute FnlGammaAtoms  (contibution due to Gamma(Rj))
                FnlGammaAtomsElementalContribution(
                  forceContributionFnlGammaAtoms,
                  matrixFreeData,
                  forceEvalNLP,
                  oncvNonLocalOp,
                  numNonLocalAtomsCurrentProcessPsP,
                  globalChargeIdNonLocalAtomPsP,
                  numberPseudoWaveFunctionsPerAtomPsP,
                  cell,
                  cellIdToCellNumberMap,
#ifdef USE_COMPLEX
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
                  dftPtr->d_oncvClassPtr->getNonLocalOperator()
                    ->getAtomCenteredKpointIndexedSphericalFnQuadValues(),
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);


                if (!d_dftParams.floatingNuclearCharges)
                  {
                    FnlGammaxElementalContribution(
                      FVectQuads,
                      matrixFreeData,
                      numQuadPointsNLP,
                      oncvNonLocalOp,
                      numNonLocalAtomsCurrentProcessPsP,
                      globalChargeIdNonLocalAtomPsP,
                      numberPseudoWaveFunctionsPerAtomPsP,
                      cell,
                      cellIdToCellNumberMap,
                      dftPtr->d_oncvClassPtr->getNonLocalOperator()
                        ->getAtomCenteredKpointIndexedSphericalFnQuadValues(),
                      projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);

                    for (dftfe::uInt q = 0; q < numQuadPointsNLP; ++q)
                      forceEvalNLP.submit_value(spinPolarizedFactorVect *
                                                  FVectQuads[q],
                                                q);

                    forceEvalNLP.integrate(dealii::EvaluationFlags::values);

#ifdef USE_COMPLEX
                    forceEvalNLP.distribute_local_to_global(
                      d_configForceVectorLinFEKPoints);
#else
                    forceEvalNLP.distribute_local_to_global(
                      d_configForceVectorLinFE);
#endif
                  } // no floating charges check
              }     // macro cell loop
          }         // pseudopotential check

        if (dftPtr->isHubbardCorrectionsUsed())
          {
            dealii::AlignedVector<
              dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
              FVectHubbardQuads(numQuadPointsNLP, zeroTensor3);

            const dftfe::uInt numNonLocalAtomsCurrentProcessHubbard =
              (dftPtr->getHubbardClassPtr()
                 ->getNonLocalOperator()
                 ->getTotalAtomInCurrentProcessor());

            std::vector<dftfe::Int> nonLocalAtomIdHubbard,
              globalChargeIdNonLocalAtomHubbard;
            nonLocalAtomIdHubbard.resize(numNonLocalAtomsCurrentProcessHubbard);
            globalChargeIdNonLocalAtomHubbard.resize(
              numNonLocalAtomsCurrentProcessHubbard);

            std::vector<dftfe::uInt> numberPseudoWaveFunctionsPerAtomHubbard;
            numberPseudoWaveFunctionsPerAtomHubbard.resize(
              numNonLocalAtomsCurrentProcessHubbard);
            for (dftfe::uInt iAtom = 0;
                 iAtom < numNonLocalAtomsCurrentProcessHubbard;
                 iAtom++)
              {
                globalChargeIdNonLocalAtomHubbard[iAtom] =
                  dftPtr->getHubbardClassPtr()->getGlobalAtomId(iAtom);
                numberPseudoWaveFunctionsPerAtomHubbard[iAtom] =
                  dftPtr->getHubbardClassPtr()
                    ->getTotalNumberOfSphericalFunctionsForAtomId(iAtom);
              }

            const std::shared_ptr<
              AtomicCenteredNonLocalOperator<dataTypes::number, memorySpace>>
              hubbardNonLocalOp =
                dftPtr->getHubbardClassPtr()->getNonLocalOperator();

            for (dftfe::uInt cell = 0; cell < matrixFreeData.n_cell_batches();
                 ++cell)
              {
                forceEvalNLP.reinit(cell);

                // compute FnlGammaAtoms  (contibution due to Gamma(Rj))
                FnlGammaAtomsElementalContribution(
                  forceContributionFnlGammaAtomsHubbard,
                  matrixFreeData,
                  forceEvalNLP,
                  hubbardNonLocalOp,
                  numNonLocalAtomsCurrentProcessHubbard,
                  globalChargeIdNonLocalAtomHubbard,
                  numberPseudoWaveFunctionsPerAtomHubbard,
                  cell,
                  cellIdToCellNumberMap,
#ifdef USE_COMPLEX
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHubbard,
#endif
                  dftPtr->getHubbardClassPtr()
                    ->getNonLocalOperator()
                    ->getAtomCenteredKpointIndexedSphericalFnQuadValues(),
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard);


                if (!d_dftParams.floatingNuclearCharges)
                  {
                    FnlGammaxElementalContribution(
                      FVectHubbardQuads,
                      matrixFreeData,
                      numQuadPointsNLP,
                      hubbardNonLocalOp,
                      numNonLocalAtomsCurrentProcessHubbard,
                      globalChargeIdNonLocalAtomHubbard,
                      numberPseudoWaveFunctionsPerAtomHubbard,
                      cell,
                      cellIdToCellNumberMap,
                      dftPtr->getHubbardClassPtr()
                        ->getNonLocalOperator()
                        ->getAtomCenteredKpointIndexedSphericalFnQuadValues(),
                      projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHubbard);

                    for (dftfe::uInt q = 0; q < numQuadPointsNLP; ++q)
                      forceEvalNLP.submit_value(spinPolarizedFactorVect *
                                                  FVectHubbardQuads[q],
                                                q);

                    forceEvalNLP.integrate(dealii::EvaluationFlags::values);

#ifdef USE_COMPLEX
                    // TODO check if this can be re used
                    forceEvalNLP.distribute_local_to_global(
                      d_configForceVectorLinFEKPoints);
#else
                    // TODO check if this can be re used
                    forceEvalNLP.distribute_local_to_global(
                      d_configForceVectorLinFE);
#endif
                  } // no floating charges check
              }     // macro cell loop
          }         // isHubbardCorrectionsUsed() check
      }             // spin index

    // add global Fnl contribution due to Gamma(Rj) to the configurational force
    // vector
    if (isPseudopotential)
      {
        if (d_dftParams.spinPolarized == 1)
          for (auto &iter : forceContributionFnlGammaAtoms)
            {
              std::vector<double> &fnlvec = iter.second;
              for (dftfe::uInt i = 0; i < fnlvec.size(); i++)
                fnlvec[i] *= spinPolarizedFactor;
            }

        if (d_dftParams.floatingNuclearCharges)
          {
#ifdef USE_COMPLEX
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFnlGammaAtoms, d_forceAtomsFloatingKPoints);
#else
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFnlGammaAtoms, d_forceAtomsFloating);
#endif
          }
        else
          distributeForceContributionFnlGammaAtoms(
            forceContributionFnlGammaAtoms);
      }

    if (dftPtr->isHubbardCorrectionsUsed())
      {
        if (d_dftParams.spinPolarized == 1)
          for (auto &iter : forceContributionFnlGammaAtomsHubbard)
            {
              std::vector<double> &fnlvec = iter.second;
              for (dftfe::uInt i = 0; i < fnlvec.size(); i++)
                fnlvec[i] *= spinPolarizedFactor;
            }

        if (d_dftParams.floatingNuclearCharges)
          {
#ifdef USE_COMPLEX
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFnlGammaAtomsHubbard,
              d_forceAtomsFloatingKPoints);
#else
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFnlGammaAtomsHubbard, d_forceAtomsFloating);
#endif
          }
        else
          distributeForceContributionFnlGammaAtoms(
            forceContributionFnlGammaAtomsHubbard);
      }


    MPI_Barrier(d_mpiCommParent);
    double enowfc_time = MPI_Wtime();

    /////////// Compute contribution independent of wavefunctions
    ////////////////////
    if (bandGroupTaskId == 0)
      {
        // kpoint group parallelization data structures
        const dftfe::uInt numberKptGroups =
          dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

        const dftfe::uInt kptGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
        std::vector<dftfe::Int> kptGroupLowHighPlusOneIndices;

        if (numMacroCells > 0)
          dftUtils::createKpointParallelizationIndices(
            dftPtr->interpoolcomm,
            numMacroCells,
            kptGroupLowHighPlusOneIndices);

        dftPtr->d_basisOperationsPtrHost->reinit(0,
                                                 0,
                                                 dftPtr->d_densityQuadratureId,
                                                 false);
        std::vector<
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          gradDensityOutValuesSpinPolarized;

        bool isGradDensityDataRequired =
          (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
             ->getDensityBasedFamilyType() == densityFamilyType::GGA);
        if (isGradDensityDataRequired)
          {
            gradDensityOutValuesSpinPolarized = gradRhoOutValues;

            if (d_dftParams.spinPolarized == 0)
              gradDensityOutValuesSpinPolarized.push_back(
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::HOST>(
                  gradRhoOutValues[0].size(), 0.0));
          }

        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoSpin0QuadsVect(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoSpin1QuadsVect(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoSpin0Quads(numQuadPoints, zeroTensor4);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoSpin1Quads(numQuadPoints, zeroTensor4);
        dealii::AlignedVector<dealii::VectorizedArray<double>> excQuads(
          numQuadPoints, dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<dealii::VectorizedArray<double>>
          vxcRhoOutSpin0Quads(numQuadPoints,
                              dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<dealii::VectorizedArray<double>>
          vxcRhoOutSpin1Quads(numQuadPoints,
                              dealii::make_vectorized_array(0.0));
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoCoreQuads(numQuadPoints, zeroTensor3);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoCoreQuads(numQuadPoints, zeroTensor4);
        std::map<dftfe::uInt, std::vector<double>>
          forceContributionNonlinearCoreCorrectionGammaAtoms;

        std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
          xDensityOutDataOut;
        std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
          cDensityOutDataOut;

        std::vector<double> &xEnergyDensityOut =
          xDensityOutDataOut[xcRemainderOutputDataAttributes::e];
        std::vector<double> &cEnergyDensityOut =
          cDensityOutDataOut[xcRemainderOutputDataAttributes::e];

        std::vector<double> &pdexDensityOutSpinUp =
          xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
        std::vector<double> &pdexDensityOutSpinDown = xDensityOutDataOut
          [xcRemainderOutputDataAttributes::pdeDensitySpinDown];
        std::vector<double> &pdecDensityOutSpinUp =
          cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
        std::vector<double> &pdecDensityOutSpinDown = cDensityOutDataOut
          [xcRemainderOutputDataAttributes::pdeDensitySpinDown];

        if (isGradDensityDataRequired)
          {
            xDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
              std::vector<double>();
            cDensityOutDataOut[xcRemainderOutputDataAttributes::pdeSigma] =
              std::vector<double>();
          }


        auto quadPointsAll = dftPtr->d_basisOperationsPtrHost->quadPoints();

        auto quadWeightsAll = dftPtr->d_basisOperationsPtrHost->JxW();

        for (dftfe::uInt cell = 0; cell < matrixFreeData.n_cell_batches();
             ++cell)
          {
            if (cell < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
              {
                forceEval.reinit(cell);

                std::fill(gradRhoSpin0QuadsVect.begin(),
                          gradRhoSpin0QuadsVect.end(),
                          zeroTensor3);
                std::fill(gradRhoSpin1QuadsVect.begin(),
                          gradRhoSpin1QuadsVect.end(),
                          zeroTensor3);
                std::fill(hessianRhoSpin0Quads.begin(),
                          hessianRhoSpin0Quads.end(),
                          zeroTensor4);
                std::fill(hessianRhoSpin1Quads.begin(),
                          hessianRhoSpin1Quads.end(),
                          zeroTensor4);
                std::fill(excQuads.begin(),
                          excQuads.end(),
                          dealii::make_vectorized_array(0.0));
                std::fill(vxcRhoOutSpin0Quads.begin(),
                          vxcRhoOutSpin0Quads.end(),
                          dealii::make_vectorized_array(0.0));
                std::fill(vxcRhoOutSpin1Quads.begin(),
                          vxcRhoOutSpin1Quads.end(),
                          dealii::make_vectorized_array(0.0));
                std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),
                          derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),
                          zeroTensor3);
                std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),
                          derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),
                          zeroTensor3);
                std::fill(gradRhoCoreQuads.begin(),
                          gradRhoCoreQuads.end(),
                          zeroTensor3);
                std::fill(hessianRhoCoreQuads.begin(),
                          hessianRhoCoreQuads.end(),
                          zeroTensor4);

                const dftfe::uInt numSubCells =
                  matrixFreeData.n_active_entries_per_cell_batch(cell);

                //
                for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
                     ++iSubCell)
                  {
                    subCellPtr =
                      matrixFreeData.get_cell_iterator(cell, iSubCell);
                    dealii::CellId subCellId = subCellPtr->id();


                    const dftfe::uInt subCellIndex =
                      dftPtr->d_basisOperationsPtrHost->cellIndex(subCellId);


                    dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
                      ->computeRhoTauDependentXCData(
                        *(dftPtr->d_auxDensityMatrixXCOutPtr),
                        std::make_pair<dftfe::uInt, dftfe::uInt>(
                          subCellIndex * numQuadPoints,
                          (subCellIndex + 1) * numQuadPoints),
                        xDensityOutDataOut,
                        cDensityOutDataOut);

                    std::vector<double> pdexDensityOutSigma;
                    std::vector<double> pdecDensityOutSigma;

                    if (isGradDensityDataRequired)
                      {
                        pdexDensityOutSigma = xDensityOutDataOut
                          [xcRemainderOutputDataAttributes::pdeSigma];
                        pdecDensityOutSigma = cDensityOutDataOut
                          [xcRemainderOutputDataAttributes::pdeSigma];
                      }


                    std::unordered_map<DensityDescriptorDataAttributes,
                                       std::vector<double>>
                                        densityXCOutData;
                    std::vector<double> gradDensityXCOutSpinUp;
                    std::vector<double> gradDensityXCOutSpinDown;

                    if (isGradDensityDataRequired)
                      {
                        densityXCOutData
                          [DensityDescriptorDataAttributes::gradValuesSpinUp] =
                            std::vector<double>();
                        densityXCOutData[DensityDescriptorDataAttributes::
                                           gradValuesSpinDown] =
                          std::vector<double>();
                      }

                    dftPtr->d_auxDensityMatrixXCOutPtr->applyLocalOperations(
                      std::make_pair<dftfe::uInt, dftfe::uInt>(
                        subCellIndex * numQuadPoints,
                        (subCellIndex + 1) * numQuadPoints),
                      densityXCOutData);

                    if (isGradDensityDataRequired)
                      {
                        gradDensityXCOutSpinUp = densityXCOutData
                          [DensityDescriptorDataAttributes::gradValuesSpinUp];
                        gradDensityXCOutSpinDown = densityXCOutData
                          [DensityDescriptorDataAttributes::gradValuesSpinDown];
                      }



                    if (isGradDensityDataRequired)
                      {
                        // const std::vector<double> &temp3 =
                        //  (*dftPtr->gradRhoOutValuesSpinPolarized)
                        //    .find(subCellId)
                        //    ->second;

                        const auto &gradRhoTotalOutValues =
                          gradDensityOutValuesSpinPolarized[0];
                        const auto &gradRhoMagOutValues =
                          gradDensityOutValuesSpinPolarized[1];

                        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                          for (dftfe::uInt idim = 0; idim < 3; idim++)
                            {
                              gradRhoSpin0QuadsVect[q][idim][iSubCell] =
                                (gradRhoTotalOutValues[subCellIndex *
                                                         numQuadPoints * 3 +
                                                       q * 3 + idim] +
                                 gradRhoMagOutValues[subCellIndex *
                                                       numQuadPoints * 3 +
                                                     q * 3 + idim]) /
                                2.0;
                              gradRhoSpin1QuadsVect[q][idim][iSubCell] =
                                (gradRhoTotalOutValues[subCellIndex *
                                                         numQuadPoints * 3 +
                                                       q * 3 + idim] -
                                 gradRhoMagOutValues[subCellIndex *
                                                       numQuadPoints * 3 +
                                                     q * 3 + idim]) /
                                2.0;
                            }
                      }


                    for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                      {
                        excQuads[q][iSubCell] =
                          xEnergyDensityOut[q] + cEnergyDensityOut[q];
                        vxcRhoOutSpin0Quads[q][iSubCell] =
                          pdexDensityOutSpinUp[q] + pdecDensityOutSpinUp[q];
                        vxcRhoOutSpin1Quads[q][iSubCell] =
                          pdexDensityOutSpinDown[q] + pdecDensityOutSpinDown[q];
                      }

                    if (isGradDensityDataRequired)
                      {
                        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                          {
                            for (dftfe::uInt idim = 0; idim < 3; idim++)
                              {
                                derExchCorrEnergyWithGradRhoOutSpin0Quads
                                  [q][idim][iSubCell] =
                                    2.0 *
                                    (pdexDensityOutSigma[3 * q + 0] +
                                     pdecDensityOutSigma[3 * q + 0]) *
                                    gradDensityXCOutSpinUp[3 * q + idim];
                                derExchCorrEnergyWithGradRhoOutSpin0Quads
                                  [q][idim][iSubCell] +=
                                  (pdexDensityOutSigma[3 * q + 1] +
                                   pdecDensityOutSigma[3 * q + 1]) *
                                  gradDensityXCOutSpinDown[3 * q + idim];

                                derExchCorrEnergyWithGradRhoOutSpin1Quads
                                  [q][idim][iSubCell] +=
                                  2.0 *
                                  (pdexDensityOutSigma[3 * q + 2] +
                                   pdecDensityOutSigma[3 * q + 2]) *
                                  gradDensityXCOutSpinDown[3 * q + idim];
                                derExchCorrEnergyWithGradRhoOutSpin1Quads
                                  [q][idim][iSubCell] +=
                                  (pdexDensityOutSigma[3 * q + 1] +
                                   pdecDensityOutSigma[3 * q + 1]) *
                                  gradDensityXCOutSpinUp[3 * q + idim];
                              }
                          }
                      }


                    if (d_dftParams.nonLinearCoreCorrection == true)
                      for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                        {
                          const std::vector<double> &temp1 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                            for (dftfe::uInt idim = 0; idim < 3; idim++)
                              gradRhoCoreQuads[q][idim][iSubCell] =
                                temp1[3 * q + idim] / 2.0;

                          if (isGradDensityDataRequired)
                            {
                              const std::vector<double> &temp2 =
                                hessianRhoCoreValues.find(subCellId)->second;
                              for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                                for (dftfe::uInt idim = 0; idim < 3; ++idim)
                                  for (dftfe::uInt jdim = 0; jdim < 3; ++jdim)
                                    hessianRhoCoreQuads
                                      [q][idim][jdim][iSubCell] =
                                        temp2[9 * q + 3 * idim + jdim] / 2.0;
                            }
                        }

                  } // subcell loop

                if (d_dftParams.nonLinearCoreCorrection)
                  {
                    FNonlinearCoreCorrectionGammaAtomsElementalContributionSpinPolarized(
                      forceContributionNonlinearCoreCorrectionGammaAtoms,
                      forceEval,
                      matrixFreeData,
                      cell,
                      vxcRhoOutSpin0Quads,
                      vxcRhoOutSpin1Quads,
                      derExchCorrEnergyWithGradRhoOutSpin0Quads,
                      derExchCorrEnergyWithGradRhoOutSpin1Quads,
                      gradRhoCoreAtoms,
                      hessianRhoCoreAtoms,
                      isGradDensityDataRequired);
                  }

                for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                  {
                    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> E =
                      eshelbyTensorSP::getELocXcEshelbyTensor(
                        gradRhoSpin0QuadsVect[q],
                        gradRhoSpin1QuadsVect[q],
                        excQuads[q],
                        derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                        derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

                    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                      zeroTensor3;

                    if (d_dftParams.nonLinearCoreCorrection)
                      F += eshelbyTensorSP::getFNonlinearCoreCorrection(
                        vxcRhoOutSpin0Quads[q],
                        vxcRhoOutSpin1Quads[q],
                        derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                        derExchCorrEnergyWithGradRhoOutSpin1Quads[q],
                        gradRhoCoreQuads[q],
                        hessianRhoCoreQuads[q],
                        isGradDensityDataRequired);

                    forceEval.submit_value(F, q);
                    forceEval.submit_gradient(E, q);
                  } // quad point loop

                forceEval.integrate(dealii::EvaluationFlags::values |
                                    dealii::EvaluationFlags::gradients);
                forceEval.distribute_local_to_global(
                  d_configForceVectorLinFE); // also takes care of
                                             // constraints
              }                              // kpt paral
          }                                  // cell loop

        if (d_dftParams.nonLinearCoreCorrection)
          {
            if (d_dftParams.floatingNuclearCharges)
              accumulateForceContributionGammaAtomsFloating(
                forceContributionNonlinearCoreCorrectionGammaAtoms,
                d_forceAtomsFloating);
            else
              distributeForceContributionFPSPLocalGammaAtoms(
                forceContributionNonlinearCoreCorrectionGammaAtoms,
                d_atomsForceDofs,
                d_constraintsNoneForce,
                d_configForceVectorLinFE);
          }


        ////Add electrostatic configurational force contribution////////////////
        computeConfigurationalForceEEshelbyEElectroPhiTot(
          matrixFreeDataElectro,
          phiTotDofHandlerIndexElectro,
          smearedChargeQuadratureId,
          lpspQuadratureIdElectro,
          phiTotRhoOutElectro,
          rhoOutValues[0],
          rhoTotalOutValuesLpsp,
          gradRhoOutValues[0],
          gradRhoTotalOutValuesLpsp,
          pseudoVLocElectro,
          pseudoVLocAtomsElectro,
          vselfBinsManagerElectro);
      }

    MPI_Barrier(d_mpiCommParent);
    enowfc_time = MPI_Wtime() - enowfc_time;

    forcetotal_time = MPI_Wtime() - forcetotal_time;

    if (this_process == 0 && d_dftParams.verbosity >= 4)
      std::cout
        << "Total time for configurational force computation except Eself contribution: "
        << forcetotal_time << std::endl;

    if (d_dftParams.verbosity >= 4)
      {
        pcout << " Time taken for initialization in force: " << init_time
              << std::endl;
        pcout << " Time taken for non wfc in force: " << enowfc_time
              << std::endl;
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::computeConfigurationalForceEEshelbyEElectroPhiTot(
    const dealii::MatrixFree<3, double> &matrixFreeDataElectro,
    const dftfe::uInt                    phiTotDofHandlerIndexElectro,
    const dftfe::uInt                    smearedChargeQuadratureId,
    const dftfe::uInt                    lpspQuadratureIdElectro,
    const distributedCPUVec<double>     &phiTotRhoOutElectro,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoTotalOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoTotalOutValuesLpsp,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &gradRhoTotalOutValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &gradRhoTotalOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                           &pseudoVLocAtomsElectro,
    const vselfBinsManager &vselfBinsManagerElectro)
  {
    FEEvaluationWrapperClass<3> forceEvalElectro(
      1,
      d_dftParams.densityQuadratureRule,
      matrixFreeDataElectro,
      d_forceDofHandlerIndexElectro,
      dftPtr->d_densityQuadratureIdElectro);

    FEEvaluationWrapperClass<1> phiTotEvalElectro(
      d_dftParams.finiteElementPolynomialOrderElectrostatics,
      d_dftParams.densityQuadratureRule,
      matrixFreeDataElectro,
      phiTotDofHandlerIndexElectro,
      dftPtr->d_densityQuadratureIdElectro);

    FEEvaluationWrapperClass<1> phiTotEvalSmearedCharge(
      -1,
      1,
      matrixFreeDataElectro,
      phiTotDofHandlerIndexElectro,
      smearedChargeQuadratureId);

    FEEvaluationWrapperClass<3> forceEvalSmearedCharge(
      -1,
      1,
      matrixFreeDataElectro,
      d_forceDofHandlerIndexElectro,
      smearedChargeQuadratureId);

    FEEvaluationWrapperClass<3> forceEvalElectroLpsp(
      1,
      C_num1DQuadLPSP(d_dftParams.finiteElementPolynomialOrderElectrostatics) *
        C_numCopies1DQuadLPSP(),
      matrixFreeDataElectro,
      d_forceDofHandlerIndexElectro,
      lpspQuadratureIdElectro);

    std::map<dftfe::uInt, std::vector<double>>
      forceContributionFPSPLocalGammaAtoms;
    std::map<dftfe::uInt, std::vector<double>>
      forceContributionSmearedChargesGammaAtoms;
    std::map<dftfe::uInt, std::vector<double>>
      forceContributionShadowPotentialElectroGammaAtoms;

    const dftfe::uInt numQuadPoints         = forceEvalElectro.n_q_points;
    const dftfe::uInt numQuadPointsSmearedb = forceEvalSmearedCharge.n_q_points;
    const dftfe::uInt numQuadPointsLpsp     = forceEvalElectroLpsp.n_q_points;

    AssertThrow(
      matrixFreeDataElectro.get_quadrature(smearedChargeQuadratureId).size() ==
        numQuadPointsSmearedb,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

    AssertThrow(
      matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro).size() ==
        numQuadPointsLpsp,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

    // if (gradRhoTotalOutValuesLpsp.size() != 0)
    //  AssertThrow(
    //    gradRhoTotalOutValuesLpsp.begin()->second.size() == 3 *
    //    numQuadPointsLpsp, dealii::ExcMessage(
    //      "DFT-FE Error: mismatch in quadrature rule usage in force
    //      computation."));

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::FEValues<3> feVselfValuesElectro(
      matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro)
        .get_fe(),
      matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro),
      d_dftParams.floatingNuclearCharges && d_dftParams.smearedNuclearCharges ?
        (dealii::update_values | dealii::update_quadrature_points) :
        (dealii::update_values | dealii::update_gradients |
         dealii::update_quadrature_points));

    dealii::QIterated<3 - 1> faceQuadrature(
      dealii::QGauss<1>(C_num1DQuadLPSP(
        d_dftParams.finiteElementPolynomialOrderElectrostatics)),
      C_numCopies1DQuadLPSP());
    dealii::FEFaceValues<3> feFaceValuesElectro(
      dftPtr->d_dofHandlerRhoNodal.get_fe(),
      faceQuadrature,
      dealii::update_values | dealii::update_JxW_values |
        dealii::update_normal_vectors | dealii::update_quadrature_points);

    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      {
        zeroTensor[idim] = dealii::make_vectorized_array(0.0);
      }

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
        {
          zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
        }


    std::vector<double> tempRhoVal(numQuadPoints, 0);
    std::vector<double> tempLpspRhoVal(numQuadPointsLpsp, 0);
    std::vector<double> tempLpspGradRhoVal(numQuadPointsLpsp * 3, 0);


    dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuadsElectro(
      numQuadPoints, dealii::make_vectorized_array(0.0));
    dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuadsElectroLpsp(
      numQuadPointsLpsp, dealii::make_vectorized_array(0.0));
    dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
      numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      gradPhiTotSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      gradRhoQuadsElectro(numQuadPoints, zeroTensor);
    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      gradRhoQuadsElectroLpsp(numQuadPointsLpsp, zeroTensor);
    // dealii::AlignedVector<dealii::Tensor<1,3,dealii::VectorizedArray<double>
    // > > gradRhoAtomsQuadsElectro(numQuadPoints,zeroTensor);
    dealii::AlignedVector<dealii::VectorizedArray<double>>
      pseudoVLocQuadsElectro(numQuadPointsLpsp,
                             dealii::make_vectorized_array(0.0));

    // kpoint group parallelization data structures
    const dftfe::uInt numberKptGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

    const dftfe::uInt kptGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
    std::vector<dftfe::Int> kptGroupLowHighPlusOneIndices;

    if (matrixFreeDataElectro.n_cell_batches() > 0)
      dftUtils::createKpointParallelizationIndices(
        dftPtr->interpoolcomm,
        matrixFreeDataElectro.n_cell_batches(),
        kptGroupLowHighPlusOneIndices);

    for (dftfe::uInt cell = 0; cell < matrixFreeDataElectro.n_cell_batches();
         ++cell)
      {
        if (cell < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
            cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
          {
            std::set<dftfe::uInt> nonTrivialSmearedChargeAtomIdsMacroCell;

            const dftfe::uInt numSubCells =
              matrixFreeDataElectro.n_active_entries_per_cell_batch(cell);
            if (d_dftParams.smearedNuclearCharges)
              for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
                {
                  subCellPtr =
                    matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                  dealii::CellId                  subCellId = subCellPtr->id();
                  const std::vector<dftfe::uInt> &temp =
                    dftPtr->d_bCellNonTrivialAtomIds.find(subCellId)->second;
                  for (dftfe::Int i = 0; i < temp.size(); i++)
                    nonTrivialSmearedChargeAtomIdsMacroCell.insert(temp[i]);
                }

            forceEvalElectro.reinit(cell);
            forceEvalElectroLpsp.reinit(cell);

            phiTotEvalElectro.reinit(cell);
            phiTotEvalElectro.read_dof_values_plain(phiTotRhoOutElectro);
            phiTotEvalElectro.evaluate(dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);

            if (d_dftParams.smearedNuclearCharges &&
                nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
              {
                forceEvalSmearedCharge.reinit(cell);
                phiTotEvalSmearedCharge.reinit(cell);
              }

            std::fill(rhoQuadsElectro.begin(),
                      rhoQuadsElectro.end(),
                      dealii::make_vectorized_array(0.0));
            std::fill(rhoQuadsElectroLpsp.begin(),
                      rhoQuadsElectroLpsp.end(),
                      dealii::make_vectorized_array(0.0));
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            std::fill(gradPhiTotSmearedChargeQuads.begin(),
                      gradPhiTotSmearedChargeQuads.end(),
                      zeroTensor);
            std::fill(gradRhoQuadsElectro.begin(),
                      gradRhoQuadsElectro.end(),
                      zeroTensor);
            std::fill(gradRhoQuadsElectroLpsp.begin(),
                      gradRhoQuadsElectroLpsp.end(),
                      zeroTensor);
            // std::fill(gradRhoAtomsQuadsElectro.begin(),gradRhoAtomsQuadsElectro.end(),zeroTensor);
            std::fill(pseudoVLocQuadsElectro.begin(),
                      pseudoVLocQuadsElectro.end(),
                      dealii::make_vectorized_array(0.0));

            for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr =
                  matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                dealii::CellId subCellId = subCellPtr->id();

                const dftfe::uInt subCellIndex =
                  dftPtr->d_basisOperationsPtrElectroHost->cellIndex(subCellId);

                for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                  tempRhoVal[q] =
                    rhoTotalOutValues[subCellIndex * numQuadPoints + q];

                for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                  rhoQuadsElectro[q][iSubCell] = tempRhoVal[q];
                // rhoOutValuesElectro.find(subCellId)->second[q];

                if (d_dftParams.isPseudopotential ||
                    d_dftParams.smearedNuclearCharges)
                  {
                    const std::vector<double> &tempPseudoVal =
                      pseudoVLocElectro.find(subCellId)->second;
                    // const std::vector<double> &tempLpspRhoVal =
                    //  rhoOutValuesLpsp.find(subCellId)->second;
                    // const std::vector<double> &tempLpspGradRhoVal =
                    //  gradRhoOutValuesLpsp.find(subCellId)->second;



                    for (dftfe::uInt q = 0; q < numQuadPointsLpsp; ++q)
                      {
                        tempLpspRhoVal[q] =
                          rhoTotalOutValuesLpsp[subCellIndex *
                                                  numQuadPointsLpsp +
                                                q];
                        tempLpspGradRhoVal[3 * q + 0] =
                          gradRhoTotalOutValuesLpsp[subCellIndex *
                                                      numQuadPointsLpsp * 3 +
                                                    3 * q + 0];
                        tempLpspGradRhoVal[3 * q + 1] =
                          gradRhoTotalOutValuesLpsp[subCellIndex *
                                                      numQuadPointsLpsp * 3 +
                                                    3 * q + 1];
                        tempLpspGradRhoVal[3 * q + 2] =
                          gradRhoTotalOutValuesLpsp[subCellIndex *
                                                      numQuadPointsLpsp * 3 +
                                                    3 * q + 2];
                      }

                    for (dftfe::uInt q = 0; q < numQuadPointsLpsp; ++q)
                      {
                        pseudoVLocQuadsElectro[q][iSubCell] = tempPseudoVal[q];
                        rhoQuadsElectroLpsp[q][iSubCell]    = tempLpspRhoVal[q];
                        gradRhoQuadsElectroLpsp[q][0][iSubCell] =
                          tempLpspGradRhoVal[3 * q + 0];
                        gradRhoQuadsElectroLpsp[q][1][iSubCell] =
                          tempLpspGradRhoVal[3 * q + 1];
                        gradRhoQuadsElectroLpsp[q][2][iSubCell] =
                          tempLpspGradRhoVal[3 * q + 2];
                      }
                  }

                if (d_dftParams.smearedNuclearCharges &&
                    nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
                  {
                    const std::vector<double> &bQuadValuesCell =
                      dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;
                    for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                      {
                        smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                      }
                  }
              }

            if (d_dftParams.isPseudopotential ||
                d_dftParams.smearedNuclearCharges)
              {
                FPSPLocalGammaAtomsElementalContribution(
                  forceContributionFPSPLocalGammaAtoms,
                  feVselfValuesElectro,
                  feFaceValuesElectro,
                  forceEvalElectroLpsp,
                  matrixFreeDataElectro,
                  phiTotDofHandlerIndexElectro,
                  cell,
                  rhoQuadsElectroLpsp,
                  gradRhoQuadsElectroLpsp,
                  pseudoVLocAtomsElectro,
                  vselfBinsManagerElectro,
                  d_cellsVselfBallsClosestAtomIdDofHandlerElectro);
              }

            for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
              {
                dealii::VectorizedArray<double> phiTotElectro_q =
                  phiTotEvalElectro.get_value(q);
                dealii::VectorizedArray<double> phiExtElectro_q =
                  dealii::make_vectorized_array(0.0);
                dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                  gradPhiTotElectro_q = phiTotEvalElectro.get_gradient(q);

                dealii::Tensor<2, 3, dealii::VectorizedArray<double>> E =
                  eshelbyTensor::getEElectroEshelbyTensor(phiTotElectro_q,
                                                          gradPhiTotElectro_q,
                                                          rhoQuadsElectro[q]);

                dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                  zeroTensor;


                forceEvalElectro.submit_value(F, q);
                forceEvalElectro.submit_gradient(E, q);
              }

            if (d_dftParams.isPseudopotential ||
                d_dftParams.smearedNuclearCharges)
              for (dftfe::uInt q = 0; q < numQuadPointsLpsp; ++q)
                {
                  dealii::VectorizedArray<double> phiExtElectro_q =
                    dealii::make_vectorized_array(0.0);

                  dealii::Tensor<2, 3, dealii::VectorizedArray<double>> E =
                    zeroTensor2;

                  dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                    zeroTensor;
                  dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                    gradPhiExt_q = zeroTensor;
                  F -= gradRhoQuadsElectroLpsp[q] * pseudoVLocQuadsElectro[q];

                  forceEvalElectroLpsp.submit_value(F, q);
                  forceEvalElectroLpsp.submit_gradient(E, q);
                }

            forceEvalElectro.integrate(dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);
            forceEvalElectro.distribute_local_to_global(
              d_configForceVectorLinFEElectro);

            if (d_dftParams.isPseudopotential ||
                d_dftParams.smearedNuclearCharges)
              {
                forceEvalElectroLpsp.integrate(
                  dealii::EvaluationFlags::values |
                  dealii::EvaluationFlags::gradients);
                forceEvalElectroLpsp.distribute_local_to_global(
                  d_configForceVectorLinFEElectro);
              }

            if (d_dftParams.smearedNuclearCharges &&
                nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
              {
                phiTotEvalSmearedCharge.read_dof_values_plain(
                  phiTotRhoOutElectro);
                phiTotEvalSmearedCharge.evaluate(
                  dealii::EvaluationFlags::gradients);

                for (dftfe::uInt q = 0; q < numQuadPointsSmearedb; ++q)
                  {
                    gradPhiTotSmearedChargeQuads[q] =
                      phiTotEvalSmearedCharge.get_gradient(q);

                    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> F =
                      zeroTensor;
                    F = -gradPhiTotSmearedChargeQuads[q] * smearedbQuads[q];

                    forceEvalSmearedCharge.submit_value(F, q);
                  }


                forceEvalSmearedCharge.integrate(
                  dealii::EvaluationFlags::values);
                forceEvalSmearedCharge.distribute_local_to_global(
                  d_configForceVectorLinFEElectro);

                FPhiTotSmearedChargesGammaAtomsElementalContribution(
                  forceContributionSmearedChargesGammaAtoms,
                  forceEvalSmearedCharge,
                  matrixFreeDataElectro,
                  cell,
                  gradPhiTotSmearedChargeQuads,
                  std::vector<dftfe::uInt>(
                    nonTrivialSmearedChargeAtomIdsMacroCell.begin(),
                    nonTrivialSmearedChargeAtomIdsMacroCell.end()),
                  dftPtr->d_bQuadAtomIdsAllAtoms,
                  smearedbQuads);
              }
          } // kpt paral
      }     // macro cell loop

    // add global FPSPLocal contribution due to Gamma(Rj) to the configurational
    // force vector
    if (d_dftParams.isPseudopotential || d_dftParams.smearedNuclearCharges)
      {
        if (d_dftParams.floatingNuclearCharges)
          {
            accumulateForceContributionGammaAtomsFloating(
              forceContributionFPSPLocalGammaAtoms, d_forceAtomsFloating);
          }
        else
          distributeForceContributionFPSPLocalGammaAtoms(
            forceContributionFPSPLocalGammaAtoms,
            d_atomsForceDofsElectro,
            d_constraintsNoneForceElectro,
            d_configForceVectorLinFEElectro);
      }

    if (d_dftParams.smearedNuclearCharges)
      {
        if (d_dftParams.floatingNuclearCharges)
          {
            accumulateForceContributionGammaAtomsFloating(
              forceContributionSmearedChargesGammaAtoms, d_forceAtomsFloating);
          }
        else
          distributeForceContributionFPSPLocalGammaAtoms(
            forceContributionSmearedChargesGammaAtoms,
            d_atomsForceDofsElectro,
            d_constraintsNoneForceElectro,
            d_configForceVectorLinFEElectro);
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
