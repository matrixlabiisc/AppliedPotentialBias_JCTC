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

#include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsDevice.h>
#include <linearAlgebraOperationsDeviceKernels.h>
#include <vectorUtilities.h>

static const dftfe::uInt order_lookup[][2] = {
  {500, 24}, // <= 500 ~> chebyshevOrder = 24
  {750, 30},
  {1000, 39},
  {1500, 50},
  {2000, 53},
  {3000, 57},
  {4000, 62},
  {5000, 69},
  {9000, 77},
  {14000, 104},
  {20000, 119},
  {30000, 162},
  {50000, 300},
  {80000, 450},
  {100000, 550},
  {200000, 700},
  {500000, 1000}};

namespace dftfe
{
  namespace
  {
    namespace internal
    {
      dftfe::uInt
      setChebyshevOrder(const dftfe::uInt d_upperBoundUnWantedSpectrum)
      {
        for (dftfe::Int i = 0;
             i < sizeof(order_lookup) / sizeof(order_lookup[0]);
             i++)
          {
            if (d_upperBoundUnWantedSpectrum <= order_lookup[i][0])
              return order_lookup[i][1];
          }
        return 1250;
      }
    } // namespace internal
  }   // namespace

  //
  // Constructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::
    chebyshevOrthogonalizedSubspaceIterationSolverDevice(
      const MPI_Comm      &mpi_comm_parent,
      const MPI_Comm      &mpi_comm_domain,
      double               lowerBoundWantedSpectrum,
      double               lowerBoundUnWantedSpectrum,
      double               upperBoundUnWantedSpectrum,
      const dftParameters &dftParams)
    : d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum)
    , d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum)
    , d_upperBoundUnWantedSpectrum(upperBoundUnWantedSpectrum)
    , d_mpiCommParent(mpi_comm_parent)
    , d_dftParams(dftParams)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParams.reproducible_output || dftParams.verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::summary,
                      dealii::TimerOutput::wall_times)
  {}


  //
  // reinitialize spectrum bounds
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::reinitSpectrumBounds(
    double lowerBoundWantedSpectrum,
    double lowerBoundUnWantedSpectrum,
    double upperBoundUnWantedSpectrum)
  {
    d_lowerBoundWantedSpectrum   = lowerBoundWantedSpectrum;
    d_lowerBoundUnWantedSpectrum = lowerBoundUnWantedSpectrum;
    d_upperBoundUnWantedSpectrum = upperBoundUnWantedSpectrum;
  }


  //
  // solve
  //
  double
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::solve(
    operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                            &BLASWrapperPtr,
    elpaScalaManager        &elpaScala,
    dataTypes::number       *eigenVectorsFlattenedDevice,
    const dftfe::uInt        flattenedSize,
    const dftfe::uInt        totalNumberWaveFunctions,
    std::vector<double>     &eigenValues,
    std::vector<double>     &residualNorms,
    utils::DeviceCCLWrapper &devicecclMpiCommDomain,
    const MPI_Comm          &interBandGroupComm,
    const bool               isFirstFilteringCall,
    const bool               computeResidual,
    const bool               useMixedPrecOverall,
    const bool               isFirstScf)
  {
    dealii::TimerOutput computingTimerStandard(
      operatorMatrix.getMPICommunicatorDomain(),
      pcout,
      d_dftParams.reproducible_output || d_dftParams.verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);


    //
    // allocate memory for full flattened array on device and fill it up
    //
    const dftfe::uInt localVectorSize =
      flattenedSize / totalNumberWaveFunctions;

    // band group parallelization data structures
    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);


    const dftfe::uInt vectorsBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedDeviceVec<dataTypes::number> *XBlock =
      &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 0);
    distributedDeviceVec<dataTypes::number> *HXBlock =
      &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 1);
    distributedDeviceVec<dataTypes::number> *XBlock2 =
      d_dftParams.overlapComputeCommunCheby ?
        &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 2) :
        NULL;
    distributedDeviceVec<dataTypes::number> *ResidualBlock =
      (d_dftParams.useReformulatedChFSI && !d_dftParams.useSinglePrecCheby) ?
        &operatorMatrix.getScratchFEMultivector(
          vectorsBlockSize, (d_dftParams.overlapComputeCommunCheby ? 4 : 2)) :
        NULL;
    distributedDeviceVec<dataTypes::number> *ResidualBlockNew =
      (d_dftParams.useReformulatedChFSI && !d_dftParams.useSinglePrecCheby) ?
        &operatorMatrix.getScratchFEMultivector(
          vectorsBlockSize, (d_dftParams.overlapComputeCommunCheby ? 5 : 3)) :
        NULL;
    distributedDeviceVec<dataTypes::number> *HXBlock2 =
      d_dftParams.overlapComputeCommunCheby ?
        &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 3) :
        NULL;
    distributedDeviceVec<dataTypes::numberFP32> *XBlockFP32 =
      d_dftParams.useSinglePrecCheby ?
        &operatorMatrix.getScratchFEMultivectorSinglePrec(vectorsBlockSize, 0) :
        NULL;
    distributedDeviceVec<dataTypes::numberFP32> *HXBlockFP32 =
      d_dftParams.useSinglePrecCheby ?
        &operatorMatrix.getScratchFEMultivectorSinglePrec(vectorsBlockSize, 1) :
        NULL;

    distributedDeviceVec<dataTypes::numberFP32> *XBlock2FP32 =
      d_dftParams.overlapComputeCommunCheby && d_dftParams.useSinglePrecCheby ?
        &operatorMatrix.getScratchFEMultivectorSinglePrec(vectorsBlockSize, 2) :
        NULL;
    distributedDeviceVec<dataTypes::numberFP32> *HXBlock2FP32 =
      d_dftParams.overlapComputeCommunCheby && d_dftParams.useSinglePrecCheby ?
        &operatorMatrix.getScratchFEMultivectorSinglePrec(vectorsBlockSize, 3) :
        NULL;

    distributedDeviceVec<dataTypes::number> *ResidualBlock2 =
      (d_dftParams.useReformulatedChFSI && !d_dftParams.useSinglePrecCheby &&
       d_dftParams.overlapComputeCommunCheby) ?
        &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 6) :
        NULL;
    distributedDeviceVec<dataTypes::number> *ResidualBlockNew2 =
      (d_dftParams.useReformulatedChFSI && !d_dftParams.useSinglePrecCheby &&
       d_dftParams.overlapComputeCommunCheby) ?
        &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 7) :
        NULL;

    operatorMatrix.reinitNumberWavefunctions(vectorsBlockSize);
    std::vector<double> eigenValuesBlock(vectorsBlockSize);

    if (isFirstFilteringCall)
      {
        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.enter_subsection("Lanczos upper bound");
          }

        std::pair<double, double> bounds = linearAlgebraOperations::
          generalisedLanczosLowerUpperBoundEigenSpectrum(
            BLASWrapperPtr,
            operatorMatrix,
            operatorMatrix.getScratchFEMultivector(1, 0),
            operatorMatrix.getScratchFEMultivector(1, 1),
            operatorMatrix.getScratchFEMultivector(1, 2),
            operatorMatrix.getScratchFEMultivector(1, 3),
            d_dftParams);

        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.leave_subsection("Lanczos upper bound");
          }

        d_lowerBoundWantedSpectrum   = bounds.first;
        d_upperBoundUnWantedSpectrum = bounds.second;
        d_lowerBoundUnWantedSpectrum =
          d_lowerBoundWantedSpectrum +
          (d_upperBoundUnWantedSpectrum - d_lowerBoundWantedSpectrum) *
            totalNumberWaveFunctions /
            operatorMatrix.getScratchFEMultivector(1, 0).globalSize() *
            (d_dftParams.reproducible_output ? 10.0 : 200.0);
      }
    else if (!d_dftParams.reuseLanczosUpperBoundFromFirstCall)
      {
        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.enter_subsection("Lanczos upper bound");
          }

        std::pair<double, double> bounds = linearAlgebraOperations::
          generalisedLanczosLowerUpperBoundEigenSpectrum(
            BLASWrapperPtr,
            operatorMatrix,
            operatorMatrix.getScratchFEMultivector(1, 0),
            operatorMatrix.getScratchFEMultivector(1, 1),
            operatorMatrix.getScratchFEMultivector(1, 2),
            operatorMatrix.getScratchFEMultivector(1, 3),
            d_dftParams);

        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.leave_subsection("Lanczos upper bound");
          }

        d_upperBoundUnWantedSpectrum = bounds.second;
      }

    if (d_dftParams.deviceFineGrainedTimings)
      {
        dftfe::utils::deviceSynchronize();
        computingTimerStandard.enter_subsection(
          "Chebyshev filtering on Device");
      }


    dftfe::uInt chebyshevOrder = d_dftParams.chebyshevOrder;

    //
    // set Chebyshev order
    //
    if (chebyshevOrder == 0)
      {
        chebyshevOrder =
          internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

        if (d_dftParams.orthogType.compare("CGS") == 0 &&
            !d_dftParams.isPseudopotential)
          chebyshevOrder *= 0.5;
      }

    chebyshevOrder =
      (isFirstScf && d_dftParams.isPseudopotential) ?
        chebyshevOrder *
          d_dftParams.chebyshevFilterPolyDegreeFirstScfScalingFactor :
        chebyshevOrder;


    //
    // output statements
    //
    if (d_dftParams.verbosity >= 2)
      {
        char buffer[100];

        sprintf(buffer,
                "%s:%18.10e\n",
                "upper bound of unwanted spectrum",
                d_upperBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s:%18.10e\n",
                "lower bound of unwanted spectrum",
                d_lowerBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s: %u\n\n",
                "Chebyshev polynomial degree",
                chebyshevOrder);
        pcout << buffer;
      }



    // two blocks of wavefunctions are filtered simultaneously when overlap
    // compute communication in chebyshev filtering is toggled on
    const dftfe::uInt numSimultaneousBlocks =
      d_dftParams.overlapComputeCommunCheby ? 2 : 1;
    dftfe::uInt       numSimultaneousBlocksCurrent = numSimultaneousBlocks;
    const dftfe::uInt numWfcsInBandGroup =
      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] -
      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId];
    dftfe::Int startIndexBandParal = totalNumberWaveFunctions;
    dftfe::Int numVectorsBandParal = 0;
    for (dftfe::uInt jvec = 0; jvec < totalNumberWaveFunctions;
         jvec += numSimultaneousBlocksCurrent * vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const dftfe::uInt BVec = vectorsBlockSize;

        // handle edge case when total number of blocks in a given band
        // group is not even in case of overlapping computation and
        // communciation in chebyshev filtering
        const dftfe::uInt leftIndexBandGroupMargin =
          (jvec / numWfcsInBandGroup) * numWfcsInBandGroup;
        numSimultaneousBlocksCurrent =
          ((jvec + numSimultaneousBlocks * BVec - leftIndexBandGroupMargin) <=
             numWfcsInBandGroup &&
           numSimultaneousBlocks == 2) ?
            2 :
            1;

        if ((jvec + numSimultaneousBlocksCurrent * BVec) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + numSimultaneousBlocksCurrent * BVec) >
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            if (jvec < startIndexBandParal)
              startIndexBandParal = jvec;
            numVectorsBandParal =
              jvec + numSimultaneousBlocksCurrent * BVec - startIndexBandParal;

            // copy from vector containg all wavefunction vectors to current
            // wavefunction vectors block
            BLASWrapperPtr->stridedCopyToBlockConstantStride(
              BVec,
              totalNumberWaveFunctions,
              localVectorSize,
              jvec,
              eigenVectorsFlattenedDevice,
              (*XBlock).begin());

            if (d_dftParams.overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              {
                BLASWrapperPtr->stridedCopyToBlockConstantStride(
                  BVec,
                  totalNumberWaveFunctions,
                  localVectorSize,
                  jvec + BVec,
                  eigenVectorsFlattenedDevice,
                  (*XBlock2).begin());
              }
            //
            // call Chebyshev filtering function only for the current block
            // or two simulataneous blocks (in case of overlap computation
            // and communication) to be filtered and does in-place filtering
            if (d_dftParams.useSinglePrecCheby && !isFirstFilteringCall)
              {
                eigenValuesBlock.resize(vectorsBlockSize *
                                        numSimultaneousBlocksCurrent);
                if (d_dftParams.overlapComputeCommunCheby &&
                    numSimultaneousBlocksCurrent == 2)
                  {
                    for (dftfe::uInt i = 0; i < 2 * BVec; i++)
                      {
                        eigenValuesBlock[i] = eigenValues[jvec + i];
                      }
                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*XBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*HXBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                      }
                    linearAlgebraOperationsDevice::
                      reformulatedChebyshevFilterOverlapComputeCommunication(
                        BLASWrapperPtr,
                        operatorMatrix,
                        (*XBlock),
                        (*HXBlock),
                        (*XBlock2),
                        (*HXBlock2),
                        (*XBlockFP32),
                        (*HXBlockFP32),
                        (*XBlock2FP32),
                        (*HXBlock2FP32),
                        eigenValuesBlock,
                        chebyshevOrder,
                        d_lowerBoundUnWantedSpectrum,
                        d_upperBoundUnWantedSpectrum,
                        d_lowerBoundWantedSpectrum,
                        d_dftParams.approxOverlapMatrix);
                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*XBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*HXBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                      }
                  }
                else
                  {
                    for (dftfe::uInt i = 0; i < BVec; i++)
                      {
                        eigenValuesBlock[i] = eigenValues[jvec + i];
                      }
                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                      }
                    linearAlgebraOperations::reformulatedChebyshevFilter(
                      BLASWrapperPtr,
                      operatorMatrix,
                      (*XBlock),
                      (*HXBlock),
                      (*XBlockFP32),
                      (*HXBlockFP32),
                      eigenValuesBlock,
                      chebyshevOrder,
                      d_lowerBoundUnWantedSpectrum,
                      d_upperBoundUnWantedSpectrum,
                      d_lowerBoundWantedSpectrum,
                      d_dftParams.approxOverlapMatrix);

                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                      }
                  }
              }
            else if (d_dftParams.useReformulatedChFSI && !isFirstFilteringCall)
              {
                eigenValuesBlock.resize(vectorsBlockSize *
                                        numSimultaneousBlocksCurrent);
                if (d_dftParams.overlapComputeCommunCheby &&
                    numSimultaneousBlocksCurrent == 2)
                  {
                    for (dftfe::uInt i = 0; i < 2 * BVec; i++)
                      {
                        eigenValuesBlock[i] = eigenValues[jvec + i];
                      }
                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*XBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*HXBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlock)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlockNew)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlock2)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlockNew2)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                      }
                    linearAlgebraOperationsDevice::
                      reformulatedChebyshevFilterOverlapComputeCommunication(
                        BLASWrapperPtr,
                        operatorMatrix,
                        (*XBlock),
                        (*HXBlock),
                        (*XBlock2),
                        (*HXBlock2),
                        (*ResidualBlock),
                        (*ResidualBlockNew),
                        (*ResidualBlock2),
                        (*ResidualBlockNew2),
                        eigenValuesBlock,
                        chebyshevOrder,
                        d_lowerBoundUnWantedSpectrum,
                        d_upperBoundUnWantedSpectrum,
                        d_lowerBoundWantedSpectrum,
                        d_dftParams.approxOverlapMatrix);
                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*XBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*HXBlock2).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlock)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlockNew)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlock2)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlockNew2)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                      }
                  }
                else
                  {
                    for (dftfe::uInt i = 0; i < BVec; i++)
                      {
                        eigenValuesBlock[i] = eigenValues[jvec + i];
                      }
                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlock)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlockNew)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                      }
                    linearAlgebraOperations::reformulatedChebyshevFilter(
                      BLASWrapperPtr,
                      operatorMatrix,
                      (*XBlock),
                      (*HXBlock),
                      (*ResidualBlock),
                      (*ResidualBlockNew),
                      eigenValuesBlock,
                      chebyshevOrder,
                      d_lowerBoundUnWantedSpectrum,
                      d_upperBoundUnWantedSpectrum,
                      d_lowerBoundWantedSpectrum,
                      d_dftParams.approxOverlapMatrix);

                    if (useMixedPrecOverall &&
                        d_dftParams.useSinglePrecCommunCheby)
                      {
                        (*XBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*HXBlock).setCommunicationPrecision(
                          dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlock)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlockNew)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                      }
                  }
              }
            else if (d_dftParams.overlapComputeCommunCheby &&
                     numSimultaneousBlocksCurrent == 2)
              {
                if (useMixedPrecOverall && d_dftParams.useSinglePrecCommunCheby)
                  {
                    (*XBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::single);
                    (*HXBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::single);
                    (*XBlock2).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::single);
                    (*HXBlock2).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::single);
                  }
                linearAlgebraOperationsDevice::
                  chebyshevFilterOverlapComputeCommunication(
                    operatorMatrix,
                    (*XBlock),
                    (*HXBlock),
                    (*XBlock2),
                    (*HXBlock2),
                    chebyshevOrder,
                    d_lowerBoundUnWantedSpectrum,
                    d_upperBoundUnWantedSpectrum,
                    d_lowerBoundWantedSpectrum);
                if (useMixedPrecOverall && d_dftParams.useSinglePrecCommunCheby)
                  {
                    (*XBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::full);
                    (*HXBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::full);
                    (*XBlock2).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::full);
                    (*HXBlock2).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::full);
                  }
              }
            else
              {
                if (useMixedPrecOverall && d_dftParams.useSinglePrecCommunCheby)
                  {
                    (*XBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::single);
                    (*HXBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::single);
                    if (d_dftParams.useReformulatedChFSI &&
                        !isFirstFilteringCall)
                      {
                        (*ResidualBlock)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                        (*ResidualBlockNew)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::single);
                      }
                  }

                if (d_dftParams.useReformulatedChFSI && !isFirstFilteringCall)
                  {
                    for (dftfe::uInt i = 0; i < BVec; i++)
                      {
                        eigenValuesBlock[i] = eigenValues[jvec + i];
                      }
                    linearAlgebraOperations::reformulatedChebyshevFilter(
                      BLASWrapperPtr,
                      operatorMatrix,
                      (*XBlock),
                      (*HXBlock),
                      (*ResidualBlock),
                      (*ResidualBlockNew),
                      eigenValuesBlock,
                      chebyshevOrder,
                      d_lowerBoundUnWantedSpectrum,
                      d_upperBoundUnWantedSpectrum,
                      d_lowerBoundWantedSpectrum,
                      d_dftParams.approxOverlapMatrix);
                  }

                else
                  {
                    linearAlgebraOperations::chebyshevFilter(
                      operatorMatrix,
                      (*XBlock),
                      (*HXBlock),
                      chebyshevOrder,
                      d_lowerBoundUnWantedSpectrum,
                      d_upperBoundUnWantedSpectrum,
                      d_lowerBoundWantedSpectrum);
                  }
                if (useMixedPrecOverall && d_dftParams.useSinglePrecCommunCheby)
                  {
                    (*XBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::full);
                    (*HXBlock).setCommunicationPrecision(
                      dftfe::utils::mpi::communicationPrecision::full);
                    if (d_dftParams.useReformulatedChFSI &&
                        !isFirstFilteringCall)
                      {
                        (*ResidualBlock)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                        (*ResidualBlockNew)
                          .setCommunicationPrecision(
                            dftfe::utils::mpi::communicationPrecision::full);
                      }
                  }
              }

            // copy current wavefunction vectors block to vector containing
            // all wavefunction vectors
            BLASWrapperPtr->stridedCopyFromBlockConstantStride(
              totalNumberWaveFunctions,
              BVec,
              localVectorSize,
              jvec,
              (*XBlock).begin(),
              eigenVectorsFlattenedDevice);

            if (d_dftParams.overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              BLASWrapperPtr->stridedCopyFromBlockConstantStride(
                totalNumberWaveFunctions,
                BVec,
                localVectorSize,
                jvec + BVec,
                (*XBlock2).begin(),
                eigenVectorsFlattenedDevice);
          }
        else
          {
            // set to zero wavefunctions which wont go through chebyshev
            // filtering inside a given band group
            dftfe::linearAlgebraOperationsDevice::setZero(
              numSimultaneousBlocksCurrent * BVec,
              localVectorSize,
              totalNumberWaveFunctions,
              eigenVectorsFlattenedDevice,
              jvec);
          }

      } // block loop

    if (d_dftParams.deviceFineGrainedTimings)
      {
        dftfe::utils::deviceSynchronize();
        computingTimerStandard.leave_subsection(
          "Chebyshev filtering on Device");

        if (d_dftParams.verbosity >= 4)
          pcout << "ChebyShev Filtering Done: " << std::endl;
      }


    if (numberBandGroups > 1)
      {
        std::vector<dataTypes::number> eigenVectorsFlattened(
          totalNumberWaveFunctions * localVectorSize, dataTypes::number(0.0));

        dftfe::utils::deviceMemcpyD2H(
          dftfe::utils::makeDataTypeDeviceCompatible(&eigenVectorsFlattened[0]),
          eigenVectorsFlattenedDevice,
          totalNumberWaveFunctions * localVectorSize *
            sizeof(dataTypes::number));

        MPI_Barrier(interBandGroupComm);


        MPI_Allreduce(MPI_IN_PLACE,
                      &eigenVectorsFlattened[0],
                      totalNumberWaveFunctions * localVectorSize,
                      dataTypes::mpi_type_id(&eigenVectorsFlattened[0]),
                      MPI_SUM,
                      interBandGroupComm);

        MPI_Barrier(interBandGroupComm);

        dftfe::utils::deviceMemcpyH2D(
          eigenVectorsFlattenedDevice,
          dftfe::utils::makeDataTypeDeviceCompatible(&eigenVectorsFlattened[0]),
          totalNumberWaveFunctions * localVectorSize *
            sizeof(dataTypes::number));
      }

    // if (d_dftParams.measureOnlyChebyTime)
    //  exit(0);



    if (d_dftParams.orthogType.compare("GS") == 0)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Classical Gram-Schmidt Orthonormalization not implemented in Device:"));
      }

    std::fill(eigenValues.begin(), eigenValues.end(), 0.0);


    {
      if (d_dftParams.useSubspaceProjectedSHEPGPU)
        {
          linearAlgebraOperationsDevice::pseudoGramSchmidtOrthogonalization(
            operatorMatrix,
            elpaScala,
            eigenVectorsFlattenedDevice,
            (*XBlock),
            (*HXBlock),
            localVectorSize,
            totalNumberWaveFunctions,
            d_mpiCommParent,
            operatorMatrix.getMPICommunicatorDomain(),
            devicecclMpiCommDomain,
            interBandGroupComm,
            BLASWrapperPtr,
            d_dftParams,
            useMixedPrecOverall);


          linearAlgebraOperationsDevice::rayleighRitz(
            operatorMatrix,
            elpaScala,
            eigenVectorsFlattenedDevice,
            (*XBlock),
            (*HXBlock),
            localVectorSize,
            totalNumberWaveFunctions,
            d_mpiCommParent,
            operatorMatrix.getMPICommunicatorDomain(),
            devicecclMpiCommDomain,
            interBandGroupComm,
            eigenValues,
            BLASWrapperPtr,
            d_dftParams,
            useMixedPrecOverall);
        }
      else
        {
          linearAlgebraOperationsDevice::rayleighRitzGEP(
            operatorMatrix,
            elpaScala,
            eigenVectorsFlattenedDevice,
            (*XBlock),
            (*HXBlock),
            localVectorSize,
            totalNumberWaveFunctions,
            d_mpiCommParent,
            operatorMatrix.getMPICommunicatorDomain(),
            devicecclMpiCommDomain,
            interBandGroupComm,
            eigenValues,
            BLASWrapperPtr,
            d_dftParams,
            useMixedPrecOverall);
        }
    }


    if (computeResidual)
      {
        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.enter_subsection("Residual norm");
          }

        linearAlgebraOperationsDevice::computeEigenResidualNorm(
          operatorMatrix,
          eigenVectorsFlattenedDevice,
          (*XBlock),
          (*HXBlock),
          localVectorSize,
          totalNumberWaveFunctions,
          eigenValues,
          d_mpiCommParent,
          operatorMatrix.getMPICommunicatorDomain(),
          interBandGroupComm,
          BLASWrapperPtr,
          residualNorms,
          d_dftParams,
          true);

        if (d_dftParams.deviceFineGrainedTimings)
          {
            dftfe::utils::deviceSynchronize();
            computingTimerStandard.leave_subsection("Residual norm");
          }
      }



    return d_upperBoundUnWantedSpectrum;
  }

  //
  // solve
  //


  //
  //
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverDevice::
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                &BLASWrapperPtr,
      dataTypes::number         *eigenVectorsFlattenedDevice,
      const dftfe::uInt          flattenedSize,
      const dftfe::uInt          totalNumberWaveFunctions,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double>       &densityMatDerFermiEnergy,
      utils::DeviceCCLWrapper   &devicecclMpiCommDomain,
      const MPI_Comm            &interBandGroupComm,
      dftfe::elpaScalaManager   &elpaScala)
  {
    dealii::TimerOutput computingTimerStandard(
      operatorMatrix.getMPICommunicatorDomain(),
      pcout,
      d_dftParams.reproducible_output || d_dftParams.verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    dftfe::utils::deviceSynchronize();
    computingTimerStandard.enter_subsection(
      "Density matrix first order response on Device");


    //
    // allocate memory for full flattened array on device and fill it up
    //
    const dftfe::uInt localVectorSize =
      flattenedSize / totalNumberWaveFunctions;


    const dftfe::uInt vectorsBlockSize =
      std::min(d_dftParams.chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedDeviceVec<dataTypes::number> *XBlock =
      &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 0);
    distributedDeviceVec<dataTypes::number> *HXBlock =
      &operatorMatrix.getScratchFEMultivector(vectorsBlockSize, 1);



    linearAlgebraOperationsDevice::densityMatrixEigenBasisFirstOrderResponse(
      operatorMatrix,
      eigenVectorsFlattenedDevice,
      (*XBlock),
      (*HXBlock),
      localVectorSize,
      totalNumberWaveFunctions,
      d_mpiCommParent,
      operatorMatrix.getMPICommunicatorDomain(),
      devicecclMpiCommDomain,
      interBandGroupComm,
      eigenValues,
      fermiEnergy,
      densityMatDerFermiEnergy,
      elpaScala,
      BLASWrapperPtr,
      d_dftParams);


    dftfe::utils::deviceSynchronize();
    computingTimerStandard.leave_subsection(
      "Density matrix first order response on Device");

    return;
  }

} // namespace dftfe
