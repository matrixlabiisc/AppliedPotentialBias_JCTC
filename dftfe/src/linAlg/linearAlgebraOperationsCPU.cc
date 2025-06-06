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


/** @file linearAlgebraOperationsOpt.cc
 *  @brief Contains linear algebra operations
 *
 */

#include "dftParameters.h"
#include "dftUtils.h"
#include "linearAlgebraOperations.h"
#include "linearAlgebraOperationsCPU.h"
#include "linearAlgebraOperationsInternal.h"
#include "constants.h"
#include "elpaScalaManager.h"
#include "pseudoGS.cc"
#ifdef USE_PETSC
#  include <deal.II/lac/slepc_solver.h>
#endif

namespace dftfe
{
  namespace linearAlgebraOperations
  {
    void
    inverse(double *A, dftfe::Int N)
    {
      int     nTmp  = N;
      int    *IPIV  = new int[N];
      int     LWORK = N * N;
      double *WORK  = new double[LWORK];
      int     INFO;

      dgetrf_(&nTmp, &nTmp, A, &nTmp, IPIV, &INFO);
      dgetri_(&nTmp, A, &nTmp, IPIV, WORK, &LWORK, &INFO);

      delete[] IPIV;
      delete[] WORK;
    }

    void
    callevd(const unsigned int dimensionMatrix,
            double            *matrix,
            double            *eigenValues)
    {
      int                info;
      const unsigned int lwork = 1 + 6 * dimensionMatrix +
                                 2 * dimensionMatrix * dimensionMatrix,
                         liwork = 3 + 5 * dimensionMatrix;
      std::vector<int>    iwork(liwork, 0);
      const char          jobz = 'V', uplo = 'U';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
              &uplo,
              &dimensionMatrix,
              matrix,
              &dimensionMatrix,
              eigenValues,
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<double>().swap(work);
      std::vector<int>().swap(iwork);
    }


    void
    callevd(const unsigned int    dimensionMatrix,
            std::complex<double> *matrix,
            double               *eigenValues)
    {
      int                info;
      const unsigned int lwork = 1 + 6 * dimensionMatrix +
                                 2 * dimensionMatrix * dimensionMatrix,
                         liwork = 3 + 5 * dimensionMatrix;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V', uplo = 'U';
      const unsigned int lrwork =
        1 + 5 * dimensionMatrix + 2 * dimensionMatrix * dimensionMatrix;
      std::vector<double>               rwork(lrwork);
      std::vector<std::complex<double>> work(lwork);


      zheevd_(&jobz,
              &uplo,
              &dimensionMatrix,
              matrix,
              &dimensionMatrix,
              eigenValues,
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<std::complex<double>>().swap(work);
      std::vector<int>().swap(iwork);
    }


    void
    callevr(const unsigned int    dimensionMatrix,
            std::complex<double> *matrixInput,
            std::complex<double> *eigenVectorMatrixOutput,
            double               *eigenValues)
    {
      char                              jobz = 'V', uplo = 'U', range = 'A';
      const double                      vl = 0.0, vu = 0.0;
      const unsigned int                il = 0, iu = 0;
      const double                      abstol = 1e-08;
      std::vector<unsigned int>         isuppz(2 * dimensionMatrix);
      const int                         lwork = 2 * dimensionMatrix;
      std::vector<std::complex<double>> work(lwork);
      const int                         liwork = 10 * dimensionMatrix;
      std::vector<int>                  iwork(liwork);
      const int                         lrwork = 24 * dimensionMatrix;
      std::vector<double>               rwork(lrwork);
      int                               info;

      zheevr_(&jobz,
              &range,
              &uplo,
              &dimensionMatrix,
              matrixInput,
              &dimensionMatrix,
              &vl,
              &vu,
              &il,
              &iu,
              &abstol,
              &dimensionMatrix,
              eigenValues,
              eigenVectorMatrixOutput,
              &dimensionMatrix,
              &isuppz[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);

      AssertThrow(info == 0, dealii::ExcMessage("Error in zheevr"));
    }



    void
    callevr(const unsigned int dimensionMatrix,
            double            *matrixInput,
            double            *eigenVectorMatrixOutput,
            double            *eigenValues)
    {
      char                      jobz = 'V', uplo = 'U', range = 'A';
      const double              vl = 0.0, vu = 0.0;
      const unsigned int        il = 0, iu = 0;
      const double              abstol = 0.0;
      std::vector<unsigned int> isuppz(2 * dimensionMatrix);
      const int                 lwork = 26 * dimensionMatrix;
      std::vector<double>       work(lwork);
      const int                 liwork = 10 * dimensionMatrix;
      std::vector<int>          iwork(liwork);
      int                       info;

      dsyevr_(&jobz,
              &range,
              &uplo,
              &dimensionMatrix,
              matrixInput,
              &dimensionMatrix,
              &vl,
              &vu,
              &il,
              &iu,
              &abstol,
              &dimensionMatrix,
              eigenValues,
              eigenVectorMatrixOutput,
              &dimensionMatrix,
              &isuppz[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      AssertThrow(info == 0, dealii::ExcMessage("Error in dsyevr"));
    }



    void
    callgemm(const unsigned int         numberEigenValues,
             const unsigned int         localVectorSize,
             const std::vector<double> &eigenVectorSubspaceMatrix,
             const std::vector<double> &X,
             std::vector<double>       &Y)

    {
      const char   transA = 'T', transB = 'N';
      const double alpha = 1.0, beta = 0.0;
      dgemm_(&transA,
             &transB,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha,
             &eigenVectorSubspaceMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta,
             &Y[0],
             &numberEigenValues);
    }


    void
    callgemm(const unsigned int                       numberEigenValues,
             const unsigned int                       localVectorSize,
             const std::vector<std::complex<double>> &eigenVectorSubspaceMatrix,
             const std::vector<std::complex<double>> &X,
             std::vector<std::complex<double>>       &Y)

    {
      const char                 transA = 'T', transB = 'N';
      const std::complex<double> alpha = 1.0, beta = 0.0;
      zgemm_(&transA,
             &transB,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha,
             &eigenVectorSubspaceMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta,
             &Y[0],
             &numberEigenValues);
    }



    template <typename T>
    void
    gramSchmidtOrthogonalization(T                 *X,
                                 const unsigned int numberVectors,
                                 const unsigned int localVectorSize,
                                 const MPI_Comm    &mpiComm)
    {
#ifdef USE_PETSC

      //
      // Create template PETSc vector to create BV object later
      //
      Vec templateVec;
      VecCreateMPI(mpiComm, localVectorSize, PETSC_DETERMINE, &templateVec);
      VecSetFromOptions(templateVec);


      //
      // Set BV options after creating BV object
      //
      BV columnSpaceOfVectors;
      BVCreate(mpiComm, &columnSpaceOfVectors);
      BVSetSizesFromVec(columnSpaceOfVectors, templateVec, numberVectors);
      BVSetFromOptions(columnSpaceOfVectors);


      //
      // create list of indices
      //
      std::vector<PetscInt>    indices(localVectorSize);
      std::vector<PetscScalar> data(localVectorSize, 0.0);

      PetscInt low, high;

      VecGetOwnershipRange(templateVec, &low, &high);


      for (PetscInt index = 0; index < localVectorSize; ++index)
        indices[index] = low + index;

      VecDestroy(&templateVec);

      //
      // Fill in data into BV object
      //
      Vec v;
      for (dftfe::uInt iColumn = 0; iColumn < numberVectors; ++iColumn)
        {
          BVGetColumn(columnSpaceOfVectors, iColumn, &v);
          VecSet(v, 0.0);
          for (dftfe::uInt iNode = 0; iNode < localVectorSize; ++iNode)
            data[iNode] = X[numberVectors * iNode + iColumn];

          VecSetValues(
            v, localVectorSize, &indices[0], &data[0], INSERT_VALUES);

          VecAssemblyBegin(v);
          VecAssemblyEnd(v);

          BVRestoreColumn(columnSpaceOfVectors, iColumn, &v);
        }

      //
      // orthogonalize
      //
      BVOrthogonalize(columnSpaceOfVectors, NULL);

      //
      // Copy data back into X
      //
      Vec          v1;
      PetscScalar *pointerv1;
      for (dftfe::uInt iColumn = 0; iColumn < numberVectors; ++iColumn)
        {
          BVGetColumn(columnSpaceOfVectors, iColumn, &v1);

          VecGetArray(v1, &pointerv1);

          for (dftfe::uInt iNode = 0; iNode < localVectorSize; ++iNode)
            X[numberVectors * iNode + iColumn] = pointerv1[iNode];

          VecRestoreArray(v1, &pointerv1);

          BVRestoreColumn(columnSpaceOfVectors, iColumn, &v1);
        }

      BVDestroy(&columnSpaceOfVectors);
#else
      AssertThrow(
        false,
        dealii::ExcMessage(
          "DFT-FE Error: Please link to dealii installed with petsc and slepc to Gram-Schidt orthogonalization."));
#endif
    }


    template <typename T>
    void
    rayleighRitzGEP(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                          &BLASWrapperPtr,
      elpaScalaManager    &elpaScala,
      T                   *X,
      const dftfe::uInt    numberWaveFunctions,
      const dftfe::uInt    localVectorSize,
      const MPI_Comm      &mpiCommParent,
      const MPI_Comm      &interBandGroupComm,
      const MPI_Comm      &mpi_communicator,
      std::vector<double> &eigenValues,
      const bool           useMixedPrec,
      const dftParameters &dftParams)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpi_communicator,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const dftfe::uInt rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();


      computing_timer.enter_subsection("XtHX and XtOX, RR GEP step");
      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<T> overlapMatPar(numberWaveFunctions,
                                              processGrid,
                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  T(0.0));

      dftfe::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                           processGrid,
                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  T(0.0));

      if (!(useMixedPrec) || !dftParams.useMixedPrecXtOX)
        {
          XtHXXtOX(operatorMatrix,
                   BLASWrapperPtr,
                   X,
                   numberWaveFunctions,
                   localVectorSize,
                   processGrid,
                   operatorMatrix.getMPICommunicatorDomain(),
                   interBandGroupComm,
                   dftParams,
                   projHamPar,
                   overlapMatPar);
        }
      else
        {
          XtHXXtOXMixedPrec(operatorMatrix,
                            BLASWrapperPtr,
                            X,
                            numberWaveFunctions,
                            dftParams.numCoreWfcForMixedPrecRR,
                            localVectorSize,
                            processGrid,
                            operatorMatrix.getMPICommunicatorDomain(),
                            interBandGroupComm,
                            dftParams,
                            projHamPar,
                            overlapMatPar);
        }

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<T> projHamParConjTrans(numberWaveFunctions,
                                                    processGrid,
                                                    rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  T(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      projHamPar.add(projHamParConjTrans, T(1.0), T(1.0));

      if (processGrid->is_process_active())
        for (dftfe::uInt i = 0; i < projHamPar.local_n(); ++i)
          {
            const dftfe::uInt glob_i = projHamPar.global_column(i);
            for (dftfe::uInt j = 0; j < projHamPar.local_m(); ++j)
              {
                const dftfe::uInt glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= T(0.5);
              }
          }

      computing_timer.leave_subsection("XtHX and XtOX, RR GEP step");

      //
      // compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      const dftfe::uInt numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);
      if (dftParams.useELPA)
        {
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          dftfe::ScaLAPACKMatrix<T> eigenVectors(numberWaveFunctions,
                                                 processGrid,
                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      T(0.0));

          dftfe::ScaLAPACKMatrix<T> overlapMatParConjTrans(numberWaveFunctions,
                                                           processGrid,
                                                           rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      T(0.0));



          overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

          if (processGrid->is_process_active())
            {
              int error;
              elpa_generalized_eigenvectors(elpaScala.getElpaHandle(),
                                            &projHamPar.local_el(0, 0),
                                            &overlapMatParConjTrans.local_el(0,
                                                                             0),
                                            &eigenValues[0],
                                            &eigenVectors.local_el(0, 0),
                                            0,
                                            &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(&eigenValues[0],
                    eigenValues.size(),
                    MPI_DOUBLE,
                    0,
                    operatorMatrix.getMPICommunicatorDomain());


          projHamPar.copy_conjugate_transposed(eigenVectors);

          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          // SConj=LConj*L^{T}
          computing_timer.enter_subsection(
            "Cholesky and triangular matrix invert");


          dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
          overlapMatPar.compute_cholesky_factorization();

          overlapMatPropertyPostCholesky = overlapMatPar.get_property();

          AssertThrow(
            overlapMatPropertyPostCholesky ==
              dftfe::LAPACKSupport::Property::lower_triangular,
            dealii::ExcMessage(
              "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


          // extract LConj
          dftfe::ScaLAPACKMatrix<T> LMatPar(
            numberWaveFunctions,
            processGrid,
            rowsBlockSize,
            dftfe::LAPACKSupport::Property::lower_triangular);

          if (processGrid->is_process_active())
            for (dftfe::uInt i = 0; i < LMatPar.local_n(); ++i)
              {
                const dftfe::uInt glob_i = LMatPar.global_column(i);
                for (dftfe::uInt j = 0; j < LMatPar.local_m(); ++j)
                  {
                    const dftfe::uInt glob_j = LMatPar.global_row(j);
                    if (glob_j < glob_i)
                      LMatPar.local_el(j, i) = T(0);
                    else
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                  }
              }

          // compute LConj^{-1}
          LMatPar.invert();

          computing_timer.leave_subsection(
            "Cholesky and triangular matrix invert");



          computing_timer.enter_subsection(
            "Compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR step");

          dftfe::ScaLAPACKMatrix<T> projHamParCopy(numberWaveFunctions,
                                                   processGrid,
                                                   rowsBlockSize);

          // compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
          // conjugate transpose LAPACK notation)
          LMatPar.mmult(projHamParCopy, projHamPar);
          projHamParCopy.zmCmult(projHamPar, LMatPar);

          computing_timer.leave_subsection(
            "Compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR step");
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, numberWaveFunctions - 1), true);
          projHamParCopy.copy_conjugate_transposed(projHamPar);
          projHamParCopy.mmult(projHamPar, LMatPar);

          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }

      computing_timer.enter_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      internal::broadcastAcrossInterCommScaLAPACKMat(processGrid,
                                                     projHamPar,
                                                     interBandGroupComm,
                                                     0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      computing_timer.leave_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      //
      // rotate the basis in the subspace
      // X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, stored in the column major
      // format In the above we use Q^{T}={QConjPrime}^{C}*LConj^{-1}
      if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrec))
        computing_timer.enter_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
      else
        computing_timer.enter_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR step");

      if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrec))
        internal::subspaceRotation(X,
                                   BLASWrapperPtr,
                                   numberWaveFunctions * localVectorSize,
                                   numberWaveFunctions,
                                   processGrid,
                                   interBandGroupComm,
                                   operatorMatrix.getMPICommunicatorDomain(),
                                   projHamPar,
                                   dftParams,
                                   false,
                                   false,
                                   false);
      else
        {
          if (std::is_same<T, std::complex<double>>::value)
            internal::subspaceRotationMixedPrec<T, std::complex<float>>(
              X,
              BLASWrapperPtr,
              numberWaveFunctions * localVectorSize,
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              operatorMatrix.getMPICommunicatorDomain(),
              projHamPar,
              dftParams,
              false,
              false);
          else
            internal::subspaceRotationMixedPrec<T, float>(
              X,
              BLASWrapperPtr,
              numberWaveFunctions * localVectorSize,
              numberWaveFunctions,
              processGrid,
              interBandGroupComm,
              operatorMatrix.getMPICommunicatorDomain(),
              projHamPar,
              dftParams,
              false,
              false);
        }

      if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrec))
        computing_timer.leave_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR step");
      else
        computing_timer.leave_subsection(
          "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR step");
    }
    void
    XtHXXtOXMixedPrec(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                      &BLASWrapperPtr,
      const dataTypes::number                         *X,
      const dftfe::uInt                                N,
      const dftfe::uInt                                Ncore,
      const dftfe::uInt                                numberDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftParameters                             &dftParams,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projOverlapPar,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      // create temporary arrays XBlock,Hx
      distributedCPUMultiVec<dataTypes::number> *XBlock, *HXBlock;

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      /*
       * X^{T}*H*Xc is done in a blocked approach for memory optimization:
       * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
       * has a much smaller memory compared to X^{T}*Hc*Xc.
       * X^{T} (denoted by X in the code with column major format storage)
       * is a matrix with size (N x MLoc).
       * MLoc, which is number of local dofs is denoted by numberDofs in the
       * code. Xc denotes complex conjugate of X. XcBlock is a matrix of size
       * (MLoc x B). B is the block size. A further optimization is done to
       * reduce floating point operations: As X^{T}*Hc*Xc is a Hermitian
       matrix,
       * it suffices to compute only the lower triangular part. To exploit
       this,
       * we do X^{T}*Hc*Xc=Sum_{blocks} XTrunc^{T}*Hc*XcBlock where
       XTrunc^{T}
       * is a (D x MLoc) sub matrix of X^{T} with the row indices ranging
       from
       * the lowest global index of XcBlock (denoted by jvec in the code) to
       N.
       * D=N-jvec. The parallel ScaLapack matrix projHamPar is directly
       filled
       * from the XTrunc^{T}*Hc*XcBlock result
       */

      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      std::vector<dataTypes::numberFP32> projHamBlockSinglePrec(
        N * vectorsBlockSize, 0.0);
      std::vector<dataTypes::number> projHamBlockDoublePrec(vectorsBlockSize *
                                                              vectorsBlockSize,
                                                            0.0);
      std::vector<dataTypes::number> projHamBlock(N * vectorsBlockSize, 0.0);

      std::vector<dataTypes::numberFP32> HXBlockSinglePrec;

      std::vector<dataTypes::numberFP32> XSinglePrec(X, X + numberDofs * N);


      if (dftParams.verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpiCommDomain,
          "Inside Blocked XtHX with parallel projected Ham matrix");

      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);
          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              HXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
              HXBlockSinglePrec.resize(B * numberDofs);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // fill XBlock^{T} from X:
              for (dftfe::uInt iNode = 0; iNode < numberDofs; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * N + jvec + iWave];

              // evaluate H times XBlock and store in HXBlock^{T}
              operatorMatrix.HX(*XBlock,
                                1.0,
                                0.0,
                                0.0,
                                *HXBlock,
                                onlyHPrimePartForFirstOrderDensityMatResponse);



              const char transA = 'N';
              const char transB =
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T';
              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              if (jvec + B > Ncore)
                {
                  const dftfe::uInt D = N - jvec;

                  // Comptute local XTrunc^{T}*HXcBlock.
                  BLASWrapperPtr->xgemm(transA,
                                        transB,
                                        D,
                                        B,
                                        numberDofs,
                                        &alpha,
                                        &X[0] + jvec,
                                        N,
                                        HXBlock->data(),
                                        B,
                                        &beta,
                                        &projHamBlock[0],
                                        D);

                  // Sum local XTrunc^{T}*HXcBlock across domain decomposition
                  // processors
                  MPI_Allreduce(MPI_IN_PLACE,
                                &projHamBlock[0],
                                D * B,
                                dataTypes::mpi_type_id(&projHamBlock[0]),
                                MPI_SUM,
                                mpiCommDomain);


                  // Copying only the lower triangular part to the ScaLAPACK
                  // projected Hamiltonian matrix
                  if (processGrid->is_process_active())
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + jvec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + jvec];
                          for (dftfe::uInt i = jvec + j; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                projHamPar.local_el(it->second, localColumnId) =
                                  projHamBlock[j * D + i - jvec];
                            }
                        }
                }
              else
                {
                  const dataTypes::numberFP32 alphaSinglePrec =
                                                dataTypes::numberFP32(1.0),
                                              betaSinglePrec =
                                                dataTypes::numberFP32(0.0);


                  const dftfe::uInt D = N - jvec;

                  // full prec gemm
                  BLASWrapperPtr->xgemm(transA,
                                        transB,
                                        B,
                                        B,
                                        numberDofs,
                                        &alpha,
                                        &X[0] + jvec,
                                        N,
                                        HXBlock->data(),
                                        B,
                                        &beta,
                                        &projHamBlockDoublePrec[0],
                                        B);
                  const dftfe::uInt DRem = D - B;
                  if (DRem != 0)
                    {
                      for (dftfe::uInt i = 0; i < numberDofs * B; ++i)
                        HXBlockSinglePrec[i] = HXBlock->data()[i];
                      BLASWrapperPtr->xgemm(transA,
                                            transB,
                                            DRem,
                                            B,
                                            numberDofs,
                                            &alphaSinglePrec,
                                            &XSinglePrec[0] + jvec + B,
                                            N,
                                            &HXBlockSinglePrec[0],
                                            B,
                                            &betaSinglePrec,
                                            &projHamBlockSinglePrec[0],
                                            DRem);
                    }

                  MPI_Allreduce(MPI_IN_PLACE,
                                &projHamBlockDoublePrec[0],
                                B * B,
                                dataTypes::mpi_type_id(
                                  &projHamBlockDoublePrec[0]),
                                MPI_SUM,
                                mpiCommDomain);
                  MPI_Allreduce(MPI_IN_PLACE,
                                &projHamBlockSinglePrec[0],
                                DRem * B,
                                dataTypes::mpi_type_id(
                                  &projHamBlockSinglePrec[0]),
                                MPI_SUM,
                                mpiCommDomain);

                  for (dftfe::uInt i = 0; i < B; ++i)
                    {
                      for (dftfe::uInt j = 0; j < B; ++j)
                        projHamBlock[i * D + j] =
                          projHamBlockDoublePrec[i * B + j];

                      for (dftfe::uInt j = 0; j < DRem; ++j)
                        projHamBlock[i * D + j + B] =
                          projHamBlockSinglePrec[i * DRem + j];
                    }

                  if (processGrid->is_process_active())
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + jvec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + jvec];
                          for (dftfe::uInt i = jvec + j; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                projHamPar.local_el(it->second, localColumnId) =
                                  projHamBlock[j * D + i - jvec];
                            }
                        }
                }

              // evaluate H times XBlock and store in HXBlock^{T}
              operatorMatrix.overlapMatrixTimesX(*XBlock,
                                                 1.0,
                                                 0.0,
                                                 0.0,
                                                 *HXBlock,
                                                 dftParams.approxOverlapMatrix);



              const dftfe::uInt D = N - jvec;

              BLASWrapperPtr->xgemm(transA,
                                    transB,
                                    B,
                                    B,
                                    numberDofs,
                                    &alpha,
                                    &X[0] + jvec,
                                    N,
                                    HXBlock->data(),
                                    B,
                                    &beta,
                                    &projHamBlockDoublePrec[0],
                                    B);
              const dftfe::uInt DRem = D - B;
              if (DRem != 0)
                {
                  const dataTypes::numberFP32 alphaSinglePrec =
                                                dataTypes::numberFP32(1.0),
                                              betaSinglePrec =
                                                dataTypes::numberFP32(0.0);
                  for (dftfe::uInt i = 0; i < numberDofs * B; ++i)
                    HXBlockSinglePrec[i] = HXBlock->data()[i];
                  BLASWrapperPtr->xgemm(transA,
                                        transB,
                                        DRem,
                                        B,
                                        numberDofs,
                                        &alphaSinglePrec,
                                        &XSinglePrec[0] + jvec + B,
                                        N,
                                        &HXBlockSinglePrec[0],
                                        B,
                                        &betaSinglePrec,
                                        &projHamBlockSinglePrec[0],
                                        DRem);
                }

              // Sum local XTrunc^{T}*XcBlock for double precision across
              // domain decomposition processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projHamBlockDoublePrec[0],
                            B * B,
                            dataTypes::mpi_type_id(&projHamBlockDoublePrec[0]),
                            MPI_SUM,
                            mpiCommDomain);

              // Sum local XTrunc^{T}*XcBlock for single precision across
              // domain decomposition processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projHamBlockSinglePrec[0],
                            DRem * B,
                            dataTypes::mpi_type_id(&projHamBlockSinglePrec[0]),
                            MPI_SUM,
                            mpiCommDomain);

              for (dftfe::uInt i = 0; i < B; ++i)
                {
                  for (dftfe::uInt j = 0; j < B; ++j)
                    projHamBlock[i * D + j] = projHamBlockDoublePrec[i * B + j];

                  for (dftfe::uInt j = 0; j < DRem; ++j)
                    projHamBlock[i * D + j + B] =
                      projHamBlockSinglePrec[i * DRem + j];
                }

              // Copying only the lower triangular part to the ScaLAPACK
              // overlap matrix
              if (processGrid->is_process_active())
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = jvec + j; i < N; ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projOverlapPar.local_el(it->second, localColumnId) =
                              projHamBlock[j * D + i - jvec];
                        }
                    }


            } // band parallelization

        } // block loop

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projOverlapPar, interBandGroupComm);
        }
    }


    template <typename T>
    void
    rayleighRitz(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                          &BLASWrapperPtr,
      elpaScalaManager    &elpaScala,
      T                   *X,
      const dftfe::uInt    numberWaveFunctions,
      const dftfe::uInt    localVectorSize,
      const MPI_Comm      &mpiCommParent,
      const MPI_Comm      &interBandGroupComm,
      const MPI_Comm      &mpi_communicator,
      std::vector<double> &eigenValues,
      const dftParameters &dftParams,
      const bool           doCommAfterBandParal)

    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpi_communicator,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);
      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      const dftfe::uInt rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      dftfe::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                           processGrid,
                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  T(0.0));

      computing_timer.enter_subsection("Blocked XtHX, RR step");
      XtHX(operatorMatrix,
           BLASWrapperPtr,
           X,
           numberWaveFunctions,
           localVectorSize,
           processGrid,
           mpi_communicator,
           interBandGroupComm,
           dftParams,
           projHamPar);
      computing_timer.leave_subsection("Blocked XtHX, RR step");

      //
      // compute eigendecomposition of ProjHam HConjProj= QConj*D*QConj^{C} (C
      // denotes conjugate transpose LAPACK notation)
      //
      const dftfe::uInt numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);
      if (dftParams.useELPA)
        {
          computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          dftfe::ScaLAPACKMatrix<T> eigenVectors(numberWaveFunctions,
                                                 processGrid,
                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      T(0.0));

          // For ELPA eigendecomposition the full matrix is required unlike
          // ScaLAPACK which can work with only the lower triangular part
          dftfe::ScaLAPACKMatrix<T> projHamParConjTrans(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&projHamParConjTrans.local_el(0, 0),
                      &projHamParConjTrans.local_el(0, 0) +
                        projHamParConjTrans.local_m() *
                          projHamParConjTrans.local_n(),
                      T(0.0));


          projHamParConjTrans.copy_conjugate_transposed(projHamPar);
          projHamPar.add(projHamParConjTrans, T(1.0), T(1.0));

          if (processGrid->is_process_active())
            for (dftfe::uInt i = 0; i < projHamPar.local_n(); ++i)
              {
                const dftfe::uInt glob_i = projHamPar.global_column(i);
                for (dftfe::uInt j = 0; j < projHamPar.local_m(); ++j)
                  {
                    const dftfe::uInt glob_j = projHamPar.global_row(j);
                    if (glob_i == glob_j)
                      projHamPar.local_el(j, i) *= T(0.5);
                  }
              }

          if (processGrid->is_process_active())
            {
              int error;
              elpa_eigenvectors(elpaScala.getElpaHandle(),
                                &projHamPar.local_el(0, 0),
                                &eigenValues[0],
                                &eigenVectors.local_el(0, 0),
                                &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(&eigenValues[0],
                    eigenValues.size(),
                    MPI_DOUBLE,
                    0,
                    mpi_communicator);


          eigenVectors.copy_to(projHamPar);

          computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, numberWaveFunctions - 1), true);
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }


      computing_timer.enter_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      internal::broadcastAcrossInterCommScaLAPACKMat(processGrid,
                                                     projHamPar,
                                                     interBandGroupComm,
                                                     0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      computing_timer.leave_subsection(
        "Broadcast eigvec and eigenvalues across band groups, RR step");
      //
      // rotate the basis in the subspace X = X*Q, implemented as
      // X^{T}=Qc^{C}*X^{T} with X^{T} stored in the column major format
      //
      computing_timer.enter_subsection("Blocked subspace rotation, RR step");
      dftfe::ScaLAPACKMatrix<T> projHamParCopy(numberWaveFunctions,
                                               processGrid,
                                               rowsBlockSize);
      projHamParCopy.copy_conjugate_transposed(projHamPar);
      internal::subspaceRotation(X,
                                 BLASWrapperPtr,
                                 numberWaveFunctions * localVectorSize,
                                 numberWaveFunctions,
                                 processGrid,
                                 interBandGroupComm,
                                 mpi_communicator,
                                 projHamParCopy,
                                 dftParams,
                                 false,
                                 false,
                                 doCommAfterBandParal);

      computing_timer.leave_subsection("Blocked subspace rotation, RR step");
    }



    template <typename NumberType>
    void
    elpaDiagonalization(
      elpaScalaManager                                &elpaScala,
      const dftfe::uInt                                numberWaveFunctions,
      const MPI_Comm                                  &mpiComm,
      std::vector<double>                             &eigenValues,
      dftfe::ScaLAPACKMatrix<NumberType>              &projHamPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
    {
      const dftfe::uInt rowsBlockSize = elpaScala.getScalapackBlockSize();

      dftfe::ScaLAPACKMatrix<NumberType> eigenVectors(numberWaveFunctions,
                                                      processGrid,
                                                      rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&eigenVectors.local_el(0, 0),
                  &eigenVectors.local_el(0, 0) +
                    eigenVectors.local_m() * eigenVectors.local_n(),
                  NumberType(0.0));

      // For ELPA eigendecomposition the full matrix is required unlike
      // ScaLAPACK which can work with only the lower triangular part
      dftfe::ScaLAPACKMatrix<NumberType> projHamParTrans(numberWaveFunctions,
                                                         processGrid,
                                                         rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParTrans.local_el(0, 0),
                  &projHamParTrans.local_el(0, 0) +
                    projHamParTrans.local_m() * projHamParTrans.local_n(),
                  0.0);


      projHamParTrans.copy_transposed(projHamPar);
      projHamPar.add(projHamParTrans, 1.0, 1.0);

      if (processGrid->is_process_active())
        for (dftfe::uInt i = 0; i < projHamPar.local_n(); ++i)
          {
            const dftfe::uInt glob_i = projHamPar.global_column(i);
            for (dftfe::uInt j = 0; j < projHamPar.local_m(); ++j)
              {
                const dftfe::uInt glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= 0.5;
              }
          }

      if (processGrid->is_process_active())
        {
          int error;
          elpa_eigenvectors(elpaScala.getElpaHandle(),
                            &projHamPar.local_el(0, 0),
                            &eigenValues[0],
                            &eigenVectors.local_el(0, 0),
                            &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_eigenvectors error."));
        }


      MPI_Bcast(&eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiComm);


      eigenVectors.copy_to(projHamPar);
    }

    template <typename NumberType>
    void
    elpaDiagonalizationGEP(
      elpaScalaManager                                &elpaScala,
      const dftfe::uInt                                numberWaveFunctions,
      const MPI_Comm                                  &mpiComm,
      std::vector<double>                             &eigenValues,
      dftfe::ScaLAPACKMatrix<NumberType>              &projHamPar,
      dftfe::ScaLAPACKMatrix<NumberType>              &overlapMatPar,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid)
    {
      const dftfe::uInt rowsBlockSize = elpaScala.getScalapackBlockSize();

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;

      // For ELPA cholesky only the upper triangular part is enough
      dftfe::ScaLAPACKMatrix<double> overlapMatParTrans(numberWaveFunctions,
                                                        processGrid,
                                                        rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatParTrans.local_el(0, 0),
                  &overlapMatParTrans.local_el(0, 0) +
                    overlapMatParTrans.local_m() * overlapMatParTrans.local_n(),
                  0.0);

      overlapMatParTrans.copy_transposed(overlapMatPar);

      if (processGrid->is_process_active())
        {
          int error;
          elpa_cholesky(elpaScala.getElpaHandle(),
                        &overlapMatParTrans.local_el(0, 0),
                        &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_cholesky_d error."));
        }
      overlapMatParTrans.copy_to(overlapMatPar);
      overlapMatPropertyPostCholesky =
        dftfe::LAPACKSupport::Property::upper_triangular;

      AssertThrow(
        overlapMatPropertyPostCholesky ==
            dftfe::LAPACKSupport::Property::lower_triangular ||
          overlapMatPropertyPostCholesky ==
            dftfe::LAPACKSupport::Property::upper_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));

      dftfe::ScaLAPACKMatrix<double> LMatPar(numberWaveFunctions,
                                             processGrid,
                                             rowsBlockSize,
                                             overlapMatPropertyPostCholesky);

      // copy triangular part of overlapMatPar into LMatPar
      if (processGrid->is_process_active())
        for (dftfe::uInt i = 0; i < overlapMatPar.local_n(); ++i)
          {
            const dftfe::uInt glob_i = overlapMatPar.global_column(i);
            for (dftfe::uInt j = 0; j < overlapMatPar.local_m(); ++j)
              {
                const dftfe::uInt glob_j = overlapMatPar.global_row(j);
                if (overlapMatPropertyPostCholesky ==
                    dftfe::LAPACKSupport::Property::lower_triangular)
                  {
                    if (glob_i <= glob_j)
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                    else
                      LMatPar.local_el(j, i) = 0;
                  }
                else
                  {
                    if (glob_j <= glob_i)
                      LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
                    else
                      LMatPar.local_el(j, i) = 0;
                  }
              }
          }


      // invert triangular matrix
      if (processGrid->is_process_active())
        {
          int error;
          elpa_invert_triangular(elpaScala.getElpaHandle(),
                                 &LMatPar.local_el(0, 0),
                                 &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_invert_trm_d error."));
        }

      // For ELPA eigendecomposition the full matrix is required unlike
      // ScaLAPACK which can work with only the lower triangular part
      dftfe::ScaLAPACKMatrix<double> projHamParTrans(numberWaveFunctions,
                                                     processGrid,
                                                     rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParTrans.local_el(0, 0),
                  &projHamParTrans.local_el(0, 0) +
                    projHamParTrans.local_m() * projHamParTrans.local_n(),
                  0.0);


      projHamParTrans.copy_transposed(projHamPar);
      projHamPar.add(projHamParTrans, 1.0, 1.0);

      if (processGrid->is_process_active())
        for (dftfe::uInt i = 0; i < projHamPar.local_n(); ++i)
          {
            const dftfe::uInt glob_i = projHamPar.global_column(i);
            for (dftfe::uInt j = 0; j < projHamPar.local_m(); ++j)
              {
                const dftfe::uInt glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= 0.5;
              }
          }

      dftfe::ScaLAPACKMatrix<double> projHamParCopy(numberWaveFunctions,
                                                    processGrid,
                                                    rowsBlockSize);

      if (overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular)
        {
          LMatPar.mmult(projHamParCopy, projHamPar);
          projHamParCopy.mTmult(projHamPar, LMatPar);
        }
      else
        {
          LMatPar.Tmmult(projHamParCopy, projHamPar);
          projHamParCopy.mmult(projHamPar, LMatPar);
        }

      //
      // compute eigendecomposition of ProjHam
      //
      const dftfe::uInt numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);

      dftfe::ScaLAPACKMatrix<double> eigenVectors(numberWaveFunctions,
                                                  processGrid,
                                                  rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&eigenVectors.local_el(0, 0),
                  &eigenVectors.local_el(0, 0) +
                    eigenVectors.local_m() * eigenVectors.local_n(),
                  0.0);

      if (processGrid->is_process_active())
        {
          int error;
          elpa_eigenvectors(elpaScala.getElpaHandle(),
                            &projHamPar.local_el(0, 0),
                            &eigenValues[0],
                            &eigenVectors.local_el(0, 0),
                            &error);
          AssertThrow(error == ELPA_OK,
                      dealii::ExcMessage(
                        "DFT-FE Error: elpa_eigenvectors error."));
        }


      MPI_Bcast(&eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiComm);


      eigenVectors.copy_to(projHamPar);

      projHamPar.copy_to(projHamParCopy);
      if (overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular)
        LMatPar.Tmmult(projHamPar, projHamParCopy);
      else
        LMatPar.mmult(projHamPar, projHamParCopy);
    }



    template <typename T>
    void
    computeEigenResidualNorm(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                &BLASWrapperPtr,
      T                         *X,
      const std::vector<double> &eigenValues,
      const dftfe::uInt          totalNumberVectors,
      const dftfe::uInt          localVectorSize,
      const MPI_Comm            &mpiCommParent,
      const MPI_Comm            &mpiCommDomain,
      const MPI_Comm            &interBandGroupComm,
      std::vector<double>       &residualNorm,
      const dftParameters       &dftParams)

    {
      //
      // get the number of eigenVectors
      //
      std::vector<double> residualNormSquare(totalNumberVectors, 0.0);

      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, totalNumberVectors, bandGroupLowHighPlusOneIndices);

      // create temporary arrays XBlock,HXBlock
      distributedCPUMultiVec<T> *XBlock, *HXBlock;

      // Do H*X using a blocked approach and compute
      // the residual norms: H*XBlock-XBlock*D, where
      // D is the eigenvalues matrix.
      // The blocked approach avoids additional full
      // wavefunction matrix memory
      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      for (dftfe::uInt jvec = 0; jvec < totalNumberVectors;
           jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge"
          const dftfe::uInt B =
            std::min(vectorsBlockSize, totalNumberVectors - jvec);

          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              HXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              XBlock->setValue(T(0.));
              // fill XBlock from X:
              for (dftfe::uInt iNode = 0; iNode < localVectorSize; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * totalNumberVectors + jvec + iWave];

              // evaluate H times XBlock and store in HXBlock
              operatorMatrix.overlapMatrixTimesX(*XBlock,
                                                 1.0,
                                                 0.0,
                                                 0.0,
                                                 *HXBlock,
                                                 dftParams.approxOverlapMatrix);
              for (dftfe::uInt iDof = 0; iDof < localVectorSize; ++iDof)
                for (dftfe::uInt iWave = 0; iWave < B; iWave++)
                  {
                    HXBlock->data()[B * iDof + iWave] *=
                      eigenValues[jvec + iWave];
                  }

              operatorMatrix.HX(*XBlock, 1.0, -1.0, 0.0, *HXBlock);
              if (dftParams.approxOverlapMatrix)
                {
                  BLASWrapperPtr->stridedBlockScale(
                    B,
                    localVectorSize,
                    1.0,
                    operatorMatrix.getInverseSqrtMassVector().data(),
                    HXBlock->data());
                }
              //   pointWiseScaleWithDiagonal(
              //     operatorMatrix.getInverseSqrtMassVector().data(),
              //     B,
              //     localVectorSize,
              //     HXBlock->data());
              // compute residual norms:
              for (dftfe::uInt iDof = 0; iDof < localVectorSize; ++iDof)
                for (dftfe::uInt iWave = 0; iWave < B; iWave++)
                  {
                    const double temp =
                      std::abs(HXBlock->data()[B * iDof + iWave]);
                    residualNormSquare[jvec + iWave] += temp * temp;
                  }
            }
        }


      dealii::Utilities::MPI::sum(residualNormSquare,
                                  mpiCommDomain,
                                  residualNormSquare);

      dealii::Utilities::MPI::sum(residualNormSquare,
                                  interBandGroupComm,
                                  residualNormSquare);

      if (dftParams.verbosity >= 4)
        {
          if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
            std::cout << "L-2 Norm of residue   :" << std::endl;
        }
      for (dftfe::uInt iWave = 0; iWave < totalNumberVectors; ++iWave)
        residualNorm[iWave] = sqrt(residualNormSquare[iWave]);

      if (dftParams.verbosity >= 4 &&
          dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
        for (dftfe::uInt iWave = 0; iWave < totalNumberVectors; ++iWave)
          std::cout << "eigen vector " << iWave << ": " << residualNorm[iWave]
                    << std::endl;

      if (dftParams.verbosity >= 4)
        if (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0)
          std::cout << std::endl;
    }

#ifdef USE_COMPLEX
    dftfe::uInt
    lowdenOrthogonalization(std::vector<std::complex<double>> &X,
                            const dftfe::uInt                  numberVectors,
                            const MPI_Comm                    &mpiComm,
                            const dftParameters               &dftParams)
    {
      const unsigned int localVectorSize = X.size() / numberVectors;
      std::vector<std::complex<double>> overlapMatrix(numberVectors *
                                                        numberVectors,
                                                      0.0);

      //
      // blas level 3 dgemm flags
      //
      const double       alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;

      //
      // compute overlap matrix S = {(Zc)^T}*Z on local proc
      // where Z is a matrix with size number of degrees of freedom times number
      // of column vectors and (Zc)^T is conjugate transpose of Z Since input
      // "X" is stored as number of column vectors times number of degrees of
      // freedom matrix corresponding to column-major format required for blas,
      // we compute the transpose of overlap matrix i.e S^{T} = X*{(Xc)^T} here
      //
      const char uplo  = 'U';
      const char trans = 'N';

      zherk_(&uplo,
             &trans,
             &numberEigenValues,
             &localVectorSize,
             &alpha,
             &X[0],
             &numberEigenValues,
             &beta,
             &overlapMatrix[0],
             &numberEigenValues);


      dealii::Utilities::MPI::sum(overlapMatrix, mpiComm, overlapMatrix);

      //
      // evaluate the conjugate of {S^T} to get actual overlap matrix
      //
      for (dftfe::uInt i = 0; i < overlapMatrix.size(); ++i)
        overlapMatrix[i] = std::conj(overlapMatrix[i]);


      //
      // set lapack eigen decomposition flags and compute eigendecomposition of
      // S = Q*D*Q^{H}
      //
      int                info;
      const unsigned int lwork = 1 + 6 * numberVectors +
                                 2 * numberVectors * numberVectors,
                         liwork = 3 + 5 * numberVectors;
      std::vector<int>   iwork(liwork, 0);
      const char         jobz = 'V';
      const unsigned int lrwork =
        1 + 5 * numberVectors + 2 * numberVectors * numberVectors;
      std::vector<double>               rwork(lrwork, 0.0);
      std::vector<std::complex<double>> work(lwork);
      std::vector<double>               eigenValuesOverlap(numberVectors, 0.0);

      zheevd_(&jobz,
              &uplo,
              &numberEigenValues,
              &overlapMatrix[0],
              &numberEigenValues,
              &eigenValuesOverlap[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      rwork.clear();
      std::vector<std::complex<double>>().swap(work);
      std::vector<double>().swap(rwork);
      std::vector<int>().swap(iwork);

      //
      // compute D^{-1/4} where S = Q*D*Q^{H}
      //
      std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues,
                                                         0.0);

      dftfe::uInt nanFlag = 0;
      for (dftfe::uInt i = 0; i < numberEigenValues; ++i)
        {
          invFourthRootEigenValuesMatrix[i] =
            1.0 / pow(eigenValuesOverlap[i], 1.0 / 4);
          if (std::isnan(invFourthRootEigenValuesMatrix[i]) ||
              eigenValuesOverlap[i] < 1e-13)
            {
              nanFlag = 1;
              break;
            }
        }
      nanFlag = dealii::Utilities::MPI::max(nanFlag, mpiComm);
      if (nanFlag == 1)
        return nanFlag;

      //
      // Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling
      // "zheevd"
      //
      const unsigned int inc = 1;
      for (dftfe::uInt i = 0; i < numberEigenValues; ++i)
        {
          const double scalingCoeff = invFourthRootEigenValuesMatrix[i];
          zdscal_(&numberEigenValues,
                  &scalingCoeff,
                  &overlapMatrix[0] + i * numberEigenValues,
                  &inc);
        }

      //
      // Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{H} = (Q*D^{-1/4})*(Q*D^{-1/4))^{H}
      //
      std::vector<std::complex<double>> invSqrtOverlapMatrix(
        numberEigenValues * numberEigenValues, 0.0);
      const char                 transA1 = 'N';
      const char                 transB1 = 'C';
      const std::complex<double> alpha1 = 1.0, beta1 = 0.0;


      zgemm_(&transA1,
             &transB1,
             &numberEigenValues,
             &numberEigenValues,
             &numberEigenValues,
             &alpha1,
             &overlapMatrix[0],
             &numberEigenValues,
             &overlapMatrix[0],
             &numberEigenValues,
             &beta1,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues);

      //
      // free up memory associated with overlapMatrix
      //
      overlapMatrix.clear();
      std::vector<std::complex<double>>().swap(overlapMatrix);

      //
      // Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but
      // implemented as Y^T = {S^{-1/2}}^T*{X^T} using the column major format
      // of blas
      //
      const char transA2 = 'T', transB2 = 'N';
      // dealii::parallel::distributed::Vector<std::complex<double> >
      // orthoNormalizedBasis;
      std::vector<std::complex<double>> orthoNormalizedBasis(X.size(), 0.0);

      zgemm_(&transA2,
             &transB2,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha1,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta1,
             &orthoNormalizedBasis[0],
             &numberEigenValues);


      X = orthoNormalizedBasis;

      return 0;
    }
#else
    dftfe::uInt
    lowdenOrthogonalization(std::vector<double> &X,
                            const dftfe::uInt    numberVectors,
                            const MPI_Comm      &mpiComm,
                            const dftParameters &dftParams)
    {
      const unsigned int localVectorSize = X.size() / numberVectors;

      std::vector<double> overlapMatrix(numberVectors * numberVectors, 0.0);


      dealii::ConditionalOStream pcout(
        std::cout, (dealii::Utilities::MPI::this_mpi_process(mpiComm) == 0));

      dealii::TimerOutput computing_timer(mpiComm,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);



      //
      // blas level 3 dgemm flags
      //
      const double       alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;
      const char         uplo              = 'U';
      const char         trans             = 'N';

      //
      // compute overlap matrix S = {(Z)^T}*Z on local proc
      // where Z is a matrix with size number of degrees of freedom times number
      // of column vectors and (Z)^T is transpose of Z Since input "X" is stored
      // as number of column vectors times number of degrees of freedom matrix
      // corresponding to column-major format required for blas, we compute
      // the overlap matrix as S = S^{T} = X*{X^T} here
      //

      computing_timer.enter_subsection("local overlap matrix for lowden");
      dsyrk_(&uplo,
             &trans,
             &numberEigenValues,
             &localVectorSize,
             &alpha,
             &X[0],
             &numberEigenValues,
             &beta,
             &overlapMatrix[0],
             &numberEigenValues);
      computing_timer.leave_subsection("local overlap matrix for lowden");

      dealii::Utilities::MPI::sum(overlapMatrix, mpiComm, overlapMatrix);

      std::vector<double> eigenValuesOverlap(numberVectors);
      computing_timer.enter_subsection("eigen decomp. of overlap matrix");
      callevd(numberVectors, &overlapMatrix[0], &eigenValuesOverlap[0]);
      computing_timer.leave_subsection("eigen decomp. of overlap matrix");

      //
      // compute D^{-1/4} where S = Q*D*Q^{T}
      //
      std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues);
      dftfe::uInt         nanFlag = 0;
      for (dftfe::uInt i = 0; i < numberEigenValues; ++i)
        {
          invFourthRootEigenValuesMatrix[i] =
            1.0 / pow(eigenValuesOverlap[i], 1.0 / 4);
          if (std::isnan(invFourthRootEigenValuesMatrix[i]) ||
              eigenValuesOverlap[i] < 1e-10)
            {
              nanFlag = 1;
              break;
            }
        }

      nanFlag = dealii::Utilities::MPI::max(nanFlag, mpiComm);
      if (nanFlag == 1)
        return nanFlag;

      if (nanFlag == 1)
        {
          std::cout
            << "Nan obtained: switching to more robust dsyevr for eigen decomposition "
            << std::endl;
          std::vector<double> overlapMatrixEigenVectors(numberVectors *
                                                          numberVectors,
                                                        0.0);
          eigenValuesOverlap.clear();
          eigenValuesOverlap.resize(numberVectors);
          invFourthRootEigenValuesMatrix.clear();
          invFourthRootEigenValuesMatrix.resize(numberVectors);
          computing_timer.enter_subsection("eigen decomp. of overlap matrix");
          callevr(numberVectors,
                  &overlapMatrix[0],
                  &overlapMatrixEigenVectors[0],
                  &eigenValuesOverlap[0]);
          computing_timer.leave_subsection("eigen decomp. of overlap matrix");

          overlapMatrix = overlapMatrixEigenVectors;
          overlapMatrixEigenVectors.clear();
          std::vector<double>().swap(overlapMatrixEigenVectors);

          //
          // compute D^{-1/4} where S = Q*D*Q^{T}
          //
          for (dftfe::uInt i = 0; i < numberEigenValues; ++i)
            {
              invFourthRootEigenValuesMatrix[i] =
                1.0 / pow(eigenValuesOverlap[i], (1.0 / 4.0));
              AssertThrow(
                !std::isnan(invFourthRootEigenValuesMatrix[i]),
                dealii::ExcMessage(
                  "Eigen values of overlap matrix during Lowden Orthonormalization are close to zero."));
            }
        }

      //
      // Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling
      // "dsyevd"
      //
      computing_timer.enter_subsection("scaling in Lowden");
      const unsigned int inc = 1;
      for (dftfe::uInt i = 0; i < numberEigenValues; ++i)
        {
          double scalingCoeff = invFourthRootEigenValuesMatrix[i];
          dscal_(&numberEigenValues,
                 &scalingCoeff,
                 &overlapMatrix[0] + i * numberEigenValues,
                 &inc);
        }
      computing_timer.leave_subsection("scaling in Lowden");

      //
      // Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{T} = (Q*D^{-1/4})*(Q*D^{-1/4}))^{T}
      //
      std::vector<double> invSqrtOverlapMatrix(numberEigenValues *
                                                 numberEigenValues,
                                               0.0);
      const char          transA1 = 'N';
      const char          transB1 = 'T';
      computing_timer.enter_subsection("inverse sqrt overlap");
      dgemm_(&transA1,
             &transB1,
             &numberEigenValues,
             &numberEigenValues,
             &numberEigenValues,
             &alpha,
             &overlapMatrix[0],
             &numberEigenValues,
             &overlapMatrix[0],
             &numberEigenValues,
             &beta,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues);
      computing_timer.leave_subsection("inverse sqrt overlap");

      //
      // free up memory associated with overlapMatrix
      //
      overlapMatrix.clear();
      std::vector<double>().swap(overlapMatrix);

      //
      // Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but
      // implemented as Yt = S^{-1/2}*Xt using the column major format of blas
      //
      const char transA2 = 'N', transB2 = 'N';
      // dealii::parallel::distributed::Vector<double>
      // orthoNormalizedBasis; orthoNormalizedBasis.reinit(X);
      std::vector<double> orthoNormalizedBasis(X.size(), 0.0);

      computing_timer.enter_subsection("subspace rotation in lowden");
      dgemm_(&transA2,
             &transB2,
             &numberEigenValues,
             &localVectorSize,
             &numberEigenValues,
             &alpha,
             &invSqrtOverlapMatrix[0],
             &numberEigenValues,
             &X[0],
             &numberEigenValues,
             &beta,
             &orthoNormalizedBasis[0],
             &numberEigenValues);
      computing_timer.leave_subsection("subspace rotation in lowden");


      X = orthoNormalizedBasis;

      return 0;
    }
#endif



    template <typename T>
    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                &BLASWrapperPtr,
      T                         *X,
      const dftfe::uInt          N,
      const dftfe::uInt          numberLocalDofs,
      const MPI_Comm            &mpiCommParent,
      const MPI_Comm            &mpiCommDomain,
      const MPI_Comm            &interBandGroupComm,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double>       &densityMatDerFermiEnergy,
      dftfe::elpaScalaManager   &elpaScala,
      const dftParameters       &dftParams)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const dftfe::uInt rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      dftfe::ScaLAPACKMatrix<T> projHamPrimePar(N, processGrid, rowsBlockSize);


      if (processGrid->is_process_active())
        std::fill(&projHamPrimePar.local_el(0, 0),
                  &projHamPrimePar.local_el(0, 0) +
                    projHamPrimePar.local_m() * projHamPrimePar.local_n(),
                  T(0.0));

      //
      // compute projected Hamiltonian conjugate HConjProjPrime=
      // X^{T}*HConjPrime*XConj
      //
      computing_timer.enter_subsection("Compute ProjHamPrime, DMFOR step");
      if (dftParams.singlePrecLRD)
        {
          XtHXMixedPrec(operatorMatrix,
                        BLASWrapperPtr,
                        X,
                        N,
                        N,
                        numberLocalDofs,
                        processGrid,
                        mpiCommDomain,
                        interBandGroupComm,
                        dftParams,
                        projHamPrimePar,
                        true);
        }
      else
        XtHX(operatorMatrix,
             BLASWrapperPtr,
             X,
             N,
             numberLocalDofs,
             processGrid,
             mpiCommDomain,
             interBandGroupComm,
             dftParams,
             projHamPrimePar,
             true);
      computing_timer.leave_subsection("Compute ProjHamPrime, DMFOR step");


      computing_timer.enter_subsection(
        "Recursive fermi operator expansion operations, DMFOR step");

      const dftfe::Int m    = 10;
      const double     beta = 1.0 / C_kb / dftParams.TVal;
      const double     c    = std::pow(2.0, -2.0 - m) * beta;

      std::vector<double> H0 = eigenValues;
      std::vector<double> X0(N, 0.0);
      for (dftfe::uInt i = 0; i < N; ++i)
        {
          X0[i] = 0.5 - c * (H0[i] - fermiEnergy);
        }

      dftfe::ScaLAPACKMatrix<T> densityMatPrimePar(N,
                                                   processGrid,
                                                   rowsBlockSize);
      densityMatPrimePar.add(projHamPrimePar, T(0.0), T(-c)); //-c*HPrime

      dftfe::ScaLAPACKMatrix<T> X1Temp(N, processGrid, rowsBlockSize);
      dftfe::ScaLAPACKMatrix<T> X1Tempb(N, processGrid, rowsBlockSize);
      dftfe::ScaLAPACKMatrix<T> X1Tempc(N, processGrid, rowsBlockSize);

      std::vector<double> Y0Temp(N, 0.0);

      for (dftfe::uInt i = 0; i < m; ++i)
        {
          // step1
          X1Temp.add(densityMatPrimePar, T(0.0), T(1.0));  // copy
          X1Tempb.add(densityMatPrimePar, T(0.0), T(1.0)); // copy
          X1Temp.scale_rows_realfactors(X0);
          X1Tempb.scale_columns_realfactors(X0);
          X1Temp.add(X1Tempb, T(1.0), T(1.0));

          // step2 and 3
          for (dftfe::uInt j = 0; j < N; ++j)
            {
              Y0Temp[j] = 1.0 / (2.0 * X0[j] * (X0[j] - 1.0) + 1.0);
              X0[j]     = Y0Temp[j] * X0[j] * X0[j];
            }

          // step4
          X1Tempc.add(X1Temp, T(0.0), T(1.0)); // copy
          X1Temp.scale_rows_realfactors(Y0Temp);
          X1Tempc.scale_columns_realfactors(X0);
          X1Tempc.scale_rows_realfactors(Y0Temp);
          X1Temp.add(X1Tempc, T(1.0), T(-2.0));
          X1Tempb.add(densityMatPrimePar, T(0.0), T(1.0)); // copy
          X1Tempb.scale_columns_realfactors(X0);
          X1Tempb.scale_rows_realfactors(Y0Temp);
          densityMatPrimePar.add(X1Temp, T(0.0), T(1.0));
          densityMatPrimePar.add(X1Tempb, T(1.0), T(2.0));
        }

      std::vector<double> Pmu0(N, 0.0);
      double              sum = 0.0;
      for (dftfe::uInt i = 0; i < N; ++i)
        {
          Pmu0[i] = beta * X0[i] * (1.0 - X0[i]);
          sum += Pmu0[i];
        }

      densityMatDerFermiEnergy = Pmu0;

      computing_timer.leave_subsection(
        "Recursive fermi operator expansion operations, DMFOR step");

      //
      // subspace transformation Y^{T} = DMP^T*X^{T}, implemented as
      // Y^{T}=DMPc^{C}*X^{T} with X^{T} stored in the column major format
      //
      computing_timer.enter_subsection(
        "Blocked subspace transformation, DMFOR step");

      // For subspace transformation the full matrix is required
      dftfe::ScaLAPACKMatrix<T> densityMatPrimeParConjTrans(N,
                                                            processGrid,
                                                            rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&densityMatPrimeParConjTrans.local_el(0, 0),
                  &densityMatPrimeParConjTrans.local_el(0, 0) +
                    densityMatPrimeParConjTrans.local_m() *
                      densityMatPrimeParConjTrans.local_n(),
                  T(0.0));


      densityMatPrimeParConjTrans.copy_conjugate_transposed(densityMatPrimePar);
      densityMatPrimePar.add(densityMatPrimeParConjTrans, T(1.0), T(1.0));

      if (processGrid->is_process_active())
        for (dftfe::uInt i = 0; i < densityMatPrimePar.local_n(); ++i)
          {
            const dftfe::uInt glob_i = densityMatPrimePar.global_column(i);
            for (dftfe::uInt j = 0; j < densityMatPrimePar.local_m(); ++j)
              {
                const dftfe::uInt glob_j = densityMatPrimePar.global_row(j);
                if (glob_i == glob_j)
                  densityMatPrimePar.local_el(j, i) *= T(0.5);
              }
          }

      densityMatPrimeParConjTrans.copy_conjugate_transposed(densityMatPrimePar);

      if (dftParams.singlePrecLRD)
        {
          if (std::is_same<T, std::complex<double>>::value)
            internal::subspaceRotationMixedPrec<T, std::complex<float>>(
              X,
              BLASWrapperPtr,
              numberLocalDofs * N,
              N,
              processGrid,
              interBandGroupComm,
              mpiCommDomain,
              densityMatPrimeParConjTrans,
              dftParams,
              false,
              false);
          else
            internal::subspaceRotationMixedPrec<T, float>(
              X,
              BLASWrapperPtr,
              numberLocalDofs * N,
              N,
              processGrid,
              interBandGroupComm,
              mpiCommDomain,
              densityMatPrimeParConjTrans,
              dftParams,
              false,
              false);
        }
      else
        {
          internal::subspaceRotation(X,
                                     BLASWrapperPtr,
                                     numberLocalDofs * N,
                                     N,
                                     processGrid,
                                     interBandGroupComm,
                                     mpiCommDomain,
                                     densityMatPrimeParConjTrans,
                                     dftParams,
                                     false,
                                     false,
                                     false);
        }


      computing_timer.leave_subsection(
        "Blocked subspace transformation, DMFOR step");
    }


    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
         const std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                        &BLASWrapperPtr,
         const dataTypes::number        *X,
         const dftfe::uInt               numberWaveFunctions,
         const dftfe::uInt               numberDofs,
         const MPI_Comm                 &mpiCommDomain,
         const MPI_Comm                 &interBandGroupComm,
         const dftParameters            &dftParams,
         std::vector<dataTypes::number> &ProjHam)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      //
      // Resize ProjHam
      //
      ProjHam.clear();
      ProjHam.resize(numberWaveFunctions * numberWaveFunctions, 0.0);

      //
      // create temporary array XTemp
      //

      distributedCPUMultiVec<dataTypes::number> &XTemp =
        operatorMatrix.getScratchFEMultivector(numberWaveFunctions, 0);
      for (dftfe::uInt iNode = 0; iNode < numberDofs; ++iNode)
        for (dftfe::uInt iWave = 0; iWave < numberWaveFunctions; ++iWave)
          XTemp.data()[iNode * numberWaveFunctions + iWave] =
            X[iNode * numberWaveFunctions + iWave];

      //
      // create temporary array Y
      //
      distributedCPUMultiVec<dataTypes::number> &Y =
        operatorMatrix.getScratchFEMultivector(numberWaveFunctions, 1);

      //
      // evaluate H times XTemp and store in Y
      //
      operatorMatrix.HX(XTemp, 1.0, 0.0, 0.0, Y);
      const unsigned int numberWaveFunctionsTmp = numberWaveFunctions;
      const unsigned int numberDofsTmp          = numberDofs;
#ifdef USE_COMPLEX
      for (dftfe::uInt i = 0; i < Y.locallyOwnedSize(); ++i)
        Y.data()[i] = std::conj(Y.data()[i]);

      const std::complex<double> alpha = 1.0, beta = 0.0;
      BLASWrapperPtr->xgemm('N',
                            'T',
                            numberWaveFunctions,
                            numberWaveFunctions,
                            numberDofs,
                            &alpha,
                            Y.begin(),
                            numberWaveFunctions,
                            &X[0],
                            numberWaveFunctions,
                            &beta,
                            &ProjHam[0],
                            numberWaveFunctions);
#else
      char         transA = 'N';
      char         transB = 'T';
      const double alpha = 1.0, beta = 0.0;

      BLASWrapperPtr->xgemm('N',
                            'T',
                            numberWaveFunctions,
                            numberWaveFunctions,
                            numberDofs,
                            &alpha,
                            &X[0],
                            numberWaveFunctions,
                            Y.begin(),
                            numberWaveFunctions,
                            &beta,
                            &ProjHam[0],
                            numberWaveFunctions);
#endif
      dealii::Utilities::MPI::sum(ProjHam, mpiCommDomain, ProjHam);
    }

    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
         const std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                         &BLASWrapperPtr,
         const dataTypes::number                         *X,
         const dftfe::uInt                                numberWaveFunctions,
         const dftfe::uInt                                numberDofs,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         const MPI_Comm                                  &mpiCommDomain,
         const MPI_Comm                                  &interBandGroupComm,
         const dftParameters                             &dftParams,
         dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      // create temporary arrays XBlock,Hx
      distributedCPUMultiVec<dataTypes::number> *XBlock, *HXBlock;

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm,
        numberWaveFunctions,
        bandGroupLowHighPlusOneIndices);

      /*
       * X^{T}*Hc*Xc is done in a blocked approach for memory optimization:
       * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
       * has a much smaller memory compared to X^{T}*H*Xc.
       * X^{T} (denoted by X in the code with column major format storage)
       * is a matrix with size (N x MLoc).
       * N is denoted by numberWaveFunctions in the code.
       * MLoc, which is number of local dofs is denoted by numberDofs in the
       * code. Xc denotes complex conjugate of X. XcBlock is a matrix of size
       * (MLoc x B). B is the block size. A further optimization is done to
       * reduce floating point operations: As X^{T}*Hc*Xc is a Hermitian matrix,
       * it suffices to compute only the lower triangular part. To exploit this,
       * we do X^{T}*Hc*Xc=Sum_{blocks} XTrunc^{T}*H*XcBlock where XTrunc^{T} is
       * a (D x MLoc) sub matrix of X^{T} with the row indices ranging from the
       * lowest global index of XcBlock (denoted by jvec in the code) to N.
       * D=N-jvec. The parallel ScaLapack matrix projHamPar is directly filled
       * from the XTrunc^{T}*Hc*XcBlock result
       */

      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      std::vector<dataTypes::number> projHamBlock(numberWaveFunctions *
                                                    vectorsBlockSize,
                                                  dataTypes::number(0.0));

      if (dftParams.verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpiCommDomain,
          "Inside Blocked XtHX with parallel projected Ham matrix");

      for (dftfe::uInt jvec = 0; jvec < numberWaveFunctions;
           jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B =
            std::min(vectorsBlockSize, numberWaveFunctions - jvec);
          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              HXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // fill XBlock^{T} from X:
              for (dftfe::uInt iNode = 0; iNode < numberDofs; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * numberWaveFunctions + jvec + iWave];

              // evaluate H times XBlock and store in HXBlock^{T}
              operatorMatrix.HX(*XBlock,
                                1.0,
                                0.0,
                                0.0,
                                *HXBlock,
                                onlyHPrimePartForFirstOrderDensityMatResponse);
              MPI_Barrier(mpiCommDomain);

              const char transA = 'N';
              const char transB =
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T';

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              std::fill(projHamBlock.begin(),
                        projHamBlock.end(),
                        dataTypes::number(0.));

              const dftfe::uInt D = numberWaveFunctions - jvec;

              // Comptute local XTrunc^{T}*HXcBlock.
              BLASWrapperPtr->xgemm(transA,
                                    transB,
                                    D,
                                    B,
                                    numberDofs,
                                    &alpha,
                                    &X[0] + jvec,
                                    numberWaveFunctions,
                                    HXBlock->data(),
                                    B,
                                    &beta,
                                    &projHamBlock[0],
                                    D);

              MPI_Barrier(mpiCommDomain);
              // Sum local XTrunc^{T}*HXcBlock across domain decomposition
              // processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projHamBlock[0],
                            D * B,
                            dataTypes::mpi_type_id(&projHamBlock[0]),
                            MPI_SUM,
                            mpiCommDomain);
              // Copying only the lower triangular part to the ScaLAPACK
              // projected Hamiltonian matrix
              if (processGrid->is_process_active())
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = j + jvec; i < numberWaveFunctions;
                           ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projHamPar.local_el(it->second, localColumnId) =
                              projHamBlock[j * D + i - jvec];
                        }
                    }

            } // band parallelization

        } // block loop

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }

    void
    XtOX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
         const std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                         &BLASWrapperPtr,
         const dataTypes::number                         *X,
         const dftfe::uInt                                numberWaveFunctions,
         const dftfe::uInt                                numberDofs,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         const MPI_Comm                                  &mpiCommDomain,
         const MPI_Comm                                  &interBandGroupComm,
         const dftParameters                             &dftParams,
         dftfe::ScaLAPACKMatrix<dataTypes::number>       &projOverlapPar)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      // create temporary arrays XBlock,Hx
      distributedCPUMultiVec<dataTypes::number> *XBlock, *OXBlock;

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projOverlapPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm,
        numberWaveFunctions,
        bandGroupLowHighPlusOneIndices);

      /*
       * X^{T}*Hc*Xc is done in a blocked approach for memory optimization:
       * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
       * has a much smaller memory compared to X^{T}*H*Xc.
       * X^{T} (denoted by X in the code with column major format storage)
       * is a matrix with size (N x MLoc).
       * N is denoted by numberWaveFunctions in the code.
       * MLoc, which is number of local dofs is denoted by numberDofs in the
       * code. Xc denotes complex conjugate of X. XcBlock is a matrix of size
       * (MLoc x B). B is the block size. A further optimization is done to
       * reduce floating point operations: As X^{T}*Hc*Xc is a Hermitian matrix,
       * it suffices to compute only the lower triangular part. To exploit this,
       * we do X^{T}*Hc*Xc=Sum_{blocks} XTrunc^{T}*H*XcBlock where XTrunc^{T} is
       * a (D x MLoc) sub matrix of X^{T} with the row indices ranging from the
       * lowest global index of XcBlock (denoted by jvec in the code) to N.
       * D=N-jvec. The parallel ScaLapack matrix projHamPar is directly filled
       * from the XTrunc^{T}*Hc*XcBlock result
       */

      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      std::vector<dataTypes::number> projOverlapBlock(numberWaveFunctions *
                                                        vectorsBlockSize,
                                                      dataTypes::number(0.0));

      if (dftParams.verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpiCommDomain,
          "Inside Blocked XtOX with parallel projected Overlap matrix");
      for (dftfe::uInt jvec = 0; jvec < numberWaveFunctions;
           jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B =
            std::min(vectorsBlockSize, numberWaveFunctions - jvec);
          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              OXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // fill XBlock^{T} from X:
              for (dftfe::uInt iNode = 0; iNode < numberDofs; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * numberWaveFunctions + jvec + iWave];

              // evaluate H times XBlock and store in HXBlock^{T}
              operatorMatrix.overlapMatrixTimesX(*XBlock,
                                                 1.0,
                                                 0.0,
                                                 0.0,
                                                 *OXBlock,
                                                 dftParams.approxOverlapMatrix);
              MPI_Barrier(mpiCommDomain);

              const char transA = 'N';
              const char transB =
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T';

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              std::fill(projOverlapBlock.begin(),
                        projOverlapBlock.end(),
                        dataTypes::number(0.));

              const dftfe::uInt D = numberWaveFunctions - jvec;

              // Comptute local XTrunc^{T}*HXcBlock.
              BLASWrapperPtr->xgemm(transA,
                                    transB,
                                    D,
                                    B,
                                    numberDofs,
                                    &alpha,
                                    &X[0] + jvec,
                                    numberWaveFunctions,
                                    OXBlock->data(),
                                    B,
                                    &beta,
                                    &projOverlapBlock[0],
                                    D);

              MPI_Barrier(mpiCommDomain);
              // Sum local XTrunc^{T}*HXcBlock across domain decomposition
              // processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projOverlapBlock[0],
                            D * B,
                            dataTypes::mpi_type_id(&projOverlapBlock[0]),
                            MPI_SUM,
                            mpiCommDomain);
              // Copying only the lower triangular part to the ScaLAPACK
              // projected Hamiltonian matrix
              if (processGrid->is_process_active())
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = j + jvec; i < numberWaveFunctions;
                           ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projOverlapPar.local_el(it->second, localColumnId) =
                              projOverlapBlock[j * D + i - jvec];
                        }
                    }

            } // band parallelization

        } // block loop

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projOverlapPar, interBandGroupComm);
        }
    }


    void
    XtHXMixedPrec(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                      &BLASWrapperPtr,
      const dataTypes::number                         *X,
      const dftfe::uInt                                N,
      const dftfe::uInt                                Ncore,
      const dftfe::uInt                                numberDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftParameters                             &dftParams,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projHamPar,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      // create temporary arrays XBlock,Hx
      distributedCPUMultiVec<dataTypes::number> *XBlock, *HXBlock;

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projHamPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      /*
       * X^{T}*H*Xc is done in a blocked approach for memory optimization:
       * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
       * has a much smaller memory compared to X^{T}*Hc*Xc.
       * X^{T} (denoted by X in the code with column major format storage)
       * is a matrix with size (N x MLoc).
       * MLoc, which is number of local dofs is denoted by numberDofs in the
       * code. Xc denotes complex conjugate of X. XcBlock is a matrix of size
       * (MLoc x B). B is the block size. A further optimization is done to
       * reduce floating point operations: As X^{T}*Hc*Xc is a Hermitian
       matrix,
       * it suffices to compute only the lower triangular part. To exploit
       this,
       * we do X^{T}*Hc*Xc=Sum_{blocks} XTrunc^{T}*Hc*XcBlock where
       XTrunc^{T}
       * is a (D x MLoc) sub matrix of X^{T} with the row indices ranging
       from
       * the lowest global index of XcBlock (denoted by jvec in the code) to
       N.
       * D=N-jvec. The parallel ScaLapack matrix projHamPar is directly
       filled
       * from the XTrunc^{T}*Hc*XcBlock result
       */

      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      std::vector<dataTypes::numberFP32> projHamBlockSinglePrec(
        N * vectorsBlockSize, 0.0);
      std::vector<dataTypes::number> projHamBlock(N * vectorsBlockSize, 0.0);
      std::vector<dataTypes::number> projHamBlockDoublePrec(vectorsBlockSize *
                                                              vectorsBlockSize,
                                                            0.0);
      std::vector<dataTypes::numberFP32> HXBlockSinglePrec;

      std::vector<dataTypes::numberFP32> XSinglePrec(X, X + numberDofs * N);

      if (dftParams.verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpiCommDomain,
          "Inside Blocked XtHX with parallel projected Ham matrix");

      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);
          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              HXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
              HXBlockSinglePrec.resize(B * numberDofs);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // fill XBlock^{T} from X:
              for (dftfe::uInt iNode = 0; iNode < numberDofs; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * N + jvec + iWave];

              // evaluate H times XBlock and store in HXBlock^{T}
              operatorMatrix.HX(*XBlock,
                                1.0,
                                0.0,
                                0.0,
                                *HXBlock,
                                onlyHPrimePartForFirstOrderDensityMatResponse);


              MPI_Barrier(mpiCommDomain);

              const char transA = 'N';
              const char transB =
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T';
              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              std::fill(projHamBlock.begin(),
                        projHamBlock.end(),
                        dataTypes::number(0.));

              if (jvec + B > Ncore)
                {
                  const dftfe::uInt D = N - jvec;

                  // Comptute local XTrunc^{T}*HXcBlock.
                  BLASWrapperPtr->xgemm(transA,
                                        transB,
                                        D,
                                        B,
                                        numberDofs,
                                        &alpha,
                                        &X[0] + jvec,
                                        N,
                                        HXBlock->data(),
                                        B,
                                        &beta,
                                        &projHamBlock[0],
                                        D);

                  MPI_Barrier(mpiCommDomain);
                  // Sum local XTrunc^{T}*HXcBlock across domain decomposition
                  // processors
                  MPI_Allreduce(MPI_IN_PLACE,
                                &projHamBlock[0],
                                D * B,
                                dataTypes::mpi_type_id(&projHamBlock[0]),
                                MPI_SUM,
                                mpiCommDomain);


                  // Copying only the lower triangular part to the ScaLAPACK
                  // projected Hamiltonian matrix
                  if (processGrid->is_process_active())
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + jvec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + jvec];
                          for (dftfe::uInt i = jvec + j; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                projHamPar.local_el(it->second, localColumnId) =
                                  projHamBlock[j * D + i - jvec];
                            }
                        }
                }
              else
                {
                  const dataTypes::numberFP32 alphaSinglePrec =
                                                dataTypes::numberFP32(1.0),
                                              betaSinglePrec =
                                                dataTypes::numberFP32(0.0);



                  const dftfe::uInt D = N - jvec;
                  // full prec gemm
                  BLASWrapperPtr->xgemm(transA,
                                        transB,
                                        B,
                                        B,
                                        numberDofs,
                                        &alpha,
                                        &X[0] + jvec,
                                        N,
                                        HXBlock->data(),
                                        B,
                                        &beta,
                                        &projHamBlockDoublePrec[0],
                                        B);
                  const dftfe::uInt DRem = D - B;
                  // single prec gemm
                  if (DRem != 0)
                    {
                      for (dftfe::uInt i = 0; i < numberDofs * B; ++i)
                        HXBlockSinglePrec[i] = HXBlock->data()[i];
                      BLASWrapperPtr->xgemm(transA,
                                            transB,
                                            DRem,
                                            B,
                                            numberDofs,
                                            &alphaSinglePrec,
                                            &XSinglePrec[0] + jvec + B,
                                            N,
                                            &HXBlockSinglePrec[0],
                                            B,
                                            &betaSinglePrec,
                                            &projHamBlockSinglePrec[0],
                                            DRem);
                    }

                  MPI_Barrier(mpiCommDomain);
                  MPI_Allreduce(MPI_IN_PLACE,
                                &projHamBlockDoublePrec[0],
                                B * B,
                                dataTypes::mpi_type_id(
                                  &projHamBlockDoublePrec[0]),
                                MPI_SUM,
                                mpiCommDomain);
                  MPI_Allreduce(MPI_IN_PLACE,
                                &projHamBlockSinglePrec[0],
                                DRem * B,
                                dataTypes::mpi_type_id(
                                  &projHamBlockSinglePrec[0]),
                                MPI_SUM,
                                mpiCommDomain);


                  for (dftfe::uInt i = 0; i < B; ++i)
                    {
                      for (dftfe::uInt j = 0; j < B; ++j)
                        projHamBlock[i * D + j] =
                          projHamBlockDoublePrec[i * B + j];

                      for (dftfe::uInt j = 0; j < DRem; ++j)
                        projHamBlock[i * D + j + B] =
                          projHamBlockSinglePrec[i * DRem + j];
                    }

                  if (processGrid->is_process_active())
                    for (dftfe::uInt j = 0; j < B; ++j)
                      if (globalToLocalColumnIdMap.find(j + jvec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const dftfe::uInt localColumnId =
                            globalToLocalColumnIdMap[j + jvec];
                          for (dftfe::uInt i = jvec + j; i < N; ++i)
                            {
                              std::unordered_map<dftfe::uInt,
                                                 dftfe::uInt>::iterator it =
                                globalToLocalRowIdMap.find(i);
                              if (it != globalToLocalRowIdMap.end())
                                projHamPar.local_el(it->second, localColumnId) =
                                  projHamBlock[j * D + i - jvec];
                            }
                        }
                }


            } // band parallelization

        } // block loop

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }

    void
    XtOXMixedPrec(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                      &BLASWrapperPtr,
      const dataTypes::number                         *X,
      const dftfe::uInt                                N,
      const dftfe::uInt                                Ncore,
      const dftfe::uInt                                numberDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      const MPI_Comm                                  &interBandGroupComm,
      const dftParameters                             &dftParams,
      dftfe::ScaLAPACKMatrix<dataTypes::number>       &projOverlapPar)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      // create temporary arrays XBlock,Hx
      distributedCPUMultiVec<dataTypes::number> *XBlock, *OXBlock;

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projOverlapPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      /*
       * X^{T}*H*Xc is done in a blocked approach for memory optimization:
       * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
       * has a much smaller memory compared to X^{T}*Hc*Xc.
       * X^{T} (denoted by X in the code with column major format storage)
       * is a matrix with size (N x MLoc).
       * MLoc, which is number of local dofs is denoted by numberDofs in the
       * code. Xc denotes complex conjugate of X. XcBlock is a matrix of size
       * (MLoc x B). B is the block size. A further optimization is done to
       * reduce floating point operations: As X^{T}*Hc*Xc is a Hermitian
       matrix,
       * it suffices to compute only the lower triangular part. To exploit
       this,
       * we do X^{T}*Hc*Xc=Sum_{blocks} XTrunc^{T}*Hc*XcBlock where
       XTrunc^{T}
       * is a (D x MLoc) sub matrix of X^{T} with the row indices ranging
       from
       * the lowest global index of XcBlock (denoted by jvec in the code) to
       N.
       * D=N-jvec. The parallel ScaLapack matrix projHamPar is directly
       filled
       * from the XTrunc^{T}*Hc*XcBlock result
       */

      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      std::vector<dataTypes::numberFP32> projOverlapBlockSinglePrec(
        N * vectorsBlockSize, 0.0);
      std::vector<dataTypes::number> projOverlapBlockDoublePrec(
        vectorsBlockSize * vectorsBlockSize, 0.0);
      std::vector<dataTypes::number> projOverlapBlock(N * vectorsBlockSize,
                                                      0.0);

      std::vector<dataTypes::numberFP32> OXBlockSinglePrec;

      std::vector<dataTypes::numberFP32> XSinglePrec(X, X + numberDofs * N);
      const char                         transA = 'N';
      const char                         transB =
        std::is_same<dataTypes::number, std::complex<double>>::value ? 'C' :
                                                                                               'T';
      if (dftParams.verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpiCommDomain,
          "Inside Blocked XtOX with parallel projected Overlap matrix");

      for (dftfe::uInt jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B = std::min(vectorsBlockSize, N - jvec);
          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              OXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
              OXBlockSinglePrec.resize(B * numberDofs);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // fill XBlock^{T} from X:
              for (dftfe::uInt iNode = 0; iNode < numberDofs; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * N + jvec + iWave];

              operatorMatrix.overlapMatrixTimesX(*XBlock,
                                                 1.0,
                                                 0.0,
                                                 0.0,
                                                 *OXBlock,
                                                 dftParams.approxOverlapMatrix);



              const dftfe::uInt       D     = N - jvec;
              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              BLASWrapperPtr->xgemm(transA,
                                    transB,
                                    B,
                                    B,
                                    numberDofs,
                                    &alpha,
                                    &X[0] + jvec,
                                    N,
                                    OXBlock->data(),
                                    B,
                                    &beta,
                                    &projOverlapBlockDoublePrec[0],
                                    B);
              const dftfe::uInt DRem = D - B;
              if (DRem != 0)
                {
                  const dataTypes::numberFP32 alphaSinglePrec =
                                                dataTypes::numberFP32(1.0),
                                              betaSinglePrec =
                                                dataTypes::numberFP32(0.0);
                  for (dftfe::uInt i = 0; i < numberDofs * B; ++i)
                    OXBlockSinglePrec[i] = OXBlock->data()[i];
                  BLASWrapperPtr->xgemm(transA,
                                        transB,
                                        DRem,
                                        B,
                                        numberDofs,
                                        &alphaSinglePrec,
                                        &XSinglePrec[0] + jvec + B,
                                        N,
                                        &OXBlockSinglePrec[0],
                                        B,
                                        &betaSinglePrec,
                                        &projOverlapBlockSinglePrec[0],
                                        DRem);
                }

              // Sum local XTrunc^{T}*XcBlock for double precision across
              // domain decomposition processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projOverlapBlockDoublePrec[0],
                            B * B,
                            dataTypes::mpi_type_id(
                              &projOverlapBlockDoublePrec[0]),
                            MPI_SUM,
                            mpiCommDomain);

              // Sum local XTrunc^{T}*XcBlock for single precision across
              // domain decomposition processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projOverlapBlockSinglePrec[0],
                            DRem * B,
                            dataTypes::mpi_type_id(
                              &projOverlapBlockSinglePrec[0]),
                            MPI_SUM,
                            mpiCommDomain);

              for (dftfe::uInt i = 0; i < B; ++i)
                {
                  for (dftfe::uInt j = 0; j < B; ++j)
                    projOverlapBlock[i * D + j] =
                      projOverlapBlockDoublePrec[i * B + j];

                  for (dftfe::uInt j = 0; j < DRem; ++j)
                    projOverlapBlock[i * D + j + B] =
                      projOverlapBlockSinglePrec[i * DRem + j];
                }

              // Copying only the lower triangular part to the ScaLAPACK
              // overlap matrix
              if (processGrid->is_process_active())
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = jvec + j; i < N; ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projOverlapPar.local_el(it->second, localColumnId) =
                              projOverlapBlock[j * D + i - jvec];
                        }
                    }

            } // band parallelization

        } // block loop

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projOverlapPar, interBandGroupComm);
        }
    }

    void
    XtHXXtOX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
             const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
               dftfe::utils::MemorySpace::HOST>>               &BLASWrapperPtr,
             const dataTypes::number                           *X,
             const dftfe::uInt                                numberComponents,
             const dftfe::uInt                                numberLocalDofs,
             const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
             const MPI_Comm                                  &mpiCommDomain,
             const MPI_Comm                            &interBandGroupComm,
             const dftParameters                       &dftParams,
             dftfe::ScaLAPACKMatrix<dataTypes::number> &projHamPar,
             dftfe::ScaLAPACKMatrix<dataTypes::number> &projOverlapPar,
             const bool onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      //
      // Get access to number of locally owned nodes on the current processor
      //

      // create temporary arrays XBlock,Hx
      distributedCPUMultiVec<dataTypes::number> *XBlock, *OXBlock;

      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
      std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        projOverlapPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);
      // band group parallelization data structures
      const dftfe::uInt numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const dftfe::uInt bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, numberComponents, bandGroupLowHighPlusOneIndices);
      /*
       * X^{T}*Hc*Xc is done in a blocked approach for memory optimization:
       * Sum_{blocks} X^{T}*Hc*XcBlock. The result of each X^{T}*Hc*XcBlock
       * has a much smaller memory compared to X^{T}*H*Xc.
       * X^{T} (denoted by X in the code with column major format storage)
       * is a matrix with size (N x MLoc).
       * N is denoted by numberWaveFunctions in the code.
       * MLoc, which is number of local dofs is denoted by numberDofs in the
       * code. Xc denotes complex conjugate of X. XcBlock is a matrix of size
       * (MLoc x B). B is the block size. A further optimization is done to
       * reduce floating point operations: As X^{T}*Hc*Xc is a Hermitian matrix,
       * it suffices to compute only the lower triangular part. To exploit this,
       * we do X^{T}*Hc*Xc=Sum_{blocks} XTrunc^{T}*H*XcBlock where XTrunc^{T} is
       * a (D x MLoc) sub matrix of X^{T} with the row indices ranging from the
       * lowest global index of XcBlock (denoted by jvec in the code) to N.
       * D=N-jvec. The parallel ScaLapack matrix projOverlapPar is directly
       * filled from the XTrunc^{T}*Hc*XcBlock result
       */

      const dftfe::uInt vectorsBlockSize =
        std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

      std::vector<dataTypes::number> projBlock(numberComponents *
                                                 vectorsBlockSize,
                                               dataTypes::number(0.0));

      if (dftParams.verbosity >= 4)
        dftUtils::printCurrentMemoryUsage(
          mpiCommDomain,
          "Inside Blocked XtOX with parallel projected Overlap matrix");

      for (dftfe::uInt jvec = 0; jvec < numberComponents;
           jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const dftfe::uInt B =
            std::min(vectorsBlockSize, numberComponents - jvec);
          if (jvec == 0 || B != vectorsBlockSize)
            {
              XBlock  = &operatorMatrix.getScratchFEMultivector(B, 0);
              OXBlock = &operatorMatrix.getScratchFEMultivector(B, 1);
            }

          if ((jvec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // fill XBlock^{T} from X:
              for (dftfe::uInt iNode = 0; iNode < numberLocalDofs; ++iNode)
                for (dftfe::uInt iWave = 0; iWave < B; ++iWave)
                  XBlock->data()[iNode * B + iWave] =
                    X[iNode * numberComponents + jvec + iWave];
              // XtOX operations

              operatorMatrix.overlapMatrixTimesX(*XBlock,
                                                 1.0,
                                                 0.0,
                                                 0.0,
                                                 *OXBlock,
                                                 dftParams.approxOverlapMatrix);
              MPI_Barrier(mpiCommDomain);
              const char transA = 'N';
              const char transB =
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  'C' :
                  'T';

              const dataTypes::number alpha = dataTypes::number(1.0),
                                      beta  = dataTypes::number(0.0);
              std::fill(projBlock.begin(),
                        projBlock.end(),
                        dataTypes::number(0.));

              const dftfe::uInt D = numberComponents - jvec;

              // Comptute local XTrunc^{T}*HXcBlock.
              BLASWrapperPtr->xgemm(transA,
                                    transB,
                                    D,
                                    B,
                                    numberLocalDofs,
                                    &alpha,
                                    &X[0] + jvec,
                                    numberComponents,
                                    OXBlock->data(),
                                    B,
                                    &beta,
                                    &projBlock[0],
                                    D);

              MPI_Barrier(mpiCommDomain);
              // Sum local XTrunc^{T}*HXcBlock across domain decomposition
              // processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projBlock[0],
                            D * B,
                            dataTypes::mpi_type_id(&projBlock[0]),
                            MPI_SUM,
                            mpiCommDomain);
              // Copying only the lower triangular part to the ScaLAPACK
              // projected Hamiltonian matrix
              if (processGrid->is_process_active())
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = j + jvec; i < numberComponents; ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projOverlapPar.local_el(it->second, localColumnId) =
                              projBlock[j * D + i - jvec];
                        }
                    }
              // XtHX operations
              operatorMatrix.HX(*XBlock,
                                1.0,
                                0.0,
                                0.0,
                                *OXBlock,
                                onlyHPrimePartForFirstOrderDensityMatResponse);
              MPI_Barrier(mpiCommDomain);

              std::fill(projBlock.begin(),
                        projBlock.end(),
                        dataTypes::number(0.));


              // Comptute local XTrunc^{T}*HXcBlock.
              BLASWrapperPtr->xgemm(transA,
                                    transB,
                                    D,
                                    B,
                                    numberLocalDofs,
                                    &alpha,
                                    &X[0] + jvec,
                                    numberComponents,
                                    OXBlock->data(),
                                    B,
                                    &beta,
                                    &projBlock[0],
                                    D);

              MPI_Barrier(mpiCommDomain);
              // Sum local XTrunc^{T}*HXcBlock across domain decomposition
              // processors
              MPI_Allreduce(MPI_IN_PLACE,
                            &projBlock[0],
                            D * B,
                            dataTypes::mpi_type_id(&projBlock[0]),
                            MPI_SUM,
                            mpiCommDomain);
              // Copying only the lower triangular part to the ScaLAPACK
              // projected Hamiltonian matrix
              if (processGrid->is_process_active())
                for (dftfe::uInt j = 0; j < B; ++j)
                  if (globalToLocalColumnIdMap.find(j + jvec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const dftfe::uInt localColumnId =
                        globalToLocalColumnIdMap[j + jvec];
                      for (dftfe::uInt i = j + jvec; i < numberComponents; ++i)
                        {
                          std::unordered_map<dftfe::uInt, dftfe::uInt>::iterator
                            it = globalToLocalRowIdMap.find(i);
                          if (it != globalToLocalRowIdMap.end())
                            projHamPar.local_el(it->second, localColumnId) =
                              projBlock[j * D + i - jvec];
                        }
                    }


            } // band parallelization

        } // block loop

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projOverlapPar, interBandGroupComm);
          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, projHamPar, interBandGroupComm);
        }
    }



    template void
    gramSchmidtOrthogonalization(dataTypes::number *,
                                 const unsigned int,
                                 const unsigned int localVectorSize,
                                 const MPI_Comm &);

    template dftfe::uInt
    pseudoGramSchmidtOrthogonalization(
      elpaScalaManager &elpaScala,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
      dataTypes::number *,
      const dftfe::uInt,
      const dftfe::uInt localVectorSize,
      const MPI_Comm &,
      const MPI_Comm &,
      const MPI_Comm      &mpiComm,
      const bool           useMixedPrec,
      const dftParameters &dftParams);

    template void
    rayleighRitz(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                       &BLASWrapperPtr,
      elpaScalaManager &elpaScala,
      dataTypes::number *,
      const dftfe::uInt numberWaveFunctions,
      const dftfe::uInt localVectorSize,
      const MPI_Comm &,
      const MPI_Comm &,
      const MPI_Comm &,
      std::vector<double> &eigenValues,
      const dftParameters &dftParams,
      const bool           doCommAfterBandParal);

    template void
    rayleighRitzGEP(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                       &BLASWrapperPtr,
      elpaScalaManager &elpaScala,
      dataTypes::number *,
      const dftfe::uInt numberWaveFunctions,
      const dftfe::uInt localVectorSize,
      const MPI_Comm &,
      const MPI_Comm &,
      const MPI_Comm &,
      std::vector<double> &eigenValues,
      const bool           useMixedPrec,
      const dftParameters &dftParams);



    template void
    computeEigenResidualNorm(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                &BLASWrapperPtr,
      dataTypes::number         *X,
      const std::vector<double> &eigenValues,
      const dftfe::uInt          totalNumberVectors,
      const dftfe::uInt          localVectorSize,
      const MPI_Comm            &mpiCommParent,
      const MPI_Comm            &mpiCommDomain,
      const MPI_Comm            &interBandGroupComm,
      std::vector<double>       &residualNorm,
      const dftParameters       &dftParams);

    template void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                &BLASWrapperPtr,
      dataTypes::number         *X,
      const dftfe::uInt          N,
      const dftfe::uInt          numberLocalDofs,
      const MPI_Comm            &mpiCommParent,
      const MPI_Comm            &mpiCommDomain,
      const MPI_Comm            &interBandGroupComm,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double>       &densityMatDerFermiEnergy,
      elpaScalaManager          &elpaScala,
      const dftParameters       &dftParams);

  } // namespace linearAlgebraOperations

} // namespace dftfe
