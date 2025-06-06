// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef linearAlgebraOperationsInternal_h
#define linearAlgebraOperationsInternal_h

#include <headers.h>
#include <operator.h>
#include "process_grid.h"
#include "scalapackWrapper.h"
#include "dftParameters.h"
#include <BLASWrapper.h>
#include <elpa/elpa.h>
#include <unordered_map>
#include <dftUtils.h>
namespace dftfe
{
  namespace linearAlgebraOperations
  {
    /**
     *  @brief Contains internal functions used in linearAlgebraOperations
     *
     *  @author Sambit Das
     */
    namespace internal
    {
      /** @brief setup ELPA parameters.
       *
       */
      void
      setupELPAHandleParameters(
        const MPI_Comm &mpi_communicator,
        MPI_Comm       &processGridCommunicatorActive,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::uInt                                na,
        const dftfe::uInt                                nev,
        const dftfe::uInt                                blockSize,
        elpa_t                                          &elpaHandle,
        const dftParameters                             &dftParams);

      /** @brief Wrapper function to create a two dimensional processor grid for a square matrix in
       * dftfe::ScaLAPACKMatrix storage format.
       *
       */
      void
      createProcessGridSquareMatrix(
        const MPI_Comm                            &mpi_communicator,
        const dftfe::uInt                          size,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftParameters                       &dftParams,
        const bool                                 useOnlyThumbRule = false);

      /** @brief Wrapper function to create a two dimensional processor grid for a rectangular matrix in
       * dftfe::ScaLAPACKMatrix storage format.
       *
       */
      void
      createProcessGridRectangularMatrix(
        const MPI_Comm                            &mpi_communicator,
        const dftfe::uInt                          sizeRows,
        const dftfe::uInt                          sizeColumns,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftParameters                       &dftParams);


      /** @brief Creates global row/column id to local row/column ids for dftfe::ScaLAPACKMatrix
       *
       */
      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::ScaLAPACKMatrix<T>                 &mat,
        std::unordered_map<dftfe::uInt, dftfe::uInt>    &globalToLocalRowIdMap,
        std::unordered_map<dftfe::uInt, dftfe::uInt> &globalToLocalColumnIdMap);


      /** @brief Mpi all reduce of ScaLAPACKMat across a given inter communicator.
       * Used for band parallelization.
       *
       */
      template <typename T>
      void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T>                       &mat,
        const MPI_Comm                                  &interComm);



      /** @brief scale a ScaLAPACKMat with a scalar
       *
       *
       */
      template <typename T>
      void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                  &BLASWrapperPtr,
        dftfe::ScaLAPACKMatrix<T> &mat,
        const T                    scalar);


      /** @brief MPI_Bcast of ScaLAPACKMat across a given inter communicator from a given broadcast root.
       * Used for band parallelization.
       *
       */
      template <typename T>
      void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T>                       &mat,
        const MPI_Comm                                  &interComm,
        const dftfe::uInt                                broadcastRoot);

      /** @brief Computes Sc=X^{T}*Xc and stores in a parallel ScaLAPACK matrix.
       * X^{T} is the subspaceVectorsArray stored in the column major format (N
       * x M). Sc is the overlapMatPar.
       *
       * The overlap matrix computation and filling is done in a blocked
       * approach which avoids creation of full serial overlap matrix memory,
       * and also avoids creation of another full X memory.
       *
       */
      template <typename T>
      void
      fillParallelOverlapMatrix(
        const T *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
        const dftfe::uInt                                XLocalSize,
        const dftfe::uInt                                numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<T>                       &overlapMatPar,
        const dftParameters                             &dftParams);


      /** @brief Computes Sc=X^{T}*Xc and stores in a parallel ScaLAPACK matrix.
       * X^{T} is the subspaceVectorsArray stored in the column major format (N
       * x M). Sc is the overlapMatPar.
       *
       * The overlap matrix computation and filling is done in a blocked
       * approach which avoids creation of full serial overlap matrix memory,
       * and also avoids creation of another full X memory.
       *
       */
      template <typename T, typename TLowPrec>
      void
      fillParallelOverlapMatrixMixedPrec(
        const T *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
        const dftfe::uInt                                XLocalSize,
        const dftfe::uInt                                numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<T>                       &overlapMatPar,
        const dftParameters                             &dftParams);


      /** @brief Computes X^{T}=Q*X^{T} inplace. X^{T} is the subspaceVectorsArray
       * stored in the column major format (N x M). Q is rotationMatPar (N x N).
       *
       * The subspace rotation inside this function is done in a blocked
       * approach which avoids creation of full serial rotation matrix memory,
       * and also avoids creation of another full subspaceVectorsArray memory.
       * subspaceVectorsArrayLocalSize=N*M
       *
       */
      template <typename T>
      void
      subspaceRotation(
        T *subspaceVectorsArray,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        const dftfe::ScaLAPACKMatrix<T>                 &rotationMatPar,
        const dftParameters                             &dftParams,
        const bool rotationMatTranspose   = false,
        const bool isRotationMatLowerTria = false,
        const bool doCommAfterBandParal   = true);

      /** @brief Computes X^{T}=Q*X^{T} inplace. X^{T} is the subspaceVectorsArray
       * stored in the column major format (N x M). Q is rotationMatPar (N x N).
       *
       * The subspace rotation inside this function is done in a blocked
       * approach which avoids creation of full serial rotation matrix memory,
       * and also avoids creation of another full subspaceVectorsArray memory.
       * subspaceVectorsArrayLocalSize=N*M
       *
       */
      template <typename T, typename TLowPrec>
      void
      subspaceRotationMixedPrec(
        T *subspaceVectorsArray,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        const dftfe::ScaLAPACKMatrix<T>                 &rotationMatPar,
        const dftParameters                             &dftParams,
        const bool rotationMatTranspose = false,
        const bool doCommAfterBandParal = true);


      /** @brief Computes Y^{T}=Q*X^{T}.
       *
       * X^{T} is stored in the column major format (N x M). Q is extracted from
       * the supplied QMat as Q=QMat{1:numberTopVectors}. If QMat is in column
       * major format set QMatTranspose=false, otherwise set to true if in row
       * major format. The dimensions (in row major format) of QMat could be
       * either a) (N x numberTopVectors) or b) (N x N) where
       * numberTopVectors!=N. In this case it is assumed that Q is stored in the
       * first numberTopVectors columns of QMat. The subspace rotation inside
       * this function is done in a blocked approach which avoids creation of
       * full serial rotation matrix memory, and also avoids creation of another
       * full X memory. subspaceVectorsArrayLocalSize=N*M
       *
       */
      template <typename T>
      void
      subspaceRotationSpectrumSplit(
        const T *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        T                *Y,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::uInt                                numberTopVectors,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        const dftfe::ScaLAPACKMatrix<T>                 &QMat,
        const dftParameters                             &dftParams,
        const bool                                       QMatTranspose = false);

      /** @brief Computes Y^{T}=Q*X^{T}.
       *
       * X^{T} is stored in the column major format (N x M). Q is extracted from
       * the supplied QMat as Q=QMat{1:numberTopVectors}. If QMat is in column
       * major format set QMatTranspose=false, otherwise set to true if in row
       * major format. The dimensions (in row major format) of QMat could be
       * either a) (N x numberTopVectors) or b) (N x N) where
       * numberTopVectors!=N. In this case it is assumed that Q is stored in the
       * first numberTopVectors columns of QMat. The subspace rotation inside
       * this function is done in a blocked approach which avoids creation of
       * full serial rotation matrix memory, and also avoids creation of another
       * full X memory. subspaceVectorsArrayLocalSize=N*M
       *
       */
      template <typename T, typename TLowPrec>
      void
      subspaceRotationSpectrumSplitMixedPrec(
        const T *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        T                *Y,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::uInt                                numberTopVectors,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        const dftfe::ScaLAPACKMatrix<T>                 &QMat,
        const dftParameters                             &dftParams,
        const bool                                       QMatTranspose = false);


      /** @brief Computes X^{T}=Q*X^{T} inplace. X^{T} is the subspaceVectorsArray
       * stored in the column major format (N x M). Q is rotationMatPar (N x N).
       *
       * The subspace rotation inside this function is done in a blocked
       * approach which avoids creation of full serial rotation matrix memory,
       * and also avoids creation of another full subspaceVectorsArray memory.
       * subspaceVectorsArrayLocalSize=N*M
       *
       */
      template <typename T, typename TLowPrec>
      void
      subspaceRotationCGSMixedPrec(
        T *subspaceVectorsArray,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const MPI_Comm                                  &interBandGroupComm,
        const MPI_Comm                                  &mpiComm,
        const dftfe::ScaLAPACKMatrix<T>                 &rotationMatPar,
        const dftParameters                             &dftParams,
        const bool rotationMatTranspose = false,
        const bool doCommAfterBandParal = true);
    } // namespace internal
  }   // namespace linearAlgebraOperations
} // namespace dftfe
#endif
