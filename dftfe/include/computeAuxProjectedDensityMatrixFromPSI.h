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

#ifndef computeAuxProjectedDensityMatrixFromPSI_H_
#define computeAuxProjectedDensityMatrixFromPSI_H_

#include <headers.h>
#include <dftParameters.h>
#include <FEBasisOperations.h>
#include <AuxDensityMatrix.h>

namespace dftfe
{
  template <typename NumberType, dftfe::utils::MemorySpace memorySpace>
  void
  computeAuxProjectedDensityMatrixFromPSI(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> &X,
    const dftfe::uInt                       totalNumWaveFunctions,
    const std::vector<std::vector<double>> &partialOccupancies,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &basisOperationsPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                  &BLASWrapperPtr,
    const dftfe::uInt              matrixFreeDofhandlerIndex,
    const dftfe::uInt              quadratureIndex,
    const std::vector<double>     &kPointWeights,
    AuxDensityMatrix<memorySpace> &auxDensityMatrixRepresentation,
    const MPI_Comm                &mpiCommParent,
    const MPI_Comm                &domainComm,
    const MPI_Comm                &interpoolcomm,
    const MPI_Comm                &interBandGroupComm,
    const dftParameters           &dftParams);
} // namespace dftfe
#endif
