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
//

#include <FEBasisOperationsKernelsInternal.h>


namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    __global__ void
    reshapeFromNonAffineDeviceKernel(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numCells,
                                     const ValueType  *copyFromVec,
                                     ValueType        *copyToVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = numQuads * numCells * numVecs * 3;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex  = index / numVecs;
          dftfe::uInt iVec        = index - blockIndex * numVecs;
          dftfe::uInt blockIndex2 = blockIndex / numQuads;
          dftfe::uInt iQuad       = blockIndex - blockIndex2 * numQuads;
          dftfe::uInt iCell       = blockIndex2 / 3;
          dftfe::uInt iDim        = blockIndex2 - iCell * 3;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[iVec + iDim * numVecs + iQuad * 3 * numVecs +
                        iCell * 3 * numQuads * numVecs]);
        }
    }
    template <typename ValueType>
    __global__ void
    reshapeToNonAffineDeviceKernel(const dftfe::uInt numVecs,
                                   const dftfe::uInt numQuads,
                                   const dftfe::uInt numCells,
                                   const ValueType  *copyFromVec,
                                   ValueType        *copyToVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = numQuads * numCells * numVecs * 3;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex  = index / numVecs;
          dftfe::uInt iVec        = index - blockIndex * numVecs;
          dftfe::uInt blockIndex2 = blockIndex / numQuads;
          dftfe::uInt iQuad       = blockIndex - blockIndex2 * numQuads;
          dftfe::uInt iCell       = blockIndex2 / 3;
          dftfe::uInt iDim        = blockIndex2 - iCell * 3;
          dftfe::utils::copyValue(copyToVec + iVec + iDim * numVecs +
                                    iQuad * 3 * numVecs +
                                    iCell * 3 * numQuads * numVecs,
                                  copyFromVec[index]);
        }
    }
  } // namespace
  namespace basis
  {
    namespace FEBasisOperationsKernelsInternal
    {
      template <typename ValueType>

      void
      reshapeFromNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                       const dftfe::uInt numQuads,
                                       const dftfe::uInt numCells,
                                       const ValueType  *copyFromVec,
                                       ValueType        *copyToVec)
      {
        DFTFE_LAUNCH_KERNEL(
          reshapeFromNonAffineDeviceKernel,
          (numVecs * numCells * numQuads * 3) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
            1,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          numVecs,
          numQuads,
          numCells,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
      }
      template <typename ValueType>
      void
      reshapeToNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numCells,
                                     const ValueType  *copyFromVec,
                                     ValueType        *copyToVec)
      {
        DFTFE_LAUNCH_KERNEL(
          reshapeToNonAffineDeviceKernel,
          (numVecs * numCells * numQuads * 3) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
            1,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          numVecs,
          numQuads,
          numCells,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
      }
      template void
      reshapeFromNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                       const dftfe::uInt numQuads,
                                       const dftfe::uInt numCells,
                                       const double     *copyFromVec,
                                       double           *copyToVec);
      template void
      reshapeFromNonAffineLayoutDevice(const dftfe::uInt           numVecs,
                                       const dftfe::uInt           numQuads,
                                       const dftfe::uInt           numCells,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double>       *copyToVec);

      template void
      reshapeToNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numCells,
                                     const double     *copyFromVec,
                                     double           *copyToVec);
      template void
      reshapeToNonAffineLayoutDevice(const dftfe::uInt           numVecs,
                                     const dftfe::uInt           numQuads,
                                     const dftfe::uInt           numCells,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double>       *copyToVec);

    } // namespace FEBasisOperationsKernelsInternal
  }   // namespace basis
} // namespace dftfe
