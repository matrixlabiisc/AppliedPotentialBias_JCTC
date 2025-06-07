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

/*
 * @author Sambit Das
 */

#ifndef dftfeMPICommunicatorP2P_h
#define dftfeMPICommunicatorP2P_h

#include <MemorySpaceType.h>
#include <MPIPatternP2P.h>
#include <TypeConfig.h>
#include <MemoryStorage.h>
#include <DataTypeOverloads.h>
#include <dftfeDataTypes.h>
#ifdef DFTFE_WITH_DEVICE
#  include <DeviceTypeConfig.h>
#  if defined(DFTFE_WITH_CUDA_NCCL)
#    include <nccl.h>
#  elif defined(DFTFE_WITH_HIP_RCCL)
#    include <rccl.h>
#  endif
#endif


namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      enum class communicationProtocol
      {
        mpiHost,
        mpiDevice,
        nccl
      };

      enum class communicationPrecision
      {
        single,
        full
      };


      template <typename ValueType, MemorySpace memorySpace>
      class MPICommunicatorP2P
      {
      public:
        MPICommunicatorP2P(
          std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
          const dftfe::uInt                                 blockSize);

        void
        updateGhostValues(MemoryStorage<ValueType, memorySpace> &dataArray,
                          const dftfe::uInt communicationChannel = 0);

        void
        accumulateAddLocallyOwned(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const dftfe::uInt                      communicationChannel = 0);


        void
        updateGhostValuesBegin(MemoryStorage<ValueType, memorySpace> &dataArray,
                               const dftfe::uInt communicationChannel = 0);

        void
        updateGhostValuesEnd(MemoryStorage<ValueType, memorySpace> &dataArray);

        void
        accumulateAddLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const dftfe::uInt                      communicationChannel = 0);

        void
        accumulateAddLocallyOwnedEnd(
          MemoryStorage<ValueType, memorySpace> &dataArray);

        /*
         * @Brief This function copies the value from the ghost nodes
         * to its corresponding local node
         */
        void
        accumulateInsertLocallyOwned(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const dftfe::uInt                      communicationChannel = 0);

        void
        accumulateInsertLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const dftfe::uInt                      communicationChannel = 0);

        void
        accumulateInsertLocallyOwnedEnd(
          MemoryStorage<ValueType, memorySpace> &dataArray);

        std::shared_ptr<const MPIPatternP2P<memorySpace>>
        getMPIPatternP2P() const;

        dftfe::Int
        getBlockSize() const;

        void
        setCommunicationPrecision(communicationPrecision precision);

      private:
        std::shared_ptr<const MPIPatternP2P<memorySpace>> d_mpiPatternP2P;

        dftfe::uInt d_blockSize;

        dftfe::uInt d_locallyOwnedSize;

        dftfe::uInt d_ghostSize;

        MemoryStorage<ValueType, memorySpace> d_sendRecvBuffer;

        MemoryStorage<double, memorySpace> d_tempDoubleRealArrayForAtomics;

        MemoryStorage<double, memorySpace> d_tempDoubleImagArrayForAtomics;

        MemoryStorage<float, memorySpace> d_tempFloatRealArrayForAtomics;

        MemoryStorage<float, memorySpace> d_tempFloatImagArrayForAtomics;

        MemoryStorage<
          typename dftfe::dataTypes::singlePrecType<ValueType>::type,
          memorySpace>
          d_sendRecvBufferSinglePrec;

        MemoryStorage<
          typename dftfe::dataTypes::singlePrecType<ValueType>::type,
          memorySpace>
          d_ghostDataCopySinglePrec;

#ifdef DFTFE_WITH_DEVICE
        std::shared_ptr<MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>
          d_ghostDataCopyHostPinnedPtr;

        std::shared_ptr<MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>
          d_sendRecvBufferHostPinnedPtr;

        std::shared_ptr<MemoryStorage<
          typename dftfe::dataTypes::singlePrecType<ValueType>::type,
          MemorySpace::HOST_PINNED>>
          d_ghostDataCopySinglePrecHostPinnedPtr;

        std::shared_ptr<MemoryStorage<
          typename dftfe::dataTypes::singlePrecType<ValueType>::type,
          MemorySpace::HOST_PINNED>>
          d_sendRecvBufferSinglePrecHostPinnedPtr;
#endif // DFTFE_WITH_DEVICE

        std::vector<MPI_Request> d_requestsUpdateGhostValues;
        std::vector<MPI_Request> d_requestsAccumulateAddLocallyOwned;
        std::vector<MPI_Request> d_requestsAccumulateInsertLocallyOwned;
        MPI_Comm                 d_mpiCommunicator;
#ifdef DFTFE_WITH_DEVICE
        dftfe::utils::deviceStream_t d_deviceCommStream;
#endif
        communicationProtocol  d_commProtocol;
        communicationPrecision d_commPrecision;
      };

    } // namespace mpi
  }   // namespace utils
} // namespace dftfe
#endif // dftfeMPICommunicatorP2P_h
