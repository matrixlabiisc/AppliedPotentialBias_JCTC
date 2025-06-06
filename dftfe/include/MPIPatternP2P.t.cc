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
 * @author Bikash Kanungo
 */

#include <Exceptions.h>
#include <MPITags.h>
#include <MPIRequestersBase.h>
#include <MPIRequestersNBX.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <numeric>
#include <TypeConfig.h>
#include <dftfeDataTypes.h>
namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      namespace
      {
        void
        getAllOwnedRanges(const dftfe::uInt         ownedRangeStart,
                          const dftfe::uInt         ownedRangeEnd,
                          std::vector<dftfe::uInt> &allOwnedRanges,
                          const MPI_Comm           &mpiComm)
        {
          int         nprocs = 1;
          int         err    = MPI_Comm_size(mpiComm, &nprocs);
          std::string errMsg = "Error occured while using MPI_Comm_size. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);
          std::vector<int> recvCounts(nprocs, 2);
          std::vector<int> displs(nprocs, 0);
          allOwnedRanges.resize(2 * nprocs);
          for (int i = 0; i < nprocs; ++i)
            displs[i] = 2 * i;

          std::vector<dftfe::uInt> ownedRanges = {ownedRangeStart,
                                                  ownedRangeEnd};
          MPI_Allgatherv(&ownedRanges[0],
                         2,
                         dftfe::dataTypes::mpi_type_id(ownedRanges.data()),
                         &allOwnedRanges[0],
                         &recvCounts[0],
                         &displs[0],
                         dftfe::dataTypes::mpi_type_id(allOwnedRanges.data()),
                         mpiComm);
        }

        void
        getGhostProcIdToLocalGhostIndicesMap(
          const std::vector<dftfe::uInt> &ghostIndices,
          const dftfe::uInt               nGlobalIndices,
          const std::vector<dftfe::uInt> &allOwnedRanges,
          std::map<dftfe::uInt, std::vector<dftfe::uInt>>
                         &ghostProcIdToLocalGhostIndices,
          const MPI_Comm &mpiComm)
        {
          int         nprocs = 1;
          int         err    = MPI_Comm_size(mpiComm, &nprocs);
          std::string errMsg = "Error occured while using MPI_Comm_size. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPI_SUCCESS, errMsg);

          //
          // NOTE: The locally owned ranges need not be ordered as per the
          // processor ranks. That is ranges for processor 0, 1, ...., P-1 given
          // by [N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P) need not
          // honor the fact that N_0, N_1, ..., N_P are increasing. However, it
          // is more efficient to perform search operations in a sorted vector.
          // Thus, we perform a sort on the end of each locally owned range and
          // also keep track of the indices during the sort
          //

          // vector to store the end of each locally owned ranges
          std::vector<dftfe::uInt> locallyOwnedRangesEnd(nprocs);

          //
          // Vector to keep track of the  indices of locallyOwnedRangesEnd
          // during sort
          std::vector<dftfe::uInt> locallyOwnedRangesEndProcIds(nprocs);
          for (dftfe::uInt i = 0; i < nprocs; ++i)
            {
              locallyOwnedRangesEnd[i] = allOwnedRanges[2 * i + 1];

              // this is required for handing trivial ranges [Nt, Nt)
              // where Nt can be arbitrary integer >=0
              if (allOwnedRanges[2 * i + 1] == allOwnedRanges[2 * i])
                locallyOwnedRangesEnd[i] = (nGlobalIndices + 1 + i);
              locallyOwnedRangesEndProcIds[i] = i;
            }

          std::sort(locallyOwnedRangesEndProcIds.begin(),
                    locallyOwnedRangesEndProcIds.end(),
                    [&locallyOwnedRangesEnd](dftfe::uInt x, dftfe::uInt y) {
                      return locallyOwnedRangesEnd[x] <
                             locallyOwnedRangesEnd[y];
                    });

          std::sort(locallyOwnedRangesEnd.begin(), locallyOwnedRangesEnd.end());

          const dftfe::uInt numGhosts = ghostIndices.size();
          for (dftfe::uInt iGhost = 0; iGhost < numGhosts; ++iGhost)
            {
              dftfe::uInt ghostIndex = ghostIndices[iGhost];
              auto        up  = std::upper_bound(locallyOwnedRangesEnd.begin(),
                                         locallyOwnedRangesEnd.end(),
                                         ghostIndex);
              std::string msg = "Ghost index " + std::to_string(ghostIndex) +
                                " not found in any of the processors";
              throwException(up != locallyOwnedRangesEnd.end(), msg);
              dftfe::uInt upPos =
                std::distance(locallyOwnedRangesEnd.begin(), up);
              dftfe::uInt procId = locallyOwnedRangesEndProcIds[upPos];
              ghostProcIdToLocalGhostIndices[procId].push_back(iGhost);
            }
        }

        bool
        checkContiguity(const std::vector<dftfe::uInt> &v)
        {
          const dftfe::uInt N           = v.size();
          bool              returnValue = true;
          for (dftfe::uInt i = 1; i < N; ++i)
            {
              if ((v[i] - 1) != v[i - 1])
                {
                  returnValue = false;
                  break;
                }
            }
          return returnValue;
        }

        struct RangeMetaData
        {
          dftfe::uInt Id;
          dftfe::uInt rangeId;
          bool        isRangeStart;
        };

        bool
        compareRangeMetaData(const RangeMetaData &x, const RangeMetaData &y)
        {
          if (x.Id == y.Id)
            return (!x.isRangeStart);
          else
            return x.Id < y.Id;
        }
        std::vector<dftfe::uInt>
        getOverlappingRangeIds(const std::vector<dftfe::uInt> &ranges)
        {
          dftfe::uInt                numRanges = ranges.size() / 2;
          std::vector<RangeMetaData> rangeMetaDataVec(0);
          for (dftfe::uInt i = 0; i < numRanges; ++i)
            {
              RangeMetaData left;
              left.Id           = ranges[2 * i];
              left.rangeId      = i;
              left.isRangeStart = true;

              RangeMetaData right;
              right.Id           = ranges[2 * i + 1];
              right.rangeId      = i;
              right.isRangeStart = false;

              // This check is required to ignore ranges with 0 elements
              if (left.Id != right.Id)
                {
                  rangeMetaDataVec.push_back(left);
                  rangeMetaDataVec.push_back(right);
                }
            }
          std::sort(rangeMetaDataVec.begin(),
                    rangeMetaDataVec.end(),
                    compareRangeMetaData);
          dftfe::Int               currentOpen = -1;
          bool                     added       = false;
          std::vector<dftfe::uInt> returnValue(0);
          for (dftfe::uInt i = 0; i < rangeMetaDataVec.size(); ++i)
            {
              dftfe::uInt rangeId = rangeMetaDataVec[i].rangeId;
              if (rangeMetaDataVec[i].isRangeStart)
                {
                  if (currentOpen == -1)
                    {
                      currentOpen = rangeId;
                      added       = false;
                    }
                  else
                    {
                      if (!added)
                        {
                          returnValue.push_back(currentOpen);
                          added = true;
                        }
                      returnValue.push_back(rangeId);
                      if (ranges[2 * rangeId + 1] > ranges[2 * currentOpen + 1])
                        {
                          currentOpen = rangeId;
                          added       = true;
                        }
                    }
                }
              else
                {
                  if (rangeId == currentOpen)
                    {
                      currentOpen = -1;
                      added       = false;
                    }
                }
            }
          return returnValue;
        }

      } // namespace

      ///
      /// Constructor with MPI
      ///
      template <dftfe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
        const std::pair<dftfe::uInt, dftfe::uInt> &locallyOwnedRange,
        const std::vector<dftfe::uInt>            &ghostIndices,
        const MPI_Comm                            &mpiComm)
        : d_locallyOwnedRange(locallyOwnedRange)
        , d_mpiComm(mpiComm)
        , d_allOwnedRanges(0)
        , d_ghostIndices(0)
        , d_numLocallyOwnedIndices(0)
        , d_numGhostIndices(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        d_myRank           = 0;
        d_nprocs           = 1;
        int         err    = MPI_Comm_size(d_mpiComm, &d_nprocs);
        std::string errMsg = "Error occured while using MPI_Comm_size. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);

        err    = MPI_Comm_rank(d_mpiComm, &d_myRank);
        errMsg = "Error occured while using MPI_Comm_rank. "
                 "Error code: " +
                 std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);

        throwException(
          d_locallyOwnedRange.second >= d_locallyOwnedRange.first,
          "In processor " + std::to_string(d_myRank) +
            ", invalid locally owned range found "
            "(i.e., the second value in the range is less than the first value).");
        d_numLocallyOwnedIndices =
          d_locallyOwnedRange.second - d_locallyOwnedRange.first;
        ///////////////////////////////////////////////////
        //////////// Ghost Data Evaluation Begin //////////
        ///////////////////////////////////////////////////

        //
        // check if the ghostIndices is strictly increasing or nor
        //
        bool isStrictlyIncreasing = std::is_sorted(ghostIndices.begin(),
                                                   ghostIndices.end(),
                                                   std::less_equal<>());
        throwException(
          isStrictlyIncreasing,
          "In processor " + std::to_string(d_myRank) +
            ", the ghost indices passed to MPIPatternP2P is not a strictly increasing set.");

        d_ghostIndices = ghostIndices;

        // copy the ghostIndices to d_ghostIndicesSetSTL
        d_ghostIndicesSetSTL.clear();
        std::copy(ghostIndices.begin(),
                  ghostIndices.end(),
                  std::inserter(d_ghostIndicesSetSTL,
                                d_ghostIndicesSetSTL.end()));
        d_ghostIndicesOptimizedIndexSet =
          OptimizedIndexSet<dftfe::uInt>(d_ghostIndicesSetSTL);

        d_numGhostIndices = ghostIndices.size();

        MemoryTransfer<memorySpace, MemorySpace::HOST> memoryTransfer;

        d_allOwnedRanges.clear();
        getAllOwnedRanges(d_locallyOwnedRange.first,
                          d_locallyOwnedRange.second,
                          d_allOwnedRanges,
                          d_mpiComm);

        std::vector<dftfe::uInt> overlappingRangeIds =
          getOverlappingRangeIds(d_allOwnedRanges);
        throwException<LogicError>(
          overlappingRangeIds.size() == 0,
          "Detected overlapping ranges among the locallyOwnedRanges passed "
          "to MPIPatternP2P");

        d_nGlobalIndices = 0;
        for (dftfe::uInt i = 0; i < d_nprocs; ++i)
          {
            d_nGlobalIndices +=
              d_allOwnedRanges[2 * i + 1] - d_allOwnedRanges[2 * i];
          }

        if (ghostIndices.size() > 0)
          {
            throwException<LogicError>(
              ghostIndices.back() < d_nGlobalIndices,
              "Detected global ghost index to be larger than (nGlobalIndices-1)");
          }

        std::map<dftfe::uInt, std::vector<dftfe::uInt>>
          ghostProcIdToLocalGhostIndices;
        getGhostProcIdToLocalGhostIndicesMap(ghostIndices,
                                             d_nGlobalIndices,
                                             d_allOwnedRanges,
                                             ghostProcIdToLocalGhostIndices,
                                             d_mpiComm);

        d_numGhostProcs = ghostProcIdToLocalGhostIndices.size();
        d_ghostProcIds.resize(d_numGhostProcs);
        d_numGhostIndicesInGhostProcs.resize(d_numGhostProcs);
        d_localGhostIndicesRanges.resize(2 * d_numGhostProcs);

        std::vector<dftfe::uInt> flattenedLocalGhostIndicesTmp(0);
        auto                     it = ghostProcIdToLocalGhostIndices.begin();
        dftfe::uInt              iGhostProc = 0;
        for (; it != ghostProcIdToLocalGhostIndices.end(); ++it)
          {
            d_ghostProcIds[iGhostProc] = it->first;
            const std::vector<dftfe::uInt> localGhostIndicesInGhostProc =
              it->second;
            bool isContiguous = checkContiguity(localGhostIndicesInGhostProc);
            std::string msg   = "In rank " + std::to_string(d_myRank) +
                              ", the local ghost indices that are owned"
                              " by rank " +
                              std::to_string(d_ghostProcIds[iGhostProc]) +
                              " does not form a contiguous set.";
            throwException<LogicError>(isContiguous, msg);
            d_numGhostIndicesInGhostProcs[iGhostProc] =
              localGhostIndicesInGhostProc.size();

            d_localGhostIndicesRanges[2 * iGhostProc] =
              *(localGhostIndicesInGhostProc.begin());
            d_localGhostIndicesRanges[2 * iGhostProc + 1] =
              *(localGhostIndicesInGhostProc.end() - 1) + 1;
            //
            // Append localGhostIndicesInGhostProc to
            // flattenedLocalGhostIndicesTmp
            //
            std::copy(localGhostIndicesInGhostProc.begin(),
                      localGhostIndicesInGhostProc.end(),
                      back_inserter(flattenedLocalGhostIndicesTmp));
            ++iGhostProc;
          }

        std::string msg = "In rank " + std::to_string(d_myRank) +
                          " mismatch of"
                          " the sizes of ghost indices. Expected size: " +
                          std::to_string(d_numGhostIndices) +
                          " Received size: " +
                          std::to_string(flattenedLocalGhostIndicesTmp.size());
        throwException<DomainError>(flattenedLocalGhostIndicesTmp.size() ==
                                      d_numGhostIndices,
                                    msg);


        d_flattenedLocalGhostIndices.resize(d_numGhostIndices);
        if (d_numGhostIndices > 0)
          memoryTransfer.copy(d_numGhostIndices,
                              d_flattenedLocalGhostIndices.begin(),
                              &flattenedLocalGhostIndicesTmp[0]);
        ///////////////////////////////////////////////////
        //////////// Ghost Data Evaluation End / //////////
        ///////////////////////////////////////////////////


        ///////////////////////////////////////////////////
        //////////// Target Data Evaluation Begin ////////
        ///////////////////////////////////////////////////
        MPIRequestersNBX mpirequesters(d_ghostProcIds, d_mpiComm);
        d_targetProcIds  = mpirequesters.getRequestingRankIds();
        d_numTargetProcs = d_targetProcIds.size();
        d_numOwnedIndicesForTargetProcs.resize(d_numTargetProcs);

        std::vector<MPI_Request> sendRequests(d_numGhostProcs);
        std::vector<MPI_Status>  sendStatuses(d_numGhostProcs);
        std::vector<MPI_Request> recvRequests(d_numTargetProcs);
        std::vector<MPI_Status>  recvStatuses(d_numTargetProcs);
        const dftfe::Int         tag =
          static_cast<dftfe::Int>(MPITags::MPI_P2P_PATTERN_TAG);
        for (dftfe::uInt iGhost = 0; iGhost < d_numGhostProcs; ++iGhost)
          {
            const dftfe::Int ghostProcId = d_ghostProcIds[iGhost];
            err = MPI_Isend(&d_numGhostIndicesInGhostProcs[iGhost],
                            1,
                            dftfe::dataTypes::mpi_type_id(
                              d_numGhostIndicesInGhostProcs.data()),
                            ghostProcId,
                            tag,
                            d_mpiComm,
                            &sendRequests[iGhost]);
            std::string errMsg = "Error occured while using MPI_Isend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        for (dftfe::uInt iTarget = 0; iTarget < d_numTargetProcs; ++iTarget)
          {
            const dftfe::Int targetProcId = d_targetProcIds[iTarget];
            err = MPI_Irecv(&d_numOwnedIndicesForTargetProcs[iTarget],
                            1,
                            dftfe::dataTypes::mpi_type_id(
                              d_numOwnedIndicesForTargetProcs.data()),
                            targetProcId,
                            tag,
                            d_mpiComm,
                            &recvRequests[iTarget]);
            std::string errMsg = "Error occured while using MPI_Irecv. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (sendRequests.size() > 0)
          {
            err    = MPI_Waitall(d_numGhostProcs,
                              sendRequests.data(),
                              sendStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            err    = MPI_Waitall(d_numTargetProcs,
                              recvRequests.data(),
                              recvStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }


        dftfe::uInt totalOwnedIndicesForTargetProcs =
          std::accumulate(d_numOwnedIndicesForTargetProcs.begin(),
                          d_numOwnedIndicesForTargetProcs.end(),
                          0);


        std::vector<dftfe::uInt> flattenedLocalTargetIndicesTmp(
          totalOwnedIndicesForTargetProcs, 0);

        std::vector<dftfe::uInt> localIndicesForGhostProc(d_numGhostIndices, 0);

        dftfe::uInt startIndex = 0;
        for (dftfe::uInt iGhost = 0; iGhost < d_numGhostProcs; ++iGhost)
          {
            const dftfe::Int numGhostIndicesInProc =
              d_numGhostIndicesInGhostProcs[iGhost];
            const dftfe::Int ghostProcId = d_ghostProcIds[iGhost];

            // We need to send what is the local index in the ghost processor
            // (i.e., the processor that owns the current processor's ghost
            // index)
            for (dftfe::uInt iIndex = 0; iIndex < numGhostIndicesInProc;
                 ++iIndex)
              {
                const dftfe::uInt ghostLocalIndex =
                  flattenedLocalGhostIndicesTmp[startIndex + iIndex];

                throwException<LogicError>(ghostLocalIndex <
                                             ghostIndices.size(),
                                           "BUG1");

                const dftfe::uInt ghostGlobalIndex =
                  ghostIndices[ghostLocalIndex];
                const dftfe::uInt ghostProcOwnedIndicesStart =
                  d_allOwnedRanges[2 * ghostProcId];
                localIndicesForGhostProc[startIndex + iIndex] =
                  (dftfe::uInt)(ghostGlobalIndex - ghostProcOwnedIndicesStart);

                throwException<LogicError>(
                  localIndicesForGhostProc[startIndex + iIndex] <
                    (d_allOwnedRanges[2 * ghostProcId + 1] -
                     d_allOwnedRanges[2 * ghostProcId]),
                  "BUG2");
              }

            err = MPI_Isend(&localIndicesForGhostProc[startIndex],
                            numGhostIndicesInProc,
                            dftfe::dataTypes::mpi_type_id(
                              localIndicesForGhostProc.data()),
                            ghostProcId,
                            tag,
                            d_mpiComm,
                            &sendRequests[iGhost]);
            std::string errMsg = "Error occured while using MPI_Isend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
            startIndex += numGhostIndicesInProc;
          }

        startIndex = 0;
        for (dftfe::uInt iTarget = 0; iTarget < d_numTargetProcs; ++iTarget)
          {
            const dftfe::Int targetProcId = d_targetProcIds[iTarget];
            const dftfe::Int numOwnedIndicesForTarget =
              d_numOwnedIndicesForTargetProcs[iTarget];
            err = MPI_Irecv(&flattenedLocalTargetIndicesTmp[startIndex],
                            numOwnedIndicesForTarget,
                            dftfe::dataTypes::mpi_type_id(
                              flattenedLocalTargetIndicesTmp.data()),
                            targetProcId,
                            tag,
                            d_mpiComm,
                            &recvRequests[iTarget]);
            std::string errMsg = "Error occured while using MPI_Irecv. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
            startIndex += numOwnedIndicesForTarget;
          }

        if (sendRequests.size() > 0)
          {
            err    = MPI_Waitall(d_numGhostProcs,
                              sendRequests.data(),
                              sendStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            err    = MPI_Waitall(d_numTargetProcs,
                              recvRequests.data(),
                              recvStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        for (dftfe::uInt i = 0; i < totalOwnedIndicesForTargetProcs; ++i)
          {
            throwException<LogicError>(
              flattenedLocalTargetIndicesTmp[i] < d_numLocallyOwnedIndices,
              "Detected local owned target index to be larger than (nLocallyOwnedIndices-1)");
          }

        d_flattenedLocalTargetIndices.resize(totalOwnedIndicesForTargetProcs);
        if (totalOwnedIndicesForTargetProcs > 0)
          memoryTransfer.copy(totalOwnedIndicesForTargetProcs,
                              d_flattenedLocalTargetIndices.begin(),
                              &flattenedLocalTargetIndicesTmp[0]);

        ///////////////////////////////////////////////////
        //////////// Target Data Evaluation End ////////
        ///////////////////////////////////////////////////
      }


      ///
      /// Constructor for a serial case
      ///
      template <dftfe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(const dftfe::uInt size)
        : d_locallyOwnedRange(std::make_pair(0, (dftfe::uInt)size))
        , d_mpiComm(MPI_COMM_SELF)
        , d_allOwnedRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_numGhostIndices(0)
        , d_ghostIndices(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        d_myRank = 0;
        d_nprocs = 1;
        throwException(
          d_locallyOwnedRange.second >= d_locallyOwnedRange.first,
          "In processor " + std::to_string(d_myRank) +
            ", invalid locally owned range found "
            "(i.e., the second value in the range is less than the first value).");
        d_numLocallyOwnedIndices =
          d_locallyOwnedRange.second - d_locallyOwnedRange.first;
        std::vector<dftfe::uInt> d_allOwnedRanges = {
          d_locallyOwnedRange.first, d_locallyOwnedRange.second};
        for (dftfe::uInt i = 0; i < d_nprocs; ++i)
          d_nGlobalIndices +=
            d_allOwnedRanges[2 * i + 1] - d_allOwnedRanges[2 * i];

        // set the d_ghostIndicesSetSTL to be of size zero
        d_ghostIndicesSetSTL.clear();
        d_ghostIndicesOptimizedIndexSet =
          OptimizedIndexSet<dftfe::uInt>(d_ghostIndicesSetSTL);
      }

      template <dftfe::utils::MemorySpace memorySpace>
      std::pair<dftfe::uInt, dftfe::uInt>
      MPIPatternP2P<memorySpace>::getLocallyOwnedRange() const
      {
        return d_locallyOwnedRange;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      const std::vector<dftfe::uInt> &
      MPIPatternP2P<memorySpace>::getGhostIndices() const
      {
        return d_ghostIndices;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      const std::vector<dftfe::uInt> &
      MPIPatternP2P<memorySpace>::getGhostProcIds() const
      {
        return d_ghostProcIds;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      const std::vector<dftfe::uInt> &
      MPIPatternP2P<memorySpace>::getNumGhostIndicesInProcs() const
      {
        return d_numGhostIndicesInGhostProcs;
      }


      template <dftfe::utils::MemorySpace memorySpace>
      const std::vector<dftfe::uInt> &
      MPIPatternP2P<memorySpace>::getGhostLocalIndicesRanges() const
      {
        return d_localGhostIndicesRanges;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::getNumGhostIndicesInProc(
        const dftfe::uInt procId) const
      {
        auto        itProcIds         = d_ghostProcIds.begin();
        auto        itNumGhostIndices = d_numGhostIndicesInGhostProcs.begin();
        dftfe::uInt numGhostIndicesInProc = 0;
        for (; itProcIds != d_ghostProcIds.end(); ++itProcIds)
          {
            numGhostIndicesInProc = *itNumGhostIndices;
            if (procId == *itProcIds)
              break;

            ++itNumGhostIndices;
          }

        std::string msg =
          "The processor Id " + std::to_string(procId) +
          " does not contain any ghost indices for the current processor"
          " (i.e., processor Id " +
          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_ghostProcIds.end(), msg);

        return numGhostIndicesInProc;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      typename MPIPatternP2P<memorySpace>::SizeTypeVector
      MPIPatternP2P<memorySpace>::getGhostLocalIndices(
        const dftfe::uInt procId) const
      {
        dftfe::uInt cumulativeIndices     = 0;
        dftfe::uInt numGhostIndicesInProc = 0;
        auto        itProcIds             = d_ghostProcIds.begin();
        auto        itNumGhostIndices = d_numGhostIndicesInGhostProcs.begin();
        for (; itProcIds != d_ghostProcIds.end(); ++itProcIds)
          {
            numGhostIndicesInProc = *itNumGhostIndices;
            if (procId == *itProcIds)
              break;

            cumulativeIndices += numGhostIndicesInProc;
            ++itNumGhostIndices;
          }

        std::string msg =
          "The processor Id " + std::to_string(procId) +
          " does not contain any ghost indices for the current processor"
          " (i.e., processor Id " +
          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_ghostProcIds.end(), msg);

        SizeTypeVector returnValue(numGhostIndicesInProc);
        MemoryTransfer<memorySpace, memorySpace>::copy(
          numGhostIndicesInProc,
          returnValue.begin(),
          d_flattenedLocalGhostIndices.begin() + cumulativeIndices);

        return returnValue;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      const std::vector<dftfe::uInt> &
      MPIPatternP2P<memorySpace>::getTargetProcIds() const
      {
        return d_targetProcIds;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      const std::vector<dftfe::uInt> &
      MPIPatternP2P<memorySpace>::getNumOwnedIndicesForTargetProcs() const
      {
        return d_numOwnedIndicesForTargetProcs;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      const typename MPIPatternP2P<memorySpace>::SizeTypeVector &
      MPIPatternP2P<memorySpace>::getOwnedLocalIndicesForTargetProcs() const
      {
        return d_flattenedLocalTargetIndices;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::getNumOwnedIndicesForTargetProc(
        const dftfe::uInt procId) const
      {
        auto        itProcIds         = d_targetProcIds.begin();
        auto        itNumOwnedIndices = d_numOwnedIndicesForTargetProcs.begin();
        dftfe::uInt numOwnedIndicesForProc = 0;
        for (; itProcIds != d_targetProcIds.end(); ++itProcIds)
          {
            numOwnedIndicesForProc = *itNumOwnedIndices;
            if (procId == *itProcIds)
              break;

            ++itNumOwnedIndices;
          }

        std::string msg = "There are no owned indices for "
                          " target processor Id " +
                          std::to_string(procId) +
                          " in the current processor"
                          " (i.e., processor Id " +
                          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_targetProcIds.end(),
                                        msg);
        return numOwnedIndicesForProc;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      typename MPIPatternP2P<memorySpace>::SizeTypeVector
      MPIPatternP2P<memorySpace>::getOwnedLocalIndices(
        const dftfe::uInt procId) const
      {
        dftfe::uInt cumulativeIndices      = 0;
        dftfe::uInt numOwnedIndicesForProc = 0;
        auto        itProcIds              = d_targetProcIds.begin();
        auto        itNumOwnedIndices = d_numOwnedIndicesForTargetProcs.begin();
        for (; itProcIds != d_targetProcIds.end(); ++itProcIds)
          {
            numOwnedIndicesForProc = *itNumOwnedIndices;
            if (procId == *itProcIds)
              break;

            cumulativeIndices += numOwnedIndicesForProc;
            ++itNumOwnedIndices;
          }

        std::string msg = "There are no owned indices for "
                          " target processor Id " +
                          std::to_string(procId) +
                          " in the current processor"
                          " (i.e., processor Id " +
                          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_targetProcIds.end(),
                                        msg);

        SizeTypeVector returnValue(numOwnedIndicesForProc);
        MemoryTransfer<memorySpace, memorySpace>::copy(
          numOwnedIndicesForProc,
          returnValue.begin(),
          d_flattenedLocalTargetIndices.begin() + cumulativeIndices);

        return returnValue;
      }



      template <dftfe::utils::MemorySpace memorySpace>
      const MPI_Comm &
      MPIPatternP2P<memorySpace>::mpiCommunicator() const
      {
        return d_mpiComm;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::nmpiProcesses() const
      {
        return d_nprocs;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::thisProcessId() const
      {
        return d_myRank;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::nGlobalIndices() const
      {
        return d_nGlobalIndices;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::localOwnedSize() const
      {
        return d_numLocallyOwnedIndices;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::localGhostSize() const
      {
        return d_numGhostIndices;
      }


      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::localToGlobal(const dftfe::uInt localId) const
      {
        dftfe::uInt returnValue = 0;
        if (localId < d_numLocallyOwnedIndices)
          {
            returnValue = d_locallyOwnedRange.first + localId;
          }
        else if (localId < (d_numLocallyOwnedIndices + d_numGhostIndices))
          {
            auto it =
              d_ghostIndices.begin() + (localId - d_numLocallyOwnedIndices);
            returnValue = *it;
          }
        else
          {
            std::string msg =
              "In processor " + std::to_string(d_myRank) +
              ", the local index " + std::to_string(localId) +
              " passed to localToGlobal() in MPIPatternP2P is"
              " larger than number of locally owned plus ghost indices.";
            throwException<InvalidArgument>(false, msg);
          }
        return returnValue;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      dftfe::uInt
      MPIPatternP2P<memorySpace>::globalToLocal(
        const dftfe::uInt globalId) const
      {
        dftfe::uInt returnValue = 0;
        if (globalId >= d_locallyOwnedRange.first &&
            globalId < d_locallyOwnedRange.second)
          {
            returnValue = globalId - d_locallyOwnedRange.first;
          }
        else
          {
            bool found = false;
            d_ghostIndicesOptimizedIndexSet.getPosition(globalId,
                                                        returnValue,
                                                        found);
            std::string msg =
              "In processor " + std::to_string(d_myRank) +
              ", the global index " + std::to_string(globalId) +
              " passed to globalToLocal() in MPIPatternP2P is"
              " neither present in its locally owned range nor in its "
              " ghost indices.";
            throwException<InvalidArgument>(found, msg);
            returnValue += d_numLocallyOwnedIndices;
          }

        return returnValue;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      bool
      MPIPatternP2P<memorySpace>::inLocallyOwnedRange(
        const dftfe::uInt globalId) const
      {
        return (globalId >= d_locallyOwnedRange.first &&
                globalId < d_locallyOwnedRange.second);
      }

      template <dftfe::utils::MemorySpace memorySpace>
      bool
      MPIPatternP2P<memorySpace>::isGhostEntry(const dftfe::uInt globalId) const
      {
        bool        found = false;
        dftfe::uInt localId;
        d_ghostIndicesOptimizedIndexSet.getPosition(globalId, localId, found);
        return found;
      }

      template <dftfe::utils::MemorySpace memorySpace>
      bool
      MPIPatternP2P<memorySpace>::isCompatible(
        const MPIPatternP2P<memorySpace> &rhs) const
      {
        if (d_nprocs != rhs.d_nprocs)
          return false;

        else if (d_nGlobalIndices != rhs.d_nGlobalIndices)
          return false;

        else if (d_locallyOwnedRange != rhs.d_locallyOwnedRange)
          return false;

        else if (d_numGhostIndices != rhs.d_numGhostIndices)
          return false;

        else
          return (d_ghostIndicesOptimizedIndexSet ==
                  rhs.d_ghostIndicesOptimizedIndexSet);
      }
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftfe
