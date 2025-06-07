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
 * @author Vishal Subramanian, Bikash Kanungo
 */

#ifndef DFTFE_INTERPOLATECELLWISEDATATOPOINTS_H
#define DFTFE_INTERPOLATECELLWISEDATATOPOINTS_H

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/range/adaptors.hpp>
#include "BLASWrapper.h"
#include "MapPointsToCells.h"
#include "Cell.h"
#include "InterpolateFromCellToLocalPoints.h"


namespace dftfe
{
  /**
   * @brief This class forms the interface for interpolating data to an arbitrary set of
   * points. This class is compatible with MPI, where the partitioning of cells
   * and the points need not be compatible. As in the points need not lie within
   * the cells assigned to the processor.
   *
   * @author Vishal Subramanian, Bikash Kanungo
   */
  template <typename T, dftfe::utils::MemorySpace memorySpace>
  class InterpolateCellWiseDataToPoints
  {
  public:
    /**
     * @brief This constructor computes the mapping between the targetPts and srcCells.
     * In case of incompatible partitioning, some targetPts can lie outside the
     * cells assigned to the processor. In that case, the unmapped points are
     * sent to other processors. Similarly it receives points from other
     * processors and checks if any of them lies within its cells. Once all the
     * points that lie within its cells are found, they are then passed to
     * interpolateLocalObj, which provides the functionality to interpolate to
     * those points
     * @param[in] srcCells Cells that are assigned to the processor
     * @param[in] interpolateLocalObj Class that can take in a Cell and provide
     * the functionality to interpolate to points that lie within that Cell.
     * @param[in] targetPts The set of points onto which the data needs to be
     * interpolated
     * @param[in] numDofsPerElem The number of basis function that is non-zero
     * overlap with each cell. This is set to be a vector so that different
     * cells can have different number of basis functions.
     * @param[in] mpiComm The mpi communicator which has been used for the
     * domain decomposition
     *
     * @author Vishal Subramanian, Bikash Kanungo
     */
    InterpolateCellWiseDataToPoints(
      const std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> &srcCells,
      std::vector<
        std::shared_ptr<InterpolateFromCellToLocalPoints<T, memorySpace>>>
                                              interpolateLocalObj,
      const std::vector<std::vector<double>> &targetPts,
      const std::vector<dftfe::uInt>         &numDofsPerElem,
      const dftfe::uInt                       verbosity,
      const MPI_Comm                         &mpiComm);
    /**
     * @brief This function interpolates from the data to the points passed to the constructor.
     * The function copies the nodal data to cell wise data and then
     * interpolates to all the points that lie within that cell. Then they are
     * copied to the output vector. At the end a mpi call is performed to gather
     * the value of points that do not lie within processor from othe
     * processors.
     * @param[in] BLASWrapperPtr BLAS Wrapper that provides the handle to the
     * linear algebra routines
     * @param[in] inputVec The input data. The input data should be of size
     * locally_owned*numberOfVectors
     * @param[in] numberOfVectors The number of vectors (blockSize) in the input
     * data
     * @param[in] mapVecToCells The mapping that tells the nodal data to the
     * cell wise data
     * @param[out] outputData The output where the nodal data is interpolated to
     * the points. The memory layout of outputData is as follows - the memory is
     * stored in the same order as the target points. In addition to the target
     * points, there are ghost points which are points that lie within its cells
     * but lie in cells assigned to a different processor.
     * @param[in] resizeData The output data should be of size (locally owned +
     * ghost)*numberOfVectors If the flag resizeData is set to true, outputData
     * is resized appropriately.
     *
     * @author Vishal Subramanian, Bikash Kanungo
     */
    void
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                              &BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T, memorySpace> &inputVec,
      const dftfe::uInt                                        numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace>
                                                  &mapVecToCells,
      dftfe::utils::MemoryStorage<T, memorySpace> &outputData,
      const dftfe::uInt                            blockSizeOfInputData,
      const dftfe::uInt                            blockSizeOfOutputData,
      const dftfe::uInt                            startIndexOfInputData,
      bool                                         resizeData = false);

    /**
     * @brief The function is same as above but set to dealii:distributed::Vector
     * @param[in] BLASWrapperPtr BLAS Wrapper that provides the handle to the
     * linear algebra routines
     * @param[in] inputVec The input data. The input data should be of size
     * locally_owned*numberOfVectors
     * @param[in] numberOfVectors The number of vectors (blockSize) in the input
     * data
     * @param[in] mapVecToCells The mapping that tells the nodal data to the
     * cell wise data
     * @param[out] outputData The output where the nodal data is interpolated to
     * the points. The memory layout of outputData is as follows - the memory is
     * stored in the same order as the target points. In addition to the target
     * points, there are ghost points which are points that lie within its cells
     * but lie in cells assigned to a different processor.
     * @param[in] resizeData The output data should be of size (locally owned +
     * ghost)*numberOfVectors If the flag resizeData is set to true, outputData
     * is resized appropriately.
     *
     * @author Vishal Subramanian, Bikash Kanungo
     */
    void
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                 &BLASWrapperPtr,
      const distributedCPUVec<T> &inputVec,
      const dftfe::uInt           numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::HOST>
        &mapVecToCells,
      dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
                       &outputData,
      const dftfe::uInt blockSizeOfInputData,
      const dftfe::uInt blockSizeOfOutputData,
      const dftfe::uInt startIndexOfInputData,
      bool              resizeData = false);

  private:
    void
    checkIfAllPointsAreFound(const std::vector<std::vector<double>> &targetPts);


    dftfe::utils::MapPointsToCells<3, 8>
      d_mapPoints; /// TODO check if M=8 is optimal
                   //    std::vector<T> d_shapeFuncValues;
    std::vector<dftfe::uInt> d_cellPointStartIndex, d_cellShapeFuncStartIndex;
    const MPI_Comm           d_mpiComm;

    std::shared_ptr<
      dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      d_mpiPatternP2PPtr;

    std::shared_ptr<
      dftfe::utils::mpi::MPICommunicatorP2P<T, dftfe::utils::MemorySpace::HOST>>
      d_mpiCommP2PPtr;

    std::shared_ptr<dftfe::utils::mpi::MPIPatternP2P<memorySpace>>
      d_mpiP2PPtrMemSpace;
    std::unique_ptr<dftfe::utils::mpi::MPICommunicatorP2P<T, memorySpace>>
      d_mpiCommPtrMemSpace;


    dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace>
      d_mapPointToProcLocalMemSpace;


    dftfe::utils::MemoryStorage<T, memorySpace> d_cellLevelParentNodalMemSpace;

    dftfe::uInt              d_numPointsLocal;
    dftfe::uInt              d_numCells;
    std::vector<dftfe::uInt> d_numDofsPerElement;
    std::vector<dftfe::uInt> d_cumulativeDofs;
    dftfe::uInt              totalDofsInCells;

    std::vector<dftfe::uInt> d_numPointsInCell;

    std::vector<std::vector<dftfe::uInt>> d_mapCellLocalToProcLocal;


    dftfe::uInt d_numLocalPtsSze;

    dftfe::uInt d_pointsFoundInProc;

    std::vector<dftfe::uInt>            d_ghostGlobalIds;
    std::pair<dftfe::uInt, dftfe::uInt> d_localRange;

    dftfe::utils::MemoryStorage<T, memorySpace> d_tempOutputMemSpace;

    std::vector<
      std::shared_ptr<InterpolateFromCellToLocalPoints<T, memorySpace>>>
                d_interpolateLocalObj;
    dftfe::uInt d_verbosity;
  }; // end of class InterpolateCellWiseDataToPoints
} // end of namespace dftfe


#endif // DFTFE_INTERPOLATECELLWISEDATATOPOINTS_H
