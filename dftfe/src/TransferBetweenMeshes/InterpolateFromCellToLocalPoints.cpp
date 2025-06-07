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

#include "InterpolateFromCellToLocalPoints.h"
namespace dftfe
{
  template <typename T, dftfe::utils::MemorySpace memorySpace>
  InterpolateFromCellToLocalPoints<T, memorySpace>::
    InterpolateFromCellToLocalPoints(
      const std::shared_ptr<const dftfe::utils::FECell<3>> &srcCell,
      dftfe::uInt                                           numNodes,
      bool                                                  memOpt)
  {
    d_srcCell  = srcCell;
    d_numNodes = numNodes;
    d_memOpt   = memOpt;
  }

  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateFromCellToLocalPoints<T, memorySpace>::
    setRealCoordinatesOfLocalPoints(dftfe::uInt          numPoints,
                                    std::vector<double> &coordinates)
  {
    d_numPoints = numPoints;
    if (d_memOpt)
      {
        d_paramCoordinates =
          d_srcCell->getParametricPointForAllPoints(numPoints, coordinates);
      }
    else
      {
        d_shapeValuesHost.resize(numPoints * d_numNodes);
        d_srcCell->getShapeFuncValues(
          numPoints, coordinates, d_shapeValuesHost, 0, d_numNodes);
        d_shapeValuesMemSpace.resize(numPoints * d_numNodes);
        d_shapeValuesMemSpace.copyFrom(d_shapeValuesHost);
      }
  }

  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateFromCellToLocalPoints<T, memorySpace>::interpolate(
    const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
               &BLASWrapperPtr,
    dftfe::uInt numberOfVectors,
    const T    *parentNodalMemSpacePtr,
    T          *outputMemSpacePtr)
  {
    const T           scalarCoeffAlpha = 1.0;
    const T           scalarCoeffBeta  = 0.0;
    const char        transA = 'N', transB = 'N';
    const dftfe::uInt inc           = 1;
    const dftfe::uInt numPoints     = 500;
    const dftfe::uInt numRemPoints  = d_numPoints % numPoints;
    const dftfe::uInt numFullPoints = d_numPoints - numRemPoints;
    std::cout << "Num Points, Num Full Points: " << numPoints << " "
              << numRemPoints << " " << d_numPoints << std::endl;
    if (d_memOpt)
      {
        d_shapeValuesHost.resize(numPoints * d_numNodes);
        d_shapeValuesMemSpace.resize(numPoints * d_numNodes);
      }
    for (dftfe::uInt iStart = 0; iStart < numFullPoints; iStart += numPoints)
      {
        std::vector<double> paramCoordinates(numPoints * 3, 0.0);
        std::memcpy(paramCoordinates.data(),
                    &d_paramCoordinates[iStart * 3],
                    numPoints * 3 * sizeof(double));
        if (d_memOpt)
          {
            d_srcCell->getShapeFuncValuesFromParametricPoints(
              numPoints, paramCoordinates, d_shapeValuesHost, 0, d_numNodes);
            d_shapeValuesMemSpace.copyFrom(d_shapeValuesHost);
          }

        BLASWrapperPtr->xgemm(transA,
                              transB,
                              numberOfVectors,
                              numPoints,
                              d_numNodes,
                              &scalarCoeffAlpha,
                              parentNodalMemSpacePtr,
                              numberOfVectors,
                              d_shapeValuesMemSpace.data(),
                              d_numNodes,
                              &scalarCoeffBeta,
                              outputMemSpacePtr + iStart * numberOfVectors,
                              numberOfVectors);
      }
    if (numRemPoints > 0)
      {
        if (d_memOpt)
          {
            d_shapeValuesHost.resize(0);
            d_shapeValuesMemSpace.resize(0);
          }
        std::vector<double> paramCoordinates(numRemPoints * 3, 0.0);
        std::memcpy(paramCoordinates.data(),
                    &d_paramCoordinates[numFullPoints * 3],
                    numRemPoints * 3 * sizeof(double));
        if (d_memOpt)
          {
            d_shapeValuesHost.resize(numRemPoints * d_numNodes);
            d_srcCell->getShapeFuncValuesFromParametricPoints(
              numRemPoints, paramCoordinates, d_shapeValuesHost, 0, d_numNodes);
            d_shapeValuesMemSpace.resize(numRemPoints * d_numNodes);
            d_shapeValuesMemSpace.copyFrom(d_shapeValuesHost);
          }

        BLASWrapperPtr->xgemm(transA,
                              transB,
                              numberOfVectors,
                              numRemPoints,
                              d_numNodes,
                              &scalarCoeffAlpha,
                              parentNodalMemSpacePtr,
                              numberOfVectors,
                              d_shapeValuesMemSpace.data(),
                              d_numNodes,
                              &scalarCoeffBeta,
                              outputMemSpacePtr +
                                numFullPoints * numberOfVectors,
                              numberOfVectors);
      }

    if (d_memOpt)
      {
        d_shapeValuesHost.resize(0);
        d_shapeValuesMemSpace.resize(0);
      }

    // ----> Old COde <-------

    // if (d_memOpt)
    //   {
    //     d_shapeValuesHost.resize(d_numPoints * d_numNodes);
    //     d_srcCell->getShapeFuncValuesFromParametricPoints(
    //       d_numPoints, d_paramCoordinates, d_shapeValuesHost, 0, d_numNodes);
    //     d_shapeValuesMemSpace.resize(d_numPoints * d_numNodes);
    //     d_shapeValuesMemSpace.copyFrom(d_shapeValuesHost);
    //   }


    // BLASWrapperPtr->xgemm(transA,
    //                       transB,
    //                       numberOfVectors,
    //                       d_numPoints,
    //                       d_numNodes,
    //                       &scalarCoeffAlpha,
    //                       parentNodalMemSpacePtr,
    //                       numberOfVectors,
    //                       d_shapeValuesMemSpace.data(),
    //                       d_numNodes,
    //                       &scalarCoeffBeta,
    //                       outputMemSpacePtr,
    //                       numberOfVectors);

    // if (d_memOpt)
    //   {
    //     d_shapeValuesHost.resize(0);
    //     d_shapeValuesMemSpace.resize(0);
    //   }
  }

  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateFromCellToLocalPoints<T, memorySpace>::interpolate(
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
    dftfe::uInt           numberOfVectors,
    const std::vector<T> &parentNodalHost,
    std::vector<T>       &outputHost)
  {
    const T           scalarCoeffAlpha = 1.0;
    const T           scalarCoeffBeta  = 0.0;
    const char        transA = 'N', transB = 'N';
    const dftfe::uInt inc = 1;


    if (d_memOpt)
      {
        d_shapeValuesHost.resize(d_numPoints * d_numNodes);
        d_srcCell->getShapeFuncValuesFromParametricPoints(
          d_numPoints, d_paramCoordinates, d_shapeValuesHost, 0, d_numNodes);
      }

    BLASWrapperPtr->xgemm(transA,
                          transB,
                          numberOfVectors,
                          d_numPoints,
                          d_numNodes,
                          &scalarCoeffAlpha,
                          &parentNodalHost[0],
                          numberOfVectors,
                          d_shapeValuesHost.data(),
                          d_numNodes,
                          &scalarCoeffBeta,
                          &outputHost[0],
                          numberOfVectors);

    if (d_memOpt)
      {
        d_shapeValuesHost.resize(0);
      }
  }

  template class InterpolateFromCellToLocalPoints<
    double,
    dftfe::utils::MemorySpace::HOST>;
#if defined(USE_COMPLEX)
  template class InterpolateFromCellToLocalPoints<
    std::complex<double>,
    dftfe::utils::MemorySpace::HOST>;
#endif

#ifdef DFTFE_WITH_DEVICE

  template class InterpolateFromCellToLocalPoints<
    double,
    dftfe::utils::MemorySpace::DEVICE>;
#  if defined(USE_COMPLEX)
  template class InterpolateFromCellToLocalPoints<
    std::complex<double>,
    dftfe::utils::MemorySpace::DEVICE>;
#  endif
#endif

} // namespace dftfe
