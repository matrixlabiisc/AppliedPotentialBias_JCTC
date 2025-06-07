// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Kartick Ramakrishnan


#include <dft.h>

namespace dftfe
{
  namespace internal
  {
    double
    sawToothPotential(const double x, const double emaxPos, const double eopreg)
    {
      double z = x - emaxPos;
      double y = z - std::floor(z);
      if (y <= eopreg)
        return ((0.5 - y / eopreg) * (1.0 - eopreg));
      else
        return ((-0.5 + (y - eopreg) / (1.0 - eopreg)) * (1.0 - eopreg));
    }
    double
    sawToothPotential2(const double x,
                       const double fracCoordL,
                       const double fracCoordR,
                       const double minVal,
                       const double maxVal)
    {
      if (x <= fracCoordL)
        return minVal;
      else if (x >= fracCoordR)
        return maxVal;
      else
        {
          double value = 0.0;
          value        = minVal + (x - fracCoordL) * (maxVal - minVal) /
                             (fracCoordR - fracCoordL);
          return value;
        }
    }


  } // namespace internal


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeUExtPotentialAtDensityQuadPoints()
  {
    pcout
      << "Computing External Electric Potential at density quadrature points using: "
      << d_dftParamsPtr->externalPotentialType << " function..." << std::endl;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId, false);
    const dftfe::uInt nCells        = d_basisOperationsPtrHost->nCells();
    const int         nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();
    d_uExtQuadValuesRho.clear();
    d_uExtQuadValuesRho.resize(nQuadsPerCell * nCells);

    for (dftfe::uInt iCell = 0; iCell < nCells; iCell++)
      {
        const double *quadPointsInCell =
          d_basisOperationsPtrHost->quadPoints().data() +
          iCell * 3 * nQuadsPerCell;
        for (int iQuad = 0; iQuad < nQuadsPerCell; iQuad++)
          {
            const double zCoordinate = (quadPointsInCell[3 * iQuad + 2] +
                                        d_domainBoundingVectors[2][2] / 2) /
                                       (d_domainBoundingVectors[2][2]);
            d_uExtQuadValuesRho[iCell * nQuadsPerCell + iQuad] =
              d_dftParamsPtr->externalPotentialSlope *
              internal::sawToothPotential(zCoordinate,
                                          d_dftParamsPtr->emaxPos,
                                          d_dftParamsPtr->eopreg) *
              d_domainBoundingVectors[2][2];
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeUExtPotentialAtNuclearQuadPoints(
    const std::map<dealii::CellId, std::vector<dftfe::uInt>>
      &smearedbNonTrivialAtomIds)
  {
    pcout
      << "Computing External Electric Potential at Nuclear quadrature points using: "
      << d_dftParamsPtr->externalPotentialType << " function..." << std::endl;
    d_basisOperationsPtrElectroHost->reinit(0,
                                            0,
                                            d_smearedChargeQuadratureIdElectro,
                                            false);
    const dftfe::uInt nCells = d_basisOperationsPtrElectroHost->nCells();
    const int nQuadsPerCell  = d_basisOperationsPtrElectroHost->nQuadsPerCell();
    d_uExtQuadValuesNuclear.clear();



    for (auto const &[key, val] : smearedbNonTrivialAtomIds)
      {
        dealii::CellId cellId = key;
        dftfe::uInt iCell = d_basisOperationsPtrElectroHost->cellIndex(cellId);
        std::vector<double> temp(nQuadsPerCell, 0.0);
        const double       *quadPointsInCell =
          d_basisOperationsPtrElectroHost->quadPoints().data() +
          iCell * 3 * nQuadsPerCell;
        for (int iQuad = 0; iQuad < nQuadsPerCell; iQuad++)
          {
            const double zCoordinate = (quadPointsInCell[3 * iQuad + 2] +
                                        d_domainBoundingVectors[2][2] / 2) /
                                       (d_domainBoundingVectors[2][2]);
            temp[iQuad] = d_dftParamsPtr->externalPotentialSlope *
                          internal::sawToothPotential(zCoordinate,
                                                      d_dftParamsPtr->emaxPos,
                                                      d_dftParamsPtr->eopreg) *
                          d_domainBoundingVectors[2][2];
          }
        d_uExtQuadValuesNuclear[cellId] = temp;
      }
  }

#include "dft.inst.cc"
} // namespace dftfe
