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
// @author  Vishal Subramanian, Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONPROJECTORSPLINE_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONPROJECTORSPLINE_H

#include "AtomCenteredSphericalFunctionSpline.h"
#include "string"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fileReaders.h>
#include <dftUtils.h>
#include <interpolation.h>


namespace dftfe
{
  class AtomCenteredSphericalFunctionProjectorSpline
    : public AtomCenteredSphericalFunctionSpline
  {
  public:
    /**
     * @brief Creates splines for radial-Local Potential from file by applying suitable BC on spline and determining the cutOff Radius
     * @param[in] filename the location of file containing the data
     * @param[in] l quantumNumber-l
     * @param[in] radialPower  mulitply the data with pow(r,radialPower)
     * @param[in] colIndex the column Number where the function data is present
     * @param[in] totalColSize the total number oc columns in file
     * @param[in] truncationTol the minimum function value afterwhich the
     * function is truncated.
     * @param[in]  consider0thEntry whether to replace the 0th fn value with the
     * 1st value in the data.
     */
    AtomCenteredSphericalFunctionProjectorSpline(std::string filename,
                                                 dftfe::uInt l,
                                                 dftfe::Int  radialPower,
                                                 dftfe::Int  colIndex,
                                                 dftfe::Int  totalColSize,
                                                 double truncationTol  = 1E-10,
                                                 bool consider0thEntry = true);
  };

} // end of namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONPROJECTORSPLINE_H
