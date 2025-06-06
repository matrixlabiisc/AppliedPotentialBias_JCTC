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
//

#ifndef converter_h
#define converter_h
#include <string>

#include "string.h"
namespace dftfe
{
  //
  // Declare pseudoUtils function
  //

  /**
   *  @brief wrapper to convert pseudopotential file from upf to dftfe format and returns the nonlinear core correction
   *  flag
   *
   *  The functionality reads a file containing list of pseudopotential files in
   * upf format and converts into into dftfe format -via- xml file format
   *
   *  @author Phani Motamarri
   */

  namespace pseudoUtils
  {
    std::vector<dftfe::Int>
    convert(const std::string &file,
            const std::string &dftfeScratchFolderName,
            const dftfe::Int   verbosity,
            const dftfe::uInt  natomTypes,
            const bool         pseudoTestsFlag);
  }
} // namespace dftfe
#endif
