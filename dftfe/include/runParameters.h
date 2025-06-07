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



#ifndef runParameters_H_
#define runParameters_H_
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <string>
#include <fstream>
#include <iostream>
#include <TypeConfig.h>

namespace dftfe
{
  /**
   * @brief Namespace which declares the input outer run parameters
   *
   *  @author Sambit Das
   */
  class runParameters
  {
  public:
    dftfe::Int  verbosity;
    std::string solvermode;
    bool        restart;
    std::string restartFilesPath;
    dftfe::Int  numberOfImages;
    bool        imageFreeze;
    double      Kmax;
    double      Kmin;
    double      pathThreshold;
    dftfe::Int  maximumNEBiteration;
    bool        useDevice;

    dftfe::uInt maxLineSearchIterCGPRP;
    std::string bfgsStepMethod;
    double      optimizermaxIonUpdateStep;
    dftfe::uInt lbfgsNumPastSteps;
    std::string optimizationSolver;
    std::string ionRelaxFlagsFile;

    std::string coordinatesFileNEB, domainVectorsFileNEB;
    runParameters() = default;

    /**
     * Parse parameters.
     */
    void
    parse_parameters(const std::string &parameter_file);
    void
    print_parameters();

  private:
    dealii::ParameterHandler prm;



  }; // class runParameters

} // namespace dftfe
#endif
