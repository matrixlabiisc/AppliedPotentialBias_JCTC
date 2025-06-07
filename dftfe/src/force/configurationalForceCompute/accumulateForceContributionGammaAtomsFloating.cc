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
// @author Sambit Das
//
#include <dft.h>
#include <force.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::accumulateForceContributionGammaAtomsFloating(
    const std::map<dftfe::uInt, std::vector<double>>
                        &forceContributionLocalGammaAtoms,
    std::vector<double> &accumForcesVector)
  {
    for (dftfe::uInt iAtom = 0; iAtom < dftPtr->atomLocations.size(); iAtom++)
      {
        std::vector<double> forceContributionLocalGammaiAtomGlobal(3);
        std::vector<double> forceContributionLocalGammaiAtomLocal(3, 0.0);

        if (forceContributionLocalGammaAtoms.find(iAtom) !=
            forceContributionLocalGammaAtoms.end())
          {
            forceContributionLocalGammaiAtomLocal =
              forceContributionLocalGammaAtoms.find(iAtom)->second;
          }
        else
          {
            std::fill(forceContributionLocalGammaiAtomLocal.begin(),
                      forceContributionLocalGammaiAtomLocal.end(),
                      0.0);
          }

        // accumulate value
        MPI_Allreduce(&(forceContributionLocalGammaiAtomLocal[0]),
                      &(forceContributionLocalGammaiAtomGlobal[0]),
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);

        for (dftfe::uInt idim = 0; idim < 3; idim++)
          accumForcesVector[iAtom * 3 + idim] +=
            forceContributionLocalGammaiAtomGlobal[idim];
      }
  }
#include "../force.inst.cc"
} // namespace dftfe
