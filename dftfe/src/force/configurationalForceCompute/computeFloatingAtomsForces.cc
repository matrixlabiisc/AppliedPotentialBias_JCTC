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
  forceClass<memorySpace>::computeFloatingAtomsForces()
  {
    dftfe::uInt vertices_per_cell = dealii::GeometryInfo<3>::vertices_per_cell;
    const std::vector<std::vector<double>> &atomLocations =
      dftPtr->atomLocations;
    const dftfe::Int numberGlobalAtoms = atomLocations.size();
    d_globalAtomsForces.clear();
    d_globalAtomsForces.resize(numberGlobalAtoms * 3, 0.0);

    // Sum over band parallelization and k point pools
    MPI_Allreduce(MPI_IN_PLACE,
                  &(d_forceAtomsFloating[0]),
                  numberGlobalAtoms * 3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  dftPtr->interBandGroupComm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &(d_forceAtomsFloating[0]),
                  numberGlobalAtoms * 3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  dftPtr->interpoolcomm);



#ifdef USE_COMPLEX
    // Sum over band parallelization and k point pools
    MPI_Allreduce(MPI_IN_PLACE,
                  &(d_forceAtomsFloatingKPoints[0]),
                  numberGlobalAtoms * 3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  dftPtr->interBandGroupComm);

    MPI_Allreduce(MPI_IN_PLACE,
                  &(d_forceAtomsFloatingKPoints[0]),
                  numberGlobalAtoms * 3,
                  MPI_DOUBLE,
                  MPI_SUM,
                  dftPtr->interpoolcomm);
#endif

    // add to total Gaussian force
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
      for (dftfe::uInt idim = 0; idim < 3; idim++)
#ifdef USE_COMPLEX
        d_globalAtomsForces[iAtom * 3 + idim] =
          d_forceAtomsFloating[iAtom * 3 + idim] +
          d_forceAtomsFloatingKPoints[iAtom * 3 + idim];
#else
        d_globalAtomsForces[iAtom * 3 + idim] =
          d_forceAtomsFloating[iAtom * 3 + idim];
#endif
  }
#include "../force.inst.cc"
} // namespace dftfe
