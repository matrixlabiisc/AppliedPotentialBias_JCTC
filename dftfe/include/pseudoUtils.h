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

#ifndef pseudoUtils_H_
#define pseudoUtils_H_

#include <headers.h>
#include <mpi.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>


namespace dftfe
{
  namespace pseudoUtils
  {
    // some inline functions
    inline void
    exchangeLocalList(const std::vector<dftfe::uInt> &masterNodeIdList,
                      std::vector<dftfe::uInt>       &globalMasterNodeIdList,
                      dftfe::uInt                     numMeshPartitions,
                      const MPI_Comm                 &mpi_communicator)
    {
      int numberMasterNodesOnLocalProc = masterNodeIdList.size();

      int *masterNodeIdListSizes = new int[numMeshPartitions];

      MPI_Allgather(&numberMasterNodesOnLocalProc,
                    1,
                    dftfe::dataTypes::mpi_type_id(
                      &numberMasterNodesOnLocalProc),
                    masterNodeIdListSizes,
                    1,
                    dftfe::dataTypes::mpi_type_id(masterNodeIdListSizes),
                    mpi_communicator);

      dftfe::Int newMasterNodeIdListSize =
        std::accumulate(&(masterNodeIdListSizes[0]),
                        &(masterNodeIdListSizes[numMeshPartitions]),
                        0);

      globalMasterNodeIdList.resize(newMasterNodeIdListSize);

      int *mpiOffsets = new int[numMeshPartitions];

      mpiOffsets[0] = 0;

      for (dftfe::Int i = 1; i < numMeshPartitions; ++i)
        mpiOffsets[i] = masterNodeIdListSizes[i - 1] + mpiOffsets[i - 1];

      MPI_Allgatherv(&(masterNodeIdList[0]),
                     numberMasterNodesOnLocalProc,
                     dftfe::dataTypes::mpi_type_id(masterNodeIdList.data()),
                     &(globalMasterNodeIdList[0]),
                     &(masterNodeIdListSizes[0]),
                     &(mpiOffsets[0]),
                     dftfe::dataTypes::mpi_type_id(
                       globalMasterNodeIdList.data()),
                     mpi_communicator);


      delete[] masterNodeIdListSizes;
      delete[] mpiOffsets;

      return;
    }


    inline void
    exchangeNumberingMap(std::map<dftfe::Int, dftfe::Int> &localMap,
                         dftfe::uInt                       numMeshPartitions,
                         const MPI_Comm                   &mpi_communicator)

    {
      std::map<dftfe::Int, dftfe::Int>::iterator iter;

      std::vector<dftfe::Int> localSpreadVec;

      iter = localMap.begin();
      while (iter != localMap.end())
        {
          localSpreadVec.push_back(iter->first);
          localSpreadVec.push_back(iter->second);

          ++iter;
        }
      int localSpreadVecSize = localSpreadVec.size();

      int *spreadVecSizes = new int[numMeshPartitions];

      MPI_Allgather(&localSpreadVecSize,
                    1,
                    dftfe::dataTypes::mpi_type_id(&localSpreadVecSize),
                    spreadVecSizes,
                    1,
                    dftfe::dataTypes::mpi_type_id(spreadVecSizes),
                    mpi_communicator);

      dftfe::Int globalSpreadVecSize =
        std::accumulate(&(spreadVecSizes[0]),
                        &(spreadVecSizes[numMeshPartitions]),
                        0);

      std::vector<dftfe::Int> globalSpreadVec(globalSpreadVecSize);

      int *mpiOffsets = new int[numMeshPartitions];

      mpiOffsets[0] = 0;

      for (dftfe::Int i = 1; i < numMeshPartitions; ++i)
        mpiOffsets[i] = spreadVecSizes[i - 1] + mpiOffsets[i - 1];

      MPI_Allgatherv(&(localSpreadVec[0]),
                     localSpreadVecSize,
                     dftfe::dataTypes::mpi_type_id(localSpreadVec.data()),
                     &(globalSpreadVec[0]),
                     &(spreadVecSizes[0]),
                     &(mpiOffsets[0]),
                     dftfe::dataTypes::mpi_type_id(globalSpreadVec.data()),
                     mpi_communicator);

      for (dftfe::Int i = 0; i < globalSpreadVecSize; i = i + 2)
        localMap[globalSpreadVec[i]] = globalSpreadVec[i + 1];


      delete[] spreadVecSizes;
      delete[] mpiOffsets;

      return;
    }

    inline void
    getRadialFunctionVal(const double                       radialCoordinate,
                         double                            &splineVal,
                         const alglib::spline1dinterpolant *spline)
    {
      splineVal = alglib::spline1dcalc(*spline, radialCoordinate);
      return;
    }

    inline void
    getSphericalHarmonicVal(const double     theta,
                            const double     phi,
                            const dftfe::Int l,
                            const dftfe::Int m,
                            double          &sphericalHarmonicVal)
    {
      if (m < 0)
        sphericalHarmonicVal =
          std::sqrt(2.0) * boost::math::spherical_harmonic_i(l, -m, theta, phi);

      else if (m == 0)
        sphericalHarmonicVal =
          boost::math::spherical_harmonic_r(l, m, theta, phi);

      else if (m > 0)
        sphericalHarmonicVal =
          std::sqrt(2.0) * boost::math::spherical_harmonic_r(l, m, theta, phi);

      return;
    }

    inline void
    convertCartesianToSpherical(double *x,
                                double &r,
                                double &theta,
                                double &phi)
    {
      double tolerance = 1e-12;
      r                = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

      if (std::fabs(r - 0.0) <= tolerance)
        {
          theta = 0.0;
          phi   = 0.0;
        }
      else
        {
          theta = std::acos(x[2] / r);
          //
          // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
          // If yes, assign phi = 0.0.
          // NOTE: In case theta = 0 or PI, phi is undetermined. The actual
          // value of phi doesn't matter in computing the enriched function
          // value or its gradient. We assign phi = 0.0 here just as a dummy
          // value
          //
          if (fabs(theta - 0.0) >= tolerance && fabs(theta - M_PI) >= tolerance)
            phi = std::atan2(x[1], x[0]);
          else
            phi = 0.0;
        }
    }
  } // namespace pseudoUtils
} // namespace dftfe
#endif
