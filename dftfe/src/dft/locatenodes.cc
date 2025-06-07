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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//
#include <dft.h>

namespace dftfe
{
  // source file for locating core atom nodes
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::locateAtomCoreNodes(
    const dealii::DoFHandler<3> &_dofHandler,
    std::map<dealii::types::global_dof_index, double>
      &atomNodeIdToChargeValueMap)
  {
    dealii::TimerOutput::Scope scope(computing_timer, "locate atom nodes");
    atomNodeIdToChargeValueMap.clear();
    const dftfe::uInt vertices_per_cell =
      dealii::GeometryInfo<3>::vertices_per_cell;

    const bool isPseudopotential = d_dftParamsPtr->isPseudopotential;

    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  _dofHandler.begin_active(),
                                                endc = _dofHandler.end();

    dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();

    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);

    // locating atom nodes
    const dftfe::uInt     numAtoms = atomLocations.size();
    std::set<dftfe::uInt> atomsTolocate;
    for (dftfe::uInt i = 0; i < numAtoms; i++)
      atomsTolocate.insert(i);
    // element loop
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        for (dftfe::uInt i = 0; i < vertices_per_cell; ++i)
          {
            const dealii::types::global_dof_index nodeID =
              cell->vertex_dof_index(i, 0);
            dealii::Point<3> feNodeGlobalCoord = cell->vertex(i);
            //
            // loop over all atoms to locate the corresponding nodes
            //
            for (std::set<dftfe::uInt>::iterator it = atomsTolocate.begin();
                 it != atomsTolocate.end();
                 ++it)
              {
                dealii::Point<3> atomCoord(atomLocations[*it][2],
                                           atomLocations[*it][3],
                                           atomLocations[*it][4]);
                if (feNodeGlobalCoord.distance(atomCoord) < 1.0e-5)
                  {
#ifdef DEBUG
                    if (isPseudopotential)
                      {
                        if (d_dftParamsPtr->verbosity >= 4)
                          {
                            std::cout << "atom core with valence charge "
                                      << atomLocations[*it][1]
                                      << " located with node id " << nodeID
                                      << " in processor " << this_mpi_process
                                      << " nodal coor " << feNodeGlobalCoord[0]
                                      << " " << feNodeGlobalCoord[1] << " "
                                      << feNodeGlobalCoord[2] << std::endl;
                          }
                      }
                    else
                      {
                        if (d_dftParamsPtr->verbosity >= 4)
                          {
                            std::cout << "atom core with charge "
                                      << atomLocations[*it][0]
                                      << " located with node id " << nodeID
                                      << " in processor " << this_mpi_process
                                      << " nodal coor " << feNodeGlobalCoord[0]
                                      << " " << feNodeGlobalCoord[1] << " "
                                      << feNodeGlobalCoord[2] << std::endl;
                          }
                      }
#endif
                    if (locallyOwnedDofs.is_element(nodeID))
                      {
                        if (isPseudopotential)
                          atomNodeIdToChargeValueMap.insert(
                            std::pair<dealii::types::global_dof_index, double>(
                              nodeID, atomLocations[*it][1]));
                        else
                          atomNodeIdToChargeValueMap.insert(
                            std::pair<dealii::types::global_dof_index, double>(
                              nodeID, atomLocations[*it][0]));
#ifdef DEBUG
                        if (d_dftParamsPtr->verbosity >= 4)
                          std::cout << " and added \n";
#endif
                      }
                    else
                      {
#ifdef DEBUG
                        if (d_dftParamsPtr->verbosity >= 4)
                          std::cout << " but skipped \n";
#endif
                      }
                    atomsTolocate.erase(*it);
                    break;
                  } // tolerance check if loop
              }     // atomsTolocate loop
          }         // vertices_per_cell loop
    MPI_Barrier(mpi_communicator);

    const dftfe::uInt totalAtomNodesFound =
      dealii::Utilities::MPI::sum(atomNodeIdToChargeValueMap.size(),
                                  mpi_communicator);
    AssertThrow(totalAtomNodesFound == numAtoms,
                dealii::ExcMessage(
                  "Atleast one atom doesn't lie on a triangulation vertex"));
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::locatePeriodicPinnedNodes(
    const dealii::DoFHandler<3>             &_dofHandler,
    const dealii::AffineConstraints<double> &constraintsBase,
    dealii::AffineConstraints<double>       &constraints)
  {
    // pin a node away from all atoms in case of full PBC for total
    // electrostatic potential solve
    if (!(d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
          d_dftParamsPtr->periodicZ))
      return;

    dealii::TimerOutput::Scope scope(computing_timer,
                                     "locate periodic pinned node");
    const dftfe::Int           numberImageCharges = d_imageIds.size();
    const dftfe::Int           numberGlobalAtoms  = atomLocations.size();
    const dftfe::Int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;

    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);
    dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();

    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);

    //
    // find vertex furthest from all nuclear charges
    //
    double                          maxDistance = -1.0;
    dealii::types::global_dof_index maxNode, minNode;

    std::map<dealii::types::global_dof_index, dealii::Point<3>>::iterator
      iterMap;
    for (iterMap = supportPoints.begin(); iterMap != supportPoints.end();
         ++iterMap)
      if (locallyOwnedDofs.is_element(iterMap->first) &&
          !(constraintsBase.is_constrained(iterMap->first) &&
            !constraintsBase.is_identity_constrained(iterMap->first)))
        {
          double minDistance                     = 1e10;
          minNode                                = -1;
          dealii::Point<3> nodalPointCoordinates = iterMap->second;
          for (dftfe::uInt iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
            {
              dealii::Point<3> atomCoor;

              if (iAtom < numberGlobalAtoms)
                {
                  atomCoor[0] = atomLocations[iAtom][2];
                  atomCoor[1] = atomLocations[iAtom][3];
                  atomCoor[2] = atomLocations[iAtom][4];
                }
              else
                {
                  //
                  // Fill with ImageAtom Coors
                  //
                  atomCoor[0] = d_imagePositions[iAtom - numberGlobalAtoms][0];
                  atomCoor[1] = d_imagePositions[iAtom - numberGlobalAtoms][1];
                  atomCoor[2] = d_imagePositions[iAtom - numberGlobalAtoms][2];
                }

              double distance = atomCoor.distance(nodalPointCoordinates);

              if (distance <= minDistance)
                {
                  minDistance = distance;
                  minNode     = iterMap->first;
                }
            }

          if (minDistance > maxDistance)
            {
              maxDistance = minDistance;
              maxNode     = iterMap->first;
            }
        }

    double globalMaxDistance;

    MPI_Allreduce(&maxDistance,
                  &globalMaxDistance,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_communicator);



    // locating pinned nodes
    std::vector<std::vector<double>> pinnedLocations;
    std::vector<double>              temp(3, 0.0);
    std::vector<double>              tempLocal(3, 0.0);
    dftfe::uInt                      taskId = 0;

    if (std::abs(maxDistance - globalMaxDistance) < 1e-07)
      taskId = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    dftfe::uInt maxTaskId;

    MPI_Allreduce(&taskId,
                  &maxTaskId,
                  1,
                  dftfe::dataTypes::mpi_type_id(&taskId),
                  MPI_MAX,
                  mpi_communicator);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == maxTaskId)
      {
#ifdef DEBUG
        if (d_dftParamsPtr->verbosity >= 4)
          std::cout << "Found Node locally on processor Id: "
                    << dealii::Utilities::MPI::this_mpi_process(
                         mpi_communicator)
                    << std::endl;
#endif
        if (locallyOwnedDofs.is_element(maxNode))
          {
            if (constraintsBase.is_identity_constrained(maxNode))
              {
                const dealii::types::global_dof_index masterNode =
                  (*constraintsBase.get_constraint_entries(maxNode))[0].first;
                dealii::Point<3> nodalPointCoordinates =
                  supportPoints.find(masterNode)->second;
                tempLocal[0] = nodalPointCoordinates[0];
                tempLocal[1] = nodalPointCoordinates[1];
                tempLocal[2] = nodalPointCoordinates[2];
              }
            else
              {
                dealii::Point<3> nodalPointCoordinates =
                  supportPoints.find(maxNode)->second;
                tempLocal[0] = nodalPointCoordinates[0];
                tempLocal[1] = nodalPointCoordinates[1];
                tempLocal[2] = nodalPointCoordinates[2];
              }
            // checkFlag = 1;
          }
      }


    MPI_Allreduce(
      &tempLocal[0], &temp[0], 3, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    pinnedLocations.push_back(temp);


    const dftfe::uInt dofs_per_cell = _dofHandler.get_fe().dofs_per_cell;
    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  _dofHandler.begin_active(),
                                                endc = _dofHandler.end();

    const dftfe::uInt     numberNodes = pinnedLocations.size();
    std::set<dftfe::uInt> nodesTolocate;
    for (dftfe::uInt i = 0; i < numberNodes; i++)
      nodesTolocate.insert(i);

    std::vector<dealii::types::global_dof_index> cell_dof_indices(
      dofs_per_cell);
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(cell_dof_indices);

          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
            {
              const dealii::types::global_dof_index nodeID =
                cell_dof_indices[i];
              dealii::Point<3> feNodeGlobalCoord =
                supportPoints[cell_dof_indices[i]];

              //
              // loop over all atoms to locate the corresponding nodes
              //
              for (std::set<dftfe::uInt>::iterator it = nodesTolocate.begin();
                   it != nodesTolocate.end();
                   ++it)
                {
                  dealii::Point<3> pinnedNodeCoord(pinnedLocations[*it][0],
                                                   pinnedLocations[*it][1],
                                                   pinnedLocations[*it][2]);
                  if (feNodeGlobalCoord.distance(pinnedNodeCoord) < 1.0e-5)
                    {
                      if (d_dftParamsPtr->verbosity >= 4)
                        std::cout << "Pinned core with nodal coordinates ("
                                  << pinnedLocations[*it][0] << " "
                                  << pinnedLocations[*it][1] << " "
                                  << pinnedLocations[*it][2]
                                  << ") located with node id " << nodeID
                                  << " in processor " << this_mpi_process
                                  << std::endl;
                      if (locallyRelevantDofs.is_element(nodeID))
                        {
                          constraints.add_line(nodeID);
                          constraints.set_inhomogeneity(nodeID, 0.0);
                        }
                      nodesTolocate.erase(*it);
                      break;
                    } // tolerance check if loop

                } // atomsTolocate loop

            } // vertices_per_cell loop

        } // locally owned cell if loop

    MPI_Barrier(mpi_communicator);
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::locateConstrainedPotentialPinnedNodes(
    const double                             zCoordinateL,
    const double                             zCoordinateR,
    const double                             potentialValueL,
    const double                             potentialValueR,
    const bool                               localizedFlag,
    const double                             localizedWidth,
    const dealii::DoFHandler<3>             &_dofHandler,
    const dealii::AffineConstraints<double> &constraintMatrixBase,
    dealii::AffineConstraints<double>       &constraintMatrix)
  {
    dealii::TimerOutput::Scope scope(computing_timer,
                                     "locate Applied potential pinned node");
    pcout << " Input Data to locate nodal points: " << zCoordinateL << " "
          << zCoordinateR << " " << potentialValueL << " " << potentialValueR
          << std::endl;
    d_constraintNodesGlobalL.clear();
    d_constraintNodesGlobalR.clear();
    d_constraintNodesL.clear();
    d_constraintNodesR.clear();
    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt dofs_per_face  = _dofHandler.get_fe().dofs_per_face;
    const dftfe::uInt dofs_per_cell  = _dofHandler.get_fe().dofs_per_cell;
    dealii::IndexSet  locallyOwnedDofs = _dofHandler.locally_owned_dofs();
    dealii::IndexSet  locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);
    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);
    std::vector<bool> dofs_touched(_dofHandler.n_dofs(), false);
    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  _dofHandler.begin_active(),
                                                endc = _dofHandler.end();
    std::vector<dealii::types::global_dof_index> cell_dof_indices(
      dofs_per_cell);
    dftfe::uInt                                  count = 0;
    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);
    double minDistanceL = 1e10;
    double minDistanceR = 1e10;
    double min_zCoordinateL, min_zCoordinateR;
    std::vector<
      std::pair<dealii::DoFHandler<3>::active_cell_iterator, dftfe::uInt>>
      cellFacePair;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(cellGlobalDofIndices);
          for (dftfe::uInt iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
              const dealii::types::global_dof_index nodeId0 =
                iFaceGlobalDofIndices[0];
              dealii::Point<3> p0 =
                d_supportPointsPRefined.find(nodeId0)->second;
              const double distanceL = std::fabs(p0[2] - zCoordinateL);
              const double distanceR = std::fabs(p0[2] - zCoordinateR);
              if (distanceL <= minDistanceL || distanceR <= minDistanceR)
                {
                  bool faceNormalZ = true;
                  for (dftfe::uInt iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      dealii::Point<3> p =
                        d_supportPointsPRefined.find(nodeId)->second;
                      if (std::fabs(p[2] - p0[2]) > 1.0e-10)
                        {
                          faceNormalZ = false;
                          break;
                        }
                    }
                  if (faceNormalZ)
                    {
                      if (distanceL < minDistanceL)
                        {
                          minDistanceL     = distanceL;
                          min_zCoordinateL = p0[2];
                        }
                      if (distanceR < minDistanceR)
                        {
                          minDistanceR     = distanceR;
                          min_zCoordinateR = p0[2];
                        }
                      cellFacePair.push_back(std::make_pair(cell, iFace));
                    }
                }
            }
        }
    double globalMinDistanceL, globalMinDistanceR;
    MPI_Allreduce(&minDistanceL,
                  &globalMinDistanceL,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_communicator);

    MPI_Allreduce(&minDistanceR,
                  &globalMinDistanceR,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_communicator);

    pcout << "Shifting of the Left surface by: " << globalMinDistanceL
          << std::endl;
    pcout << "Shifting of the Right surface by: " << globalMinDistanceR
          << std::endl;

    std::vector<dealii::Point<3>> constraintCoordinatesL,
      constraintCoordinatesR;
    // std::cout << "Processor min coordinates: " << min_zCoordinateL << " "
    //           << min_zCoordinateR << " " << this_mpi_process << std::endl;
    if ((!std::fabs(globalMinDistanceL - minDistanceL) <= 1E-8))
      min_zCoordinateL = -1000;
    if ((!std::fabs(globalMinDistanceR - minDistanceR) <= 1E-8))
      min_zCoordinateR = 1000;
    // std::cout << "Processor min coordinates: " << min_zCoordinateL << " "
    //           << min_zCoordinateR << " " << this_mpi_process << std::endl;

    double global_minZcoordinateL;
    double global_minZcoordinateR;
    MPI_Allreduce(&min_zCoordinateL,
                  &global_minZcoordinateL,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_communicator);

    MPI_Allreduce(&min_zCoordinateR,
                  &global_minZcoordinateR,
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_communicator);
    pcout << "Coorinates of the Left surface by: " << global_minZcoordinateL
          << std::endl;
    pcout << "Coorinates of the Right surface by: " << global_minZcoordinateR
          << std::endl;
    for (cell = _dofHandler.begin_active(); cell != endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(cellGlobalDofIndices);
          if (true)
            {
              for (dftfe::uInt inode = 0; inode < dofs_per_cell;
                   ++inode) // loop over Dofs
                {
                  const dealii::types::global_dof_index nodeID =
                    cellGlobalDofIndices[inode];
                  dealii::Point<3> p0 =
                    d_supportPointsPRefined.find(nodeID)->second;
                  double zCoordinateNode = p0[2];
                  if (dofs_touched[nodeID])
                    continue;
                  dofs_touched[nodeID] = true;
                  if (!(constraintMatrixBase.is_constrained(nodeID)) &&
                      !(constraintMatrixBase.is_identity_constrained(nodeID)))
                    {
                      if (zCoordinateNode <= global_minZcoordinateL)
                        {
                          constraintMatrix.add_line(nodeID);
                          constraintMatrix.set_inhomogeneity(nodeID,
                                                             potentialValueL);
                          d_constraintNodesGlobalL.push_back(nodeID);
                          if (locallyOwnedDofs.is_element(nodeID))
                            {
                              constraintCoordinatesL.push_back(p0);
                              d_constraintNodesL.push_back(nodeID);
                            }
                        }
                      else if (zCoordinateNode >= global_minZcoordinateR)
                        {
                          // if(localizedFlag)
                          // pcout<<"Cell Center: "<<cell->center()[0]<<"
                          // "<<cell->center()[1]<<" "<<localizedWidth / 2<<"
                          // "<<(!localizedFlag ||
                          //       (std::fabs(cell->center()[0] ) <=
                          //       localizedWidth / 2 &&
                          //        std::fabs(cell->center()[1]) <=
                          //        localizedWidth / 2))<<std::endl;
                          if ((!localizedFlag ||
                               (std::fabs(cell->center()[0]) <=
                                  localizedWidth / 2 &&
                                std::fabs(cell->center()[1]) <=
                                  localizedWidth / 2)))
                            {
                              constraintMatrix.add_line(nodeID);
                              constraintMatrix.set_inhomogeneity(
                                nodeID, potentialValueR);
                              d_constraintNodesGlobalR.push_back(nodeID);
                              if (locallyOwnedDofs.is_element(nodeID))
                                {
                                  constraintCoordinatesR.push_back(p0);
                                  d_constraintNodesR.push_back(nodeID);
                                }
                            }
                        }
                    }
                }
            }
        }
    MPI_Barrier(mpi_communicator);
    pcout << "Total number of MPI tasks: " << n_mpi_processes << std::endl;
    dftfe::uInt totalconstraintsL = 0.0;
    dftfe::uInt totalconstraintsR = 0.0;
    dftfe::uInt constrainsLsize   = constraintCoordinatesL.size();
    dftfe::uInt constrainsRsize   = constraintCoordinatesR.size();
    MPI_Allreduce(&constrainsLsize,
                  &totalconstraintsL,
                  1,
                  MPI_UNSIGNED,
                  MPI_SUM,
                  mpi_communicator);
    MPI_Allreduce(&constrainsRsize,
                  &totalconstraintsR,
                  1,
                  MPI_UNSIGNED,
                  MPI_SUM,
                  mpi_communicator);
    pcout << "=== Total L constrained nodes: " << totalconstraintsL
          << std::endl;
    pcout << "=== Total R constrained nodes: " << totalconstraintsR
          << std::endl;
    // for (int iProc = 0; iProc < n_mpi_processes; iProc++)
    //   {
    //     MPI_Barrier(mpi_communicator);
    //     if (iProc == this_mpi_process)
    //       {
    //         std::cout << "MPI Task: " << this_mpi_process << std::endl
    //                   << std::flush;
    //         for (int i = 0; i < constraintCoordinatesL.size(); i++)
    //           {
    //             std::cout << "Coordinate of constraint nodes for L: "
    //                       << constraintCoordinatesL[i][0] << " "
    //                       << constraintCoordinatesL[i][1] << " "
    //                       << constraintCoordinatesL[i][2] << std::endl
    //                       << std::flush;
    //           }
    //         for (int i = 0; i < constraintCoordinatesR.size(); i++)
    //           {
    //             std::cout << "Coordinate of constraint nodes for R: "
    //                       << constraintCoordinatesR[i][0] << " "
    //                       << constraintCoordinatesR[i][1] << " "
    //                       << constraintCoordinatesR[i][2] << std::endl
    //                       << std::flush;
    //           }
    //       }
    //     MPI_Barrier(mpi_communicator);
    //   }
    MPI_Barrier(mpi_communicator);
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::printAppliedPotentialAtConstraintNodes(
    const distributedCPUVec<double> &nodalField,
    const std::vector<dftfe::uInt>  &localConstrainedIndex,
    const dealii::DoFHandler<3>     &_dofHandler)
  {
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);
    dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();
    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);

    for (int iProc = 0; iProc < n_mpi_processes; iProc++)
      {
        if (iProc == this_mpi_process)
          {
            std::cout << "MPI Processer ID: " << this_mpi_process
                      << " no. of constrained elements: "
                      << localConstrainedIndex.size() << std::flush
                      << std::endl;
            for (int i = 0; i < localConstrainedIndex.size(); i++)
              {
                dealii::Point<3> nodalPointCoordinates =
                  supportPoints
                    .find(nodalField.get_partitioner()->local_to_global(
                      localConstrainedIndex[i]))
                    ->second;
                std::cout << "Nodal Value and Z coordinate: "
                          << nodalField.local_element(localConstrainedIndex[i])
                          << " " << nodalPointCoordinates[2] << std::endl;
              }
          }
        MPI_Barrier(mpi_communicator);
      }
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::printAppliedPotentialAtConstraintNodes(
    const distributedCPUVec<double>                    &nodalField,
    const std::vector<dealii::types::global_dof_index> &constraintNodesL,
    const std::vector<dealii::types::global_dof_index> &constraintNodesR,
    const dealii::DoFHandler<3>                        &_dofHandler)
  {
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);
    dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();
    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);

    // for (int iProc = 0; iProc < n_mpi_processes; iProc++)
    //   {
    //     if (iProc == this_mpi_process)
    //       {
    //         std::cout << "MPI Processer ID: " << this_mpi_process
    //                   << " no. of constrained elementsL: "
    //                   << constraintNodesL.size() << std::flush << std::endl;
    //         for (int i = 0; i < constraintNodesL.size(); i++)
    //           {
    //             dealii::Point<3> nodalPointCoordinates =
    //               supportPoints.find(constraintNodesL[i])->second;
    //             const dftfe::uInt localNodeID =
    //               nodalField.get_partitioner()->global_to_local(
    //                 constraintNodesL[i]);
    //             std::cout << "Nodal ValueL, Z coordinate and global DOF: "
    //                       << nodalField.local_element(localNodeID) << " "
    //                       << nodalPointCoordinates[2] << " "
    //                       << constraintNodesL[i] << std::endl;
    //           }
    //       }
    //     MPI_Barrier(mpi_communicator);
    //   }
    // MPI_Barrier(mpi_communicator);
    // for (int iProc = 0; iProc < n_mpi_processes; iProc++)
    //   {
    //     if (iProc == this_mpi_process)
    //       {
    //         std::cout << "MPI Processer ID: " << this_mpi_process
    //                   << " no. of constrained elementsR: "
    //                   << constraintNodesR.size() << std::flush << std::endl;
    //         for (int i = 0; i < constraintNodesR.size(); i++)
    //           {
    //             dealii::Point<3> nodalPointCoordinates =
    //               supportPoints.find(constraintNodesR[i])->second;
    //             const dftfe::uInt localNodeID =
    //               nodalField.get_partitioner()->global_to_local(
    //                 constraintNodesR[i]);
    //             std::cout << "Nodal ValueR, Z coordinate and global DOF: "
    //                       << nodalField.local_element(localNodeID) << " "
    //                       << nodalPointCoordinates[2] << " "
    //                       << constraintNodesR[i] << std::endl;
    //           }
    //       }
    //     MPI_Barrier(mpi_communicator);
    //   }
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::modifyConstrainedNodesWithVself(
    const double                             potentialValueL,
    const double                             potentialValueR,
    const dealii::DoFHandler<3>             &_dofHandler,
    const dealii::AffineConstraints<double> &constraintMatrixBase,
    const std::vector<double>               &nodalField,
    dealii::AffineConstraints<double>       &constraintMatrix)
  {
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);
    dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();
    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);
    for (dftfe::uInt i = 0; i < d_constraintNodesGlobalL.size(); i++)
      {
        const double tempValue =
          potentialValueL - nodalField[d_constraintNodesGlobalL[i]];
        constraintMatrix.add_line(d_constraintNodesGlobalL[i]);
        constraintMatrix.set_inhomogeneity(d_constraintNodesGlobalL[i],
                                           tempValue);
      }
    for (dftfe::uInt i = 0; i < d_constraintNodesGlobalR.size(); i++)
      {
        const double tempValue =
          potentialValueR - nodalField[d_constraintNodesGlobalR[i]];
        constraintMatrix.add_line(d_constraintNodesGlobalR[i]);
        constraintMatrix.set_inhomogeneity(d_constraintNodesGlobalR[i],
                                           tempValue);
      }
  }
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::printNodalFieldAtCenterLine(
    const distributedCPUVec<double> &nodalField,
    const dealii::DoFHandler<3>     &_dofHandler)
  {
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);
    dealii::IndexSet locallyOwnedDofs = _dofHandler.locally_owned_dofs();
    std::map<dealii::types::global_dof_index, dealii::Point<3>> supportPoints;
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 _dofHandler,
                                                 supportPoints);

    std::map<dealii::types::global_dof_index, dealii::Point<3>>::iterator
                                     iterMap;
    std::vector<std::vector<double>> tempVec;
    for (iterMap = supportPoints.begin(); iterMap != supportPoints.end();
         ++iterMap)
      {
        if (locallyOwnedDofs.is_element(iterMap->first))
          {
            dealii::Point<3> nodalPointCoordinates = iterMap->second;
            if (std::fabs(nodalPointCoordinates[0]) <= 1E-8 &&
                std::fabs(nodalPointCoordinates[1]) <= 1E-8)
              {
                const dftfe::uInt localNodeID =
                  nodalField.get_partitioner()->global_to_local(iterMap->first);
                std::vector<double> temp(2, 0.0);
                temp[0] = nodalPointCoordinates[2];
                temp[1] = nodalField.local_element(localNodeID);
                tempVec.push_back(temp);
              }
          }
      }
    //       std::cout << "MPI Processer ID: " << this_mpi_process
    //                 << " no. of DOFs "
    //                 << tempVec.size() << std::flush << std::endl;
    // std::cout<<std::flush<<std::endl;
    // for (int iProc = 0; iProc < n_mpi_processes; iProc++)
    // {
    //   if (iProc == this_mpi_process)
    //     {

    //       for (dftfe::uInt i = 0; i < tempVec.size(); i++)
    //         {
    //           std::cout<<"Nodal Field at z-coordinate: "<<tempVec[i][0]<<"
    //           "<<tempVec[i][1]<<std::flush<<std::endl;
    //         }
    //     }
    // }
  }


} // namespace dftfe
