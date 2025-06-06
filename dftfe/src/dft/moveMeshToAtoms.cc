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

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::calculateNearestAtomDistances()
  {
    const dftfe::uInt numberGlobalAtoms = atomLocations.size();
    const dftfe::uInt numberImageAtoms  = d_imageIdsTrunc.size();
    d_nearestAtomDistances.clear();
    d_nearestAtomIds.clear();
    d_nearestAtomDistances.resize(numberGlobalAtoms, 1e+6);
    d_nearestAtomIds.resize(numberGlobalAtoms);
    for (dftfe::uInt i = 0; i < numberGlobalAtoms; i++)
      {
        dealii::Point<3> atomCoori;
        dealii::Point<3> atomCoorj;
        atomCoori[0] = atomLocations[i][2];
        atomCoori[1] = atomLocations[i][3];
        atomCoori[2] = atomLocations[i][4];
        for (dftfe::uInt j = 0; j < (numberGlobalAtoms + numberImageAtoms); j++)
          {
            dftfe::Int jatomId;

            if (j < numberGlobalAtoms)
              {
                atomCoorj[0] = atomLocations[j][2];
                atomCoorj[1] = atomLocations[j][3];
                atomCoorj[2] = atomLocations[j][4];
                jatomId      = j;
              }
            else
              {
                atomCoorj[0] = d_imagePositionsTrunc[j - numberGlobalAtoms][0];
                atomCoorj[1] = d_imagePositionsTrunc[j - numberGlobalAtoms][1];
                atomCoorj[2] = d_imagePositionsTrunc[j - numberGlobalAtoms][2];
                jatomId      = d_imageIdsTrunc[j - numberGlobalAtoms];
              }

            const double dist = atomCoori.distance(atomCoorj);
            if (dist < d_nearestAtomDistances[i] && j != i)
              {
                d_nearestAtomDistances[i] = dist;
                d_nearestAtomIds[i]       = jatomId;
              }
          }
      }
    d_minDist = *std::min_element(d_nearestAtomDistances.begin(),
                                  d_nearestAtomDistances.end());
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Minimum distance between atoms: " << d_minDist << std::endl;

    AssertThrow(
      d_minDist >= 0.1,
      dealii::ExcMessage(
        "DFT-FE Error: Minimum distance between atoms is less than 0.1 Bohr."));
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::moveMeshToAtoms(
    dealii::Triangulation<3, 3> &triangulationMove,
    dealii::Triangulation<3, 3> &triangulationSerial,
    bool                         reuseClosestTriaVertices,
    bool                         moveSubdivided)
  {
    dealii::ConditionalOStream pcout_movemesh(
      std::cout,
      (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0));
    dealii::TimerOutput timer_movemesh(mpi_communicator,
                                       pcout_movemesh,
                                       d_dftParamsPtr->reproducible_output ||
                                           d_dftParamsPtr->verbosity < 4 ?
                                         dealii::TimerOutput::never :
                                         dealii::TimerOutput::summary,
                                       dealii::TimerOutput::wall_times);

    meshMovementGaussianClass gaussianMove(d_mpiCommParent,
                                           mpi_communicator,
                                           *d_dftParamsPtr);
    gaussianMove.init(triangulationMove,
                      triangulationSerial,
                      d_domainBoundingVectors);

    const dftfe::uInt numberGlobalAtoms = atomLocations.size();
    const dftfe::uInt numberImageAtoms  = d_imageIdsTrunc.size();

    std::vector<dealii::Point<3>> atomPoints;
    d_atomLocationsAutoMesh.resize(numberGlobalAtoms,
                                   std::vector<double>(3, 0.0));
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
      {
        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[iAtom][2];
        atomCoor[1] = atomLocations[iAtom][3];
        atomCoor[2] = atomLocations[iAtom][4];
        atomPoints.push_back(atomCoor);
        for (dftfe::uInt j = 0; j < 3; j++)
          d_atomLocationsAutoMesh[iAtom][j] = atomCoor[j];
      }

    std::vector<dealii::Point<3>>             closestTriaVertexToAtomsLocation;
    std::vector<dealii::Tensor<1, 3, double>> dispClosestTriaVerticesToAtoms;

    timer_movemesh.enter_subsection(
      "move mesh to atoms: find closest vertices");
    if (reuseClosestTriaVertices)
      {
        closestTriaVertexToAtomsLocation = d_closestTriaVertexToAtomsLocation;
        dispClosestTriaVerticesToAtoms   = d_dispClosestTriaVerticesToAtoms;
      }
    else
      {
        gaussianMove.findClosestVerticesToDestinationPoints(
          atomPoints,
          closestTriaVertexToAtomsLocation,
          dispClosestTriaVerticesToAtoms);
      }
    timer_movemesh.leave_subsection(
      "move mesh to atoms: find closest vertices");


    timer_movemesh.enter_subsection("move mesh to atoms: move mesh");
    // add control point locations and displacements corresponding to images
    if (!reuseClosestTriaVertices)
      for (dftfe::uInt iImage = 0; iImage < numberImageAtoms; iImage++)
        {
          dealii::Point<3> imageCoor;
          dealii::Point<3> correspondingAtomCoor;

          imageCoor[0]             = d_imagePositionsTrunc[iImage][0];
          imageCoor[1]             = d_imagePositionsTrunc[iImage][1];
          imageCoor[2]             = d_imagePositionsTrunc[iImage][2];
          const dftfe::Int atomId  = d_imageIdsTrunc[iImage];
          correspondingAtomCoor[0] = atomLocations[atomId][2];
          correspondingAtomCoor[1] = atomLocations[atomId][3];
          correspondingAtomCoor[2] = atomLocations[atomId][4];


          const dealii::Point<3> temp =
            closestTriaVertexToAtomsLocation[atomId] +
            (imageCoor - correspondingAtomCoor);
          closestTriaVertexToAtomsLocation.push_back(temp);
          dispClosestTriaVerticesToAtoms.push_back(
            dispClosestTriaVerticesToAtoms[atomId]);
        }

    d_closestTriaVertexToAtomsLocation = closestTriaVertexToAtomsLocation;
    d_dispClosestTriaVerticesToAtoms   = dispClosestTriaVerticesToAtoms;
    d_gaussianMovementAtomsNetDisplacements.resize(numberGlobalAtoms);
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
      d_gaussianMovementAtomsNetDisplacements[iAtom] = 0.0;

    d_controlPointLocationsCurrentMove.clear();
    d_gaussianConstantsAutoMesh.clear();
    d_flatTopWidthsAutoMeshMove.clear();
    d_gaussianConstantsForce.clear();
    d_generatorFlatTopWidths.clear();
    d_gaussianConstantsAutoMesh.resize(numberGlobalAtoms);
    d_flatTopWidthsAutoMeshMove.resize(numberGlobalAtoms);
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
      {
        d_flatTopWidthsAutoMeshMove[iAtom] =
          d_dftParamsPtr->useFlatTopGenerator ?
            std::min(0.5, 0.9 * d_nearestAtomDistances[iAtom] / 2.0 - 0.2) :
            0.0;
        d_gaussianConstantsAutoMesh[iAtom] =
          d_dftParamsPtr->reproducible_output ?
            1 / std::sqrt(0.5) :
            (std::min(0.9 * d_nearestAtomDistances[iAtom] / 2.0, 2.0) -
             d_flatTopWidthsAutoMeshMove[iAtom]);
      }

    std::vector<double> gaussianConstantsAutoMesh;
    std::vector<double> flatTopWidths;
    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms + numberImageAtoms;
         iAtom++)
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
            atomCoor[0] = d_imagePositionsTrunc[iAtom - numberGlobalAtoms][0];
            atomCoor[1] = d_imagePositionsTrunc[iAtom - numberGlobalAtoms][1];
            atomCoor[2] = d_imagePositionsTrunc[iAtom - numberGlobalAtoms][2];
          }
        d_controlPointLocationsCurrentMove.push_back(atomCoor);
      }

    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms + numberImageAtoms;
         iAtom++)
      {
        dftfe::Int atomId;
        if (iAtom < numberGlobalAtoms)
          atomId = iAtom;
        else
          atomId = d_imageIdsTrunc[iAtom - numberGlobalAtoms];

        gaussianConstantsAutoMesh.push_back(
          d_gaussianConstantsAutoMesh[atomId]);
        flatTopWidths.push_back(d_flatTopWidthsAutoMeshMove[atomId]);
      }

    const std::pair<bool, double> meshQualityMetrics =
      gaussianMove.moveMesh(closestTriaVertexToAtomsLocation,
                            dispClosestTriaVerticesToAtoms,
                            gaussianConstantsAutoMesh,
                            flatTopWidths,
                            moveSubdivided);

    timer_movemesh.leave_subsection("move mesh to atoms: move mesh");

    AssertThrow(
      !meshQualityMetrics.first,
      dealii::ExcMessage(
        "Negative jacobian created after moving closest nodes to atoms. Suggestion: increase refinement near atoms"));

    if (!reuseClosestTriaVertices)
      d_autoMeshMaxJacobianRatio = meshQualityMetrics.second;

    if (d_dftParamsPtr->verbosity >= 1 && !moveSubdivided)
      pcout
        << "Mesh quality check for Auto mesh after mesh movement, maximum jacobian ratio: "
        << meshQualityMetrics.second << std::endl;

    d_gaussianConstantsForce.resize(numberGlobalAtoms);
    d_generatorFlatTopWidths = d_flatTopWidthsAutoMeshMove;

    for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
      d_gaussianConstantsForce[iAtom] =
        d_dftParamsPtr->reproducible_output ?
          1 / std::sqrt(5.0) :
          (d_dftParamsPtr->useFlatTopGenerator ?
             d_generatorFlatTopWidths[iAtom] + 0.4 :
             (std::min(d_nearestAtomDistances[iAtom] / 2.0 - 0.3,
                       d_dftParamsPtr->gaussianConstantForce)));
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::calculateSmearedChargeWidths()
  {
    d_smearedChargeWidths.clear();

    const dftfe::uInt numberGlobalAtoms = atomLocations.size();

    d_smearedChargeWidths.resize(numberGlobalAtoms);

    if (d_dftParamsPtr->smearedNuclearCharges)
      {
        for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
          d_smearedChargeWidths[iAtom] = 0.7;

        for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
          {
            while (d_nearestAtomDistances[iAtom] <
                   (d_smearedChargeWidths[iAtom] +
                    d_smearedChargeWidths[d_nearestAtomIds[iAtom]] + 0.3))
              {
                d_smearedChargeWidths[iAtom] -= 0.05;
                d_smearedChargeWidths[d_nearestAtomIds[iAtom]] -= 0.05;
              }
          }

        for (dftfe::uInt iAtom = 0; iAtom < numberGlobalAtoms; iAtom++)
          if (d_dftParamsPtr->verbosity >= 5)
            pcout << "iAtom: " << iAtom
                  << ", Smeared charge width: " << d_smearedChargeWidths[iAtom]
                  << std::endl;
      }
  }
#include "dft.inst.cc"
} // namespace dftfe
