// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das(2017)
//

#include <dft.h>
#include <force.h>


namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::createBinObjectsForce(
    const dealii::DoFHandler<3>             &dofHandler,
    const dealii::DoFHandler<3>             &dofHandlerForce,
    const dealii::AffineConstraints<double> &hangingPlusPBCConstraints,
    const vselfBinsManager                  &vselfBinsManager,
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      &cellsVselfBallsDofHandler,
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
      &cellsVselfBallsDofHandlerForce,
    std::vector<std::map<dealii::CellId, dftfe::uInt>>
                                       &cellsVselfBallsClosestAtomIdDofHandler,
    std::map<dftfe::uInt, dftfe::uInt> &AtomIdBinIdLocalDofHandler,
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>>
      &cellFacesVselfBallSurfacesDofHandler,
    std::vector<std::map<dealii::DoFHandler<3>::active_cell_iterator,
                         std::vector<dftfe::uInt>>>
      &cellFacesVselfBallSurfacesDofHandlerForce)
  {
    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt dofs_per_cell  = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt dofs_per_face  = dofHandler.get_fe().dofs_per_face;
    const dftfe::uInt numberBins     = vselfBinsManager.getAtomIdsBins().size();
    // clear exisitng data
    cellsVselfBallsDofHandler.clear();
    cellsVselfBallsDofHandlerForce.clear();
    cellFacesVselfBallSurfacesDofHandler.clear();
    cellFacesVselfBallSurfacesDofHandlerForce.clear();
    cellsVselfBallsClosestAtomIdDofHandler.clear();
    AtomIdBinIdLocalDofHandler.clear();
    // resize
    cellsVselfBallsDofHandler.resize(numberBins);
    cellsVselfBallsDofHandlerForce.resize(numberBins);
    cellFacesVselfBallSurfacesDofHandler.resize(numberBins);
    cellFacesVselfBallSurfacesDofHandlerForce.resize(numberBins);
    cellsVselfBallsClosestAtomIdDofHandler.resize(numberBins);

    for (dftfe::uInt iBin = 0; iBin < numberBins; ++iBin)
      {
        const std::map<dealii::types::global_dof_index, dftfe::Int>
          &boundaryNodeMap = vselfBinsManager.getBoundaryFlagsBins()[iBin];
        const std::map<dealii::types::global_dof_index, dftfe::Int>
          &closestAtomBinMap = vselfBinsManager.getClosestAtomIdsBins()[iBin];
        dealii::DoFHandler<3>::active_cell_iterator cell =
          dofHandler.begin_active();
        dealii::DoFHandler<3>::active_cell_iterator endc = dofHandler.end();
        dealii::DoFHandler<3>::active_cell_iterator cellForce =
          dofHandlerForce.begin_active();
        for (; cell != endc; ++cell, ++cellForce)
          {
            if (cell->is_locally_owned())
              {
                std::vector<dftfe::uInt> dirichletFaceIds;
                std::vector<dftfe::uInt>
                  faceIdsWithAtleastOneSolvedNonHangingNode;
                std::vector<dftfe::uInt> allFaceIdsOfCell;
                dftfe::uInt              closestAtomIdSum          = 0;
                dftfe::uInt              closestAtomId             = 0;
                dftfe::uInt              nonHangingNodeIdCountCell = 0;
                for (dftfe::uInt iFace = 0; iFace < faces_per_cell; ++iFace)
                  {
                    dftfe::Int dirichletDofCount         = 0;
                    bool       isSolvedDofPresent        = false;
                    dftfe::Int nonHangingNodeIdCountFace = 0;
                    std::vector<dealii::types::global_dof_index>
                      iFaceGlobalDofIndices(dofs_per_face);
                    cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                    for (dftfe::uInt iFaceDof = 0; iFaceDof < dofs_per_face;
                         ++iFaceDof)
                      {
                        const dealii::types::global_dof_index nodeId =
                          iFaceGlobalDofIndices[iFaceDof];
                        if (!hangingPlusPBCConstraints.is_constrained(nodeId))
                          {
                            Assert(boundaryNodeMap.find(nodeId) !=
                                     boundaryNodeMap.end(),
                                   dealii::ExcMessage("BUG"));
                            Assert(closestAtomBinMap.find(nodeId) !=
                                     closestAtomBinMap.end(),
                                   dealii::ExcMessage("BUG"));

                            if (boundaryNodeMap.find(nodeId)->second != -1)
                              isSolvedDofPresent = true;
                            else
                              dirichletDofCount +=
                                boundaryNodeMap.find(nodeId)->second;

                            closestAtomId =
                              closestAtomBinMap.find(nodeId)->second;
                            closestAtomIdSum += closestAtomId;
                            nonHangingNodeIdCountCell++;
                            nonHangingNodeIdCountFace++;
                          } // non-hanging node check
                        else
                          {
                            const std::vector<
                              std::pair<dealii::types::global_dof_index,
                                        double>> *rowData =
                              hangingPlusPBCConstraints.get_constraint_entries(
                                nodeId);
                            for (dftfe::uInt j = 0; j < rowData->size(); ++j)
                              {
                                if (d_dftParams
                                      .createConstraintsFromSerialDofhandler)
                                  {
                                    if (boundaryNodeMap.find(
                                          (*rowData)[j].first) ==
                                        boundaryNodeMap.end())
                                      continue;
                                  }
                                else
                                  {
                                    Assert(boundaryNodeMap.find(
                                             (*rowData)[j].first) !=
                                             boundaryNodeMap.end(),
                                           dealii::ExcMessage("BUG"));
                                  }

                                if (boundaryNodeMap.find((*rowData)[j].first)
                                      ->second != -1)
                                  isSolvedDofPresent = true;
                                else
                                  dirichletDofCount +=
                                    boundaryNodeMap.find((*rowData)[j].first)
                                      ->second;
                              }
                          }

                      } // Face dof loop

                    if (isSolvedDofPresent)
                      {
                        faceIdsWithAtleastOneSolvedNonHangingNode.push_back(
                          iFace);
                      }
                    if (dirichletDofCount < 0)
                      {
                        dirichletFaceIds.push_back(iFace);
                      }
                    allFaceIdsOfCell.push_back(iFace);

                  } // Face loop

                // fill the target objects
                if (faceIdsWithAtleastOneSolvedNonHangingNode.size() > 0)
                  {
                    if (!(closestAtomIdSum ==
                          closestAtomId * nonHangingNodeIdCountCell))
                      {
                        std::cout << "closestAtomIdSum: " << closestAtomIdSum
                                  << ", closestAtomId: " << closestAtomId
                                  << ", nonHangingNodeIdCountCell: "
                                  << nonHangingNodeIdCountCell << std::endl;
                      }
                    AssertThrow(
                      closestAtomIdSum ==
                        closestAtomId * nonHangingNodeIdCountCell,
                      dealii::ExcMessage(
                        "cell dofs on vself ball surface have different closest atom ids, remedy- increase separation between vself balls"));

                    cellsVselfBallsDofHandler[iBin].push_back(cell);
                    cellsVselfBallsDofHandlerForce[iBin].push_back(cellForce);
                    cellsVselfBallsClosestAtomIdDofHandler[iBin][cell->id()] =
                      closestAtomId;
                    AtomIdBinIdLocalDofHandler[closestAtomId] = iBin;
                    if (dirichletFaceIds.size() > 0)
                      {
                        cellFacesVselfBallSurfacesDofHandler[iBin][cell] =
                          dirichletFaceIds;
                        cellFacesVselfBallSurfacesDofHandlerForce
                          [iBin][cellForce] = dirichletFaceIds;
                      }
                  }
              } // cell locally owned
          }     // cell loop
      }         // Bin loop

    d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro.clear();
    dealii::DoFHandler<3>::active_cell_iterator cell =
      dftPtr->d_dofHandlerRhoNodal.begin_active();
    dealii::DoFHandler<3>::active_cell_iterator endc =
      dftPtr->d_dofHandlerRhoNodal.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro[cell->id()] =
          cell;
  }
//
#include "force.inst.cc"
} // namespace dftfe
