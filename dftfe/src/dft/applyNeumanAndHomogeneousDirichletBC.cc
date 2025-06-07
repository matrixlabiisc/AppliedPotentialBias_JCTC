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
// @author  Kartick
//
#include <dft.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::applyNeumanAndHomogeneousDirichletBC(
    const dealii::DoFHandler<3>             &_dofHandler,
    const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
    dealii::AffineConstraints<double>       &constraintMatrix)

  {
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(_dofHandler,
                                                    locallyRelevantDofs);

    const dftfe::uInt vertices_per_cell =
      dealii::GeometryInfo<3>::vertices_per_cell;
    const dftfe::uInt dofs_per_cell  = _dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const dftfe::uInt dofs_per_face  = _dofHandler.get_fe().dofs_per_face;

    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

    std::vector<bool> dofs_touched(_dofHandler.n_dofs(), false);
    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  _dofHandler.begin_active(),
                                                endc = _dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(cellGlobalDofIndices);
          for (dftfe::uInt iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              const dftfe::uInt boundaryId = cell->face(iFace)->boundary_id();
              if (boundaryId == 0)
                {
                  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                  for (dftfe::uInt iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      if (!onlyHangingNodeConstraints.is_constrained(nodeId))
                        {
                          bool             boundaryDirchilet = true;
                          dealii::Point<3> node;
                          node = d_supportPointsPRefined[nodeId];
                          if (std::fabs(node[2] -
                                        d_domainBoundingVectors[2][2] / 2) <
                              1E-8)
                            boundaryDirchilet = false;

                          // if (d_dftParamsPtr->solverMode != "MD" &&
                          //     !boundaryDirchilet)
                          //   std::cout << "Neumann BC at: " << node[0] << " "
                          //             << node[1] << " " << node[2] <<
                          //             std::endl;
                          if (boundaryDirchilet)
                            {
                              constraintMatrix.add_line(nodeId);
                              constraintMatrix.set_inhomogeneity(nodeId, 0.0);
                            }
                        } // non-hanging node check
                    }     // Face dof loop
                }         // non-periodic boundary id
            }             // Face loop
        }                 // cell locally owned
  }
} // namespace dftfe
