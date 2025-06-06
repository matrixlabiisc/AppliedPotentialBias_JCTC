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
//================================================================================================================================================
//================================================================================================================================================
//    This is the source file for generating and communicating mapping tables
//    between real space points and their symmetry transformed points.
//	            Only relevant for calculations using multiple k-points and when
// USE GROUP SYMMETRY = true
//
//                                              Author : Krishnendu Ghosh,
//                                              krisg@umich.edu
//
//================================================================================================================================================
//================================================================================================================================================
//
#include "../../include/dft.h"
#include "../../include/symmetry.h"
#include "symmetrizeRho.cc"
//
namespace dftfe
{
  //================================================================================================================================================
  //							Class constructor
  //================================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  symmetryClass<memorySpace>::symmetryClass(dftClass<memorySpace> *_dftPtr,
                                            const MPI_Comm &mpi_comm_parent,
                                            const MPI_Comm &mpi_comm_domain,
                                            const MPI_Comm &_interpoolcomm)
    : dftPtr(_dftPtr)
    , FE(dealii::QGaussLobatto<1>(
        _dftPtr->getParametersObject().finiteElementPolynomialOrder + 1))
    , d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , interpoolcomm(_interpoolcomm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
  {}
  //================================================================================================================================================
  //					Wiping out mapping tables; needed between relaxation steps
  //================================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  symmetryClass<memorySpace>::clearMaps()
  {
    mappedGroup.clear();
    mappedGroupSend0.clear();
    mappedGroupSend1.clear();
    mappedGroupSend2.clear();
    mappedGroupRecvd0.clear();
    mappedGroupRecvd2.clear();
    mappedGroupRecvd1.clear();
    send_buf_size.clear();
    recv_buf_size.clear();
    rhoRecvd.clear();
    groupOffsets.clear();

    bool isGradDensityDataRequired =
      (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
         ->getDensityBasedFamilyType() == densityFamilyType::GGA);
    ;

    if (isGradDensityDataRequired)
      gradRhoRecvd.clear();
  }
  //================================================================================================================================================
  //================================================================================================================================================
  //			     The following is the main driver routine to generate and
  // communicate mapping tables
  //================================================================================================================================================
  //================================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  symmetryClass<memorySpace>::initSymmetry()
  {
    //
    dealii::QGauss<3> quadrature(
      dftPtr->getParametersObject().densityQuadratureRule);
    dealii::FEValues<3>  fe_values(dftPtr->FEEigen,
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values |
                                    dealii::update_quadrature_points);
    const dftfe::uInt    num_quad_points = quadrature.size();
    dealii::Point<3>     p, ptemp, p0;
    dealii::MappingQ1<3> mapping;
    char                 buffer[100];
    //
    std::pair<typename dealii::parallel::distributed::Triangulation<
                3>::active_cell_iterator,
              dealii::Point<3>>
                                                            mapped_cell;
    std::tuple<dftfe::Int, std::vector<double>, dftfe::Int> tupleTemp;
    std::tuple<dftfe::Int, dftfe::Int, dftfe::Int>          tupleTemp2;
    std::map<dealii::CellId, dftfe::Int>                    groupId;
    std::vector<double>                                     mappedPoint(3);
    std::vector<dftfe::Int> countGroupPerProc(dftPtr->n_mpi_processes),
      countPointPerProc(dftPtr->n_mpi_processes);
    std::vector<std::vector<dftfe::Int>> countPointsPerGroupPerProc(
      dftPtr->n_mpi_processes);
    std::vector<std::vector<dftfe::Int>> tailofGroup(dftPtr->n_mpi_processes);
    //
    dftfe::uInt                          count = 0, cell_id = 0, ownerProcId;
    dftfe::uInt                          mappedPointId;
    std::map<dealii::CellId, dftfe::Int> globalCellId_parallel;
    //
    clearMaps();
    //================================================================================================================================================
    //							Allocate memory for the mapping tables
    //================================================================================================================================================
    mappedGroup.resize(numSymm);
    mappedGroupSend0.resize(numSymm);
    mappedGroupSend1.resize(numSymm);
    mappedGroupSend2.resize(numSymm);
    mappedGroupRecvd0.resize(numSymm);
    mappedGroupRecvd2.resize(numSymm);
    mappedGroupRecvd1.resize(numSymm);
    send_buf_size.resize(numSymm);
    recv_buf_size.resize(numSymm);
    rhoRecvd.resize(numSymm);
    groupOffsets.resize(numSymm);

    bool isGradDensityDataRequired =
      (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
         ->getDensityBasedFamilyType() == densityFamilyType::GGA);
    ;

    if (isGradDensityDataRequired)
      gradRhoRecvd.resize(numSymm);
    //
    const dealii::parallel::distributed::Triangulation<3> &triangulationSer =
      (dftPtr->d_mesh).getSerialMeshUnmoved();
    typename dealii::parallel::distributed::Triangulation<
      3>::active_cell_iterator cellTemp = triangulationSer.begin_active(),
                               endcTemp = triangulationSer.end();
    for (; cellTemp != endcTemp; ++cellTemp)
      {
        globalCellId[cellTemp->id()] = cell_id;
        cell_id++;
      }
    //
    ownerProcGlobal.resize(cell_id);
    std::vector<dftfe::Int> ownerProc(cell_id, 0);
    for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
      {
        mappedGroup[iSymm] = std::vector<
          std::vector<std::tuple<dftfe::Int, dftfe::Int, dftfe::Int>>>(cell_id);
        mappedGroupSend0[iSymm] =
          std::vector<std::vector<std::vector<dftfe::Int>>>(cell_id);
        mappedGroupSend2[iSymm] =
          std::vector<std::vector<std::vector<dftfe::Int>>>(cell_id);
        mappedGroupSend1[iSymm] =
          std::vector<std::vector<std::vector<std::vector<double>>>>(cell_id);
        mappedGroupRecvd0[iSymm] =
          std::vector<std::vector<dftfe::Int>>(cell_id);
        mappedGroupRecvd2[iSymm] =
          std::vector<std::vector<dftfe::Int>>(cell_id);
        mappedGroupRecvd1[iSymm] =
          std::vector<std::vector<std::vector<double>>>(cell_id);
        send_buf_size[iSymm] =
          std::vector<std::vector<std::vector<dftfe::Int>>>(cell_id);
        recv_buf_size[iSymm] =
          std::vector<std::vector<std::vector<dftfe::Int>>>(cell_id);
        rhoRecvd[iSymm] =
          std::vector<std::vector<std::vector<double>>>(cell_id);
        groupOffsets[iSymm] =
          std::vector<std::vector<std::vector<dftfe::Int>>>(cell_id);
        if (isGradDensityDataRequired)
          gradRhoRecvd[iSymm] =
            std::vector<std::vector<std::vector<double>>>(cell_id);
      }
    //================================================================================================================================================
    //					     Create local and global maps to locate cells on their
    // hosting processors
    //================================================================================================================================================
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = (dftPtr->dofHandlerEigen).begin_active(),
      endc = (dftPtr->dofHandlerEigen).end();
    dealii::Tensor<1, 3, double> center_diff;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            cellTemp = triangulationSer.begin_active(),
            endcTemp = triangulationSer.end();
            for (; cellTemp != endcTemp; ++cellTemp)
              {
                center_diff  = cellTemp->center() - cell->center();
                double pnorm = center_diff[0] * center_diff[0] +
                               center_diff[1] * center_diff[1] +
                               center_diff[2] * center_diff[2];
                if (pnorm < 1.0E-5)
                  {
                    globalCellId_parallel[cell->id()] =
                      globalCellId[cellTemp->id()];
                    break;
                  }
              }
            dealIICellId[globalCellId_parallel[cell->id()]] = cell;
            ownerProc[globalCellId_parallel[cell->id()]]    = this_mpi_process;
          }
      }
    //
    MPI_Allreduce(&ownerProc[0],
                  &ownerProcGlobal[0],
                  cell_id,
                  dftfe::dataTypes::mpi_type_id(ownerProc.data()),
                  MPI_SUM,
                  mpi_communicator);
    //================================================================================================================================================
    //			Now enter each local cell to apply each of the symmetry operations
    // on the quad points relevant to the cell. 			Then find out which cell
    // and processor the transformed point belongs to. 			Next create maps of
    // points based on symmetry operation, cell address, and processor id.
    //================================================================================================================================================
    cell = (dftPtr->dofHandlerEigen).begin_active();
    endc = (dftPtr->dofHandlerEigen).end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
              {
                mappedGroup[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::tuple<dftfe::Int, dftfe::Int, dftfe::Int>>(
                    num_quad_points);
                mappedGroupRecvd1[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<double>>(3);
                rhoRecvd[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<double>>(dftPtr->n_mpi_processes);
              }
            for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
              {
                count = 0;
                std::fill(countGroupPerProc.begin(),
                          countGroupPerProc.end(),
                          0);
                //
                send_buf_size[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<dftfe::Int>>(dftPtr->n_mpi_processes);
                //
                mappedGroupSend0[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<dftfe::Int>>(dftPtr->n_mpi_processes);
                mappedGroupSend2[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<dftfe::Int>>(dftPtr->n_mpi_processes);
                mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<std::vector<double>>>(
                    dftPtr->n_mpi_processes);
                //
                for (dftfe::Int i = 0; i < dftPtr->n_mpi_processes; ++i)
                  {
                    send_buf_size[iSymm][globalCellId_parallel[cell->id()]][i] =
                      std::vector<dftfe::Int>(3, 0);
                    mappedGroupSend1[iSymm][globalCellId_parallel[cell->id()]]
                                    [i] = std::vector<std::vector<double>>(3);
                  }
                recv_buf_size[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<dftfe::Int>>(3);
                recv_buf_size[iSymm][globalCellId_parallel[cell->id()]][0] =
                  std::vector<dftfe::Int>(dftPtr->n_mpi_processes);
                recv_buf_size[iSymm][globalCellId_parallel[cell->id()]][1] =
                  std::vector<dftfe::Int>(dftPtr->n_mpi_processes);
                recv_buf_size[iSymm][globalCellId_parallel[cell->id()]][2] =
                  std::vector<dftfe::Int>(dftPtr->n_mpi_processes);
                //
                groupOffsets[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<dftfe::Int>>(3);
                groupOffsets[iSymm][globalCellId_parallel[cell->id()]][0] =
                  std::vector<dftfe::Int>(dftPtr->n_mpi_processes);
                groupOffsets[iSymm][globalCellId_parallel[cell->id()]][1] =
                  std::vector<dftfe::Int>(dftPtr->n_mpi_processes);
                groupOffsets[iSymm][globalCellId_parallel[cell->id()]][2] =
                  std::vector<dftfe::Int>(dftPtr->n_mpi_processes);
                //
                fe_values.reinit(cell);
                for (dftfe::uInt q_point = 0; q_point < num_quad_points;
                     ++q_point)
                  {
                    p  = fe_values.quadrature_point(q_point);
                    p0 = crys2cart(p, -1);
                    //
                    ptemp[0] = p0[0] * symmMat[iSymm][0][0] +
                               p0[1] * symmMat[iSymm][0][1] +
                               p0[2] * symmMat[iSymm][0][2];
                    ptemp[1] = p0[0] * symmMat[iSymm][1][0] +
                               p0[1] * symmMat[iSymm][1][1] +
                               p0[2] * symmMat[iSymm][1][2];
                    ptemp[2] = p0[0] * symmMat[iSymm][2][0] +
                               p0[1] * symmMat[iSymm][2][1] +
                               p0[2] * symmMat[iSymm][2][2];
                    //
                    ptemp[0] = ptemp[0] + translation[iSymm][0];
                    ptemp[1] = ptemp[1] + translation[iSymm][1];
                    ptemp[2] = ptemp[2] + translation[iSymm][2];
                    //
                    for (dftfe::uInt i = 0; i < 3; ++i)
                      {
                        while (ptemp[i] > 0.5)
                          ptemp[i] = ptemp[i] - 1.0;
                        while (ptemp[i] < -0.5)
                          ptemp[i] = ptemp[i] + 1.0;
                      }
                    p = crys2cart(ptemp, 1);
                    //
                    if (q_point == 0)
                      mapped_cell =
                        dealii::GridTools::find_active_cell_around_point(
                          mapping, triangulationSer, p);
                    else
                      {
                        double dist = 1.0E+06;
                        try
                          {
                            dealii::Point<3> p_cell =
                              mapping.transform_real_to_unit_cell(
                                mapped_cell.first, p);
                            dist =
                              dealii::GeometryInfo<3>::distance_to_unit_cell(
                                p_cell);
                            if (dist < 1.0E-10)
                              mapped_cell.second = p_cell;
                          }
                        catch (dealii::MappingQ1<3>::ExcTransformationFailed)
                          {}
                        if (dist > 1.0E-10)
                          mapped_cell =
                            dealii::GridTools::find_active_cell_around_point(
                              mapping, triangulationSer, p);
                      }
                    dealii::Point<3> pointTemp = mapped_cell.second;
                    //
                    mappedPoint[0] = pointTemp.operator()(0);
                    mappedPoint[1] = pointTemp.operator()(1);
                    mappedPoint[2] = pointTemp.operator()(2);
                    //
                    ownerProcId =
                      ownerProcGlobal[globalCellId[mapped_cell.first->id()]];
                    //
                    tupleTemp =
                      std::make_tuple(ownerProcId, mappedPoint, q_point);
                    cellMapTable[mapped_cell.first->id()].push_back(tupleTemp);
                    //
                    //
                    send_buf_size[iSymm][globalCellId_parallel
                                           [cell->id()]][ownerProcId][1] =
                      send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                   [ownerProcId][1] +
                      1;
                    send_buf_size[iSymm][globalCellId_parallel
                                           [cell->id()]][ownerProcId][2] =
                      send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                   [ownerProcId][2] +
                      1;
                    //
                  }
                std::fill(countPointPerProc.begin(),
                          countPointPerProc.end(),
                          0);
                for (std::map<dealii::CellId,
                              std::vector<std::tuple<dftfe::Int,
                                                     std::vector<double>,
                                                     dftfe::Int>>>::iterator
                       iter = cellMapTable.begin();
                     iter != cellMapTable.end();
                     ++iter)
                  {
                    std::vector<
                      std::tuple<dftfe::Int, std::vector<double>, dftfe::Int>>
                                   value = iter->second;
                    dealii::CellId key   = iter->first;
                    ownerProcId          = ownerProcGlobal[globalCellId[key]];
                    mappedGroupSend0[iSymm][globalCellId_parallel[cell->id()]]
                                    [ownerProcId]
                                      .push_back(globalCellId[key]);
                    mappedGroupSend2[iSymm][globalCellId_parallel[cell->id()]]
                                    [ownerProcId]
                                      .push_back(value.size());
                    send_buf_size[iSymm][globalCellId_parallel
                                           [cell->id()]][ownerProcId][0] =
                      send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                   [ownerProcId][0] +
                      1;
                    //
                    for (dftfe::uInt i = 0; i < value.size(); ++i)
                      {
                        mappedPoint        = std::get<1>(value[i]);
                        dftfe::Int q_point = std::get<2>(value[i]);
                        //
                        tupleTemp2 =
                          std::make_tuple(ownerProcId,
                                          0,
                                          countPointPerProc[ownerProcId]);
                        mappedGroup[iSymm][globalCellId_parallel[cell->id()]]
                                   [q_point] = tupleTemp2;
                        countPointPerProc[ownerProcId] += 1;
                        //
                        mappedGroupSend1[iSymm]
                                        [globalCellId_parallel[cell->id()]]
                                        [ownerProcId][0]
                                          .push_back(mappedPoint[0]);
                        mappedGroupSend1[iSymm]
                                        [globalCellId_parallel[cell->id()]]
                                        [ownerProcId][1]
                                          .push_back(mappedPoint[1]);
                        mappedGroupSend1[iSymm]
                                        [globalCellId_parallel[cell->id()]]
                                        [ownerProcId][2]
                                          .push_back(mappedPoint[2]);
                      }
                  }
                cellMapTable.clear();
              } // symmetry loop
          }     // is cell locally owned condition
      }         // cell loop
    //
    MPI_Barrier(mpi_communicator);
    //================================================================================================================================================
    //			      Now first prepare the flattened sending and receiving vectors
    // and then MPI gather. 		     The essential idea here is that each
    // processor collects the transformed points from all other processors. In
    // symmetrizeRho.cc each processor locally computes density on its
    // transformed points and 			     then scatters them back to the
    // processors from which the points came from.
    //================================================================================================================================================
    int recvDataSize0 = 0, recvDataSize1 = 0, send_size0, send_size1,
        send_size2;
    std::vector<dftfe::Int>          send_data0, send_data2, send_data3;
    std::vector<std::vector<double>> send_data1;
    std::vector<double>              send_data, recvdData;
    mpi_offsets0.resize(dftPtr->n_mpi_processes, 0);
    mpi_offsets1.resize(dftPtr->n_mpi_processes, 0);
    mpiGrad_offsets1.resize(dftPtr->n_mpi_processes, 0);
    recv_size0.resize(dftPtr->n_mpi_processes, 0);
    recv_size1.resize(dftPtr->n_mpi_processes, 0);
    recvGrad_size1.resize(dftPtr->n_mpi_processes, 0);
    recvdData1.resize(3);
    send_data1.resize(3);
    //
    for (dftfe::uInt proc = 0; proc < dftPtr->n_mpi_processes; ++proc)
      {
        send_size1 = 0;
        send_size0 = 0;
        cell       = (dftPtr->dofHandlerEigen).begin_active();
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                for (dftfe::uInt iSymm = 0; iSymm < numSymm; iSymm++)
                  {
                    for (dftfe::uInt iPoint = 0;
                         iPoint <
                         send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                      [proc][1];
                         ++iPoint)
                      {
                        //
                        send_data1[0].push_back(
                          mappedGroupSend1[iSymm]
                                          [globalCellId_parallel[cell->id()]]
                                          [proc][0][iPoint]);
                        send_data1[1].push_back(
                          mappedGroupSend1[iSymm]
                                          [globalCellId_parallel[cell->id()]]
                                          [proc][1][iPoint]);
                        send_data1[2].push_back(
                          mappedGroupSend1[iSymm]
                                          [globalCellId_parallel[cell->id()]]
                                          [proc][2][iPoint]);
                      }
                    send_size1 +=
                      send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                   [proc][1];
                    //
                    for (dftfe::uInt i = 0;
                         i <
                         send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                      [proc][0];
                         ++i)
                      {
                        send_data0.push_back(
                          mappedGroupSend0[iSymm]
                                          [globalCellId_parallel[cell->id()]]
                                          [proc][i]);
                        send_data2.push_back(
                          mappedGroupSend2[iSymm]
                                          [globalCellId_parallel[cell->id()]]
                                          [proc][i]);
                        send_data3.push_back(iSymm);
                      }
                    send_size0 +=
                      send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                   [proc][0];
                    //
                    rhoRecvd[iSymm][globalCellId_parallel[cell->id()]][proc]
                      .resize(
                        (1 + dftPtr->getParametersObject().spinPolarized) *
                        send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                     [proc][1]); // to be used later to recv
                                                 // symmetrized rho
                  }
              }
          }
        //
        //
        MPI_Gather(&send_size0,
                   1,
                   dftfe::dataTypes::mpi_type_id(&send_size0),
                   &(recv_size0[0]),
                   1,
                   dftfe::dataTypes::mpi_type_id(recv_size0.data()),
                   proc,
                   mpi_communicator);
        MPI_Gather(&send_size1,
                   1,
                   dftfe::dataTypes::mpi_type_id(&send_size1),
                   &(recv_size1[0]),
                   1,
                   dftfe::dataTypes::mpi_type_id(recv_size1.data()),
                   proc,
                   mpi_communicator);
        //
        if (proc == this_mpi_process)
          {
            for (dftfe::Int i = 1; i < dftPtr->n_mpi_processes; i++)
              {
                mpi_offsets0[i] = recv_size0[i - 1] + mpi_offsets0[i - 1];
                mpi_offsets1[i] = recv_size1[i - 1] + mpi_offsets1[i - 1];
              }
            //
            recvDataSize0 =
              std::accumulate(&recv_size0[0],
                              &recv_size0[dftPtr->n_mpi_processes],
                              0);
            recvdData0.resize(recvDataSize0, 0);
            recvdData2.resize(recvDataSize0, 0);
            recvdData3.resize(recvDataSize0, 0);
            //
            recvDataSize1 =
              std::accumulate(&recv_size1[0],
                              &recv_size1[dftPtr->n_mpi_processes],
                              0);
            recvdData.resize(recvDataSize1, 0.0);
            for (dftfe::uInt ipol = 0; ipol < 3; ++ipol)
              recvdData1[ipol].resize(recvDataSize1, 0.0);
          }
        //
        MPI_Gatherv(&(send_data0[0]),
                    send_size0,
                    dftfe::dataTypes::mpi_type_id(send_data0.data()),
                    &(recvdData0[0]),
                    &(recv_size0[0]),
                    &(mpi_offsets0[0]),
                    dftfe::dataTypes::mpi_type_id(recvdData0.data()),
                    proc,
                    mpi_communicator);
        MPI_Gatherv(&(send_data2[0]),
                    send_size0,
                    dftfe::dataTypes::mpi_type_id(send_data2.data()),
                    &(recvdData2[0]),
                    &(recv_size0[0]),
                    &(mpi_offsets0[0]),
                    dftfe::dataTypes::mpi_type_id(recvdData2.data()),
                    proc,
                    mpi_communicator);
        MPI_Gatherv(&(send_data3[0]),
                    send_size0,
                    dftfe::dataTypes::mpi_type_id(send_data3.data()),
                    &(recvdData3[0]),
                    &(recv_size0[0]),
                    &(mpi_offsets0[0]),
                    dftfe::dataTypes::mpi_type_id(recvdData3.data()),
                    proc,
                    mpi_communicator);
        for (dftfe::uInt ipol = 0; ipol < 3; ++ipol)
          {
            send_data = send_data1[ipol];
            MPI_Gatherv(&(send_data[0]),
                        send_size1,
                        MPI_DOUBLE,
                        &(recvdData[0]),
                        &(recv_size1[0]),
                        &(mpi_offsets1[0]),
                        MPI_DOUBLE,
                        proc,
                        mpi_communicator);
            if (proc == this_mpi_process)
              recvdData1[ipol] = recvdData;
          }
        send_data0.clear();
        send_data.clear();
        recvdData.clear();
        send_data1[0].clear();
        send_data1[1].clear();
        send_data1[2].clear();
        send_data2.clear();
        send_data3.clear();
      }
    //
    MPI_Barrier(mpi_communicator);
    //================================================================================================================================================
    //			     Prepare the receiving vectors on which computed density is to
    // be received in symmetrizeRho.cc 		We do this here instead of doing in
    // symmetrizeRho.cc, because symmetrizeRho.cc is to be called during each
    // SCF iteration 							So this better be a one time cost
    //================================================================================================================================================
    cell      = (dftPtr->dofHandlerEigen).begin_active();
    totPoints = 0;
    recv_size.resize(dftPtr->n_mpi_processes, 0);
    //
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
              {
                rhoRecvd[iSymm][globalCellId_parallel[cell->id()]] =
                  std::vector<std::vector<double>>(dftPtr->n_mpi_processes);
                for (dftfe::uInt proc = 0; proc < dftPtr->n_mpi_processes;
                     ++proc)
                  {
                    recv_size[proc] =
                      recv_size[proc] +
                      (1 + dftPtr->getParametersObject().spinPolarized) *
                        send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                     [proc][1];
                    rhoRecvd[iSymm][globalCellId_parallel[cell->id()]][proc]
                      .resize(
                        (1 + dftPtr->getParametersObject().spinPolarized) *
                        send_buf_size[iSymm][globalCellId_parallel[cell->id()]]
                                     [proc][1]);
                  }
              }
          }
      }
    //
    for (dftfe::Int i = 0; i < dftPtr->n_mpi_processes; i++)
      {
        recv_size1[i] =
          (1 + dftPtr->getParametersObject().spinPolarized) * recv_size1[i];
        mpi_offsets1[i] =
          (1 + dftPtr->getParametersObject().spinPolarized) * mpi_offsets1[i];
      }
    //
    if (isGradDensityDataRequired)
      {
        cell = (dftPtr->dofHandlerEigen).begin_active();
        for (dftfe::Int i = 0; i < dftPtr->n_mpi_processes; i++)
          {
            recvGrad_size1[i]   = 3 * recv_size1[i];
            mpiGrad_offsets1[i] = 3 * mpi_offsets1[i];
          }
        //
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
                  {
                    gradRhoRecvd[iSymm][globalCellId_parallel[cell->id()]] =
                      std::vector<std::vector<double>>(dftPtr->n_mpi_processes);
                    for (dftfe::uInt proc = 0; proc < dftPtr->n_mpi_processes;
                         ++proc)
                      gradRhoRecvd
                        [iSymm][globalCellId_parallel[cell->id()]][proc]
                          .resize(
                            (1 + dftPtr->getParametersObject().spinPolarized) *
                            3 *
                            send_buf_size[iSymm]
                                         [globalCellId_parallel[cell->id()]]
                                         [proc][1]);
                  }
              }
          }
      }
    //
  }
  //============================================================================================================================================
  //============================================================================================================================================
  //			           Just a quick snippet to compute cross product of two
  // vectors
  //============================================================================================================================================
  //============================================================================================================================================
  std::vector<double>
  cross_product(const std::vector<double> &a, const std::vector<double> &b)
  {
    std::vector<double> crossProduct(a.size(), 0.0);
    crossProduct[0] = a[1] * b[2] - a[2] * b[1];
    crossProduct[1] = a[2] * b[0] - a[0] * b[2];
    crossProduct[2] = a[0] * b[1] - a[1] * b[0];
    return crossProduct;
  }
  //================================================================================================================================================
  //================================================================================================================================================
  //			           Just a quick snippet to go back and forth between crystal
  // and cartesian coordinates. 			              flag==1 takes crystal to
  // cartesian and flag==-1 does the other way around.
  //================================================================================================================================================
  //================================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  dealii::Point<3>
  symmetryClass<memorySpace>::crys2cart(dealii::Point<3> p, dftfe::Int flag)
  {
    dealii::Point<3> ptemp;
    if (flag == 1)
      {
        ptemp[0] = p[0] * (dftPtr->d_domainBoundingVectors)[0][0] +
                   p[1] * (dftPtr->d_domainBoundingVectors)[1][0] +
                   p[2] * (dftPtr->d_domainBoundingVectors)[2][0];
        ptemp[1] = p[0] * (dftPtr->d_domainBoundingVectors)[0][1] +
                   p[1] * (dftPtr->d_domainBoundingVectors)[1][1] +
                   p[2] * (dftPtr->d_domainBoundingVectors)[2][1];
        ptemp[2] = p[0] * (dftPtr->d_domainBoundingVectors)[0][2] +
                   p[1] * (dftPtr->d_domainBoundingVectors)[1][2] +
                   p[2] * (dftPtr->d_domainBoundingVectors)[2][2];
      }
    if (flag == -1)
      {
        //----------------------------------- Here we compute the reciprocal of
        // the domain bounding vectors needed to transfer
        //--------------------------
        //------------------------------------ points from cartesian to crystal
        // coordinates. They are same as the reciprocal
        //----------------------------
        //------------------------------------ vectors for fully periodic case,
        // but different for mix boundaries
        //----------------------------------------
        std::vector<std::vector<double>> reciprocalLatticeVectors(
          3, std::vector<double>(3, 0.0));
        dftfe::uInt         periodicitySum = 0;
        std::vector<double> cross(3, 0.0);
        double              scalarConst;
        for (dftfe::uInt i = 0; i < 2; ++i)
          {
            cross =
              cross_product((dftPtr->d_domainBoundingVectors)[i + 1],
                            (dftPtr->d_domainBoundingVectors)[3 - (2 * i + 1)]);
            scalarConst = (dftPtr->d_domainBoundingVectors)[i][0] * cross[0] +
                          (dftPtr->d_domainBoundingVectors)[i][1] * cross[1] +
                          (dftPtr->d_domainBoundingVectors)[i][2] * cross[2];
            for (dftfe::uInt d = 0; d < 3; ++d)
              reciprocalLatticeVectors[i][d] = (1.0 / scalarConst) * cross[d];
          }
        //
        cross       = cross_product((dftPtr->d_domainBoundingVectors)[0],
                              (dftPtr->d_domainBoundingVectors)[1]);
        scalarConst = (dftPtr->d_domainBoundingVectors)[2][0] * cross[0] +
                      (dftPtr->d_domainBoundingVectors)[2][1] * cross[1] +
                      (dftPtr->d_domainBoundingVectors)[2][2] * cross[2];
        for (dftfe::uInt d = 0; d < 3; ++d)
          reciprocalLatticeVectors[2][d] = (1.0 / scalarConst) * cross[d];
        //
        ptemp[0] = p[0] * reciprocalLatticeVectors[0][0] +
                   p[1] * reciprocalLatticeVectors[0][1] +
                   p[2] * reciprocalLatticeVectors[0][2];
        ptemp[1] = p[0] * reciprocalLatticeVectors[1][0] +
                   p[1] * reciprocalLatticeVectors[1][1] +
                   p[2] * reciprocalLatticeVectors[1][2];
        ptemp[2] = p[0] * reciprocalLatticeVectors[2][0] +
                   p[1] * reciprocalLatticeVectors[2][1] +
                   p[2] * reciprocalLatticeVectors[2][2];
        ptemp = ptemp;
      }

    return ptemp;
  }
  //================================================================================================================================================
#include "symmetrize.inst.cc"
  //=================================================================================================================================================
} // namespace dftfe
