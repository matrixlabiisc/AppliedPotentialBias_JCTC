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

#ifndef symmetry_H_
#define symmetry_H_
#include <complex>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include "constants.h"
#include "dft.h"
#include "headers.h"

namespace dftfe
{
  /**
   * @brief density symmetrization based on irreducible Brillouin zone calculation, only relevant for calculations using point group symmetries
   *
   * @author Krishnendu Ghosh, krisg@umich.edu
   */

  template <dftfe::utils::MemorySpace memorySpace>
  class symmetryClass
  {
    // template <dftfe::uInt T1, dftfe::uInt T2>
    friend class dftClass<memorySpace>;

  public:
    /**
     * symmetryClass constructor
     */
    symmetryClass(dftClass<memorySpace> *_dftPtr,
                  const MPI_Comm        &mpi_comm_parent,
                  const MPI_Comm        &mpi_comm_domain,
                  const MPI_Comm        &_interpoolcomm);
    /**
     * Main driver routine to generate and communicate mapping tables
     */
    void
    initSymmetry();
    /**
     * computes total density by summing over all the symmetry transformed
     * points
     */
    void
    computeAndSymmetrize_rhoOut();
    /**
     * computes density at all the transformed points received from other
     * processors and scatters the density back to the corresponding processors
     */
    void
    computeLocalrhoOut();
    /**
     * Wipes out mapping tables between relaxation steps
     */
    void
    clearMaps();
    /**
     * quick snippet to go back and forth between crystal and cartesian
     * coordinates
     * @param [in] p  point that is to be transformed
     * @param [in] flag type of coordinate transformation, 1 takes crys. to cart. -1 takes cart. to crys.
     */
    dealii::Point<3>
    crys2cart(dealii::Point<3> p, dftfe::Int flag);


  private:
    dftClass<memorySpace> *dftPtr;
    /**
     * dealii based FE data structres
     */
    dealii::FE_Q<3> FE;
    /**
     * compute-time logger
     */
    dealii::TimerOutput computing_timer;
    /**
     * parallel objects
     */
    const MPI_Comm             d_mpiCommParent, mpi_communicator, interpoolcomm;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
    /**
     * Space group symmetry related data
     */
    std::vector<std::vector<std::vector<double>>> symmMat;
    dftfe::uInt                                   numSymm;
    double                                        translation[500][3];
    std::vector<std::vector<dftfe::Int>>          symmUnderGroup;
    std::vector<dftfe::Int>                       numSymmUnderGroup;
    /**
     * Data members required for storing mapping tables locally
     */
    std::map<
      dealii::CellId,
      std::vector<std::tuple<dftfe::Int, std::vector<double>, dftfe::Int>>>
      cellMapTable;
    std::vector<
      std::vector<std::vector<std::tuple<dftfe::Int, dftfe::Int, dftfe::Int>>>>
      mappedGroup;
    std::map<dftfe::Int, typename dealii::DoFHandler<3>::active_cell_iterator>
                                         dealIICellId;
    std::map<dealii::CellId, dftfe::Int> globalCellId;
    std::vector<dftfe::Int>              ownerProcGlobal;
    /**
     * Data members required for communicating mapping tables
     */
    std::vector<std::vector<std::vector<std::vector<dftfe::Int>>>>
      mappedGroupSend0;
    std::vector<std::vector<std::vector<std::vector<dftfe::Int>>>>
      mappedGroupSend2;
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>
                                                      mappedGroupSend1;
    std::vector<std::vector<std::vector<dftfe::Int>>> mappedGroupRecvd0;
    std::vector<std::vector<std::vector<dftfe::Int>>> mappedGroupRecvd2;
    std::vector<std::vector<std::vector<std::vector<double>>>>
      mappedGroupRecvd1;
    std::vector<std::vector<std::vector<std::vector<dftfe::Int>>>>
      send_buf_size;
    std::vector<std::vector<std::vector<std::vector<dftfe::Int>>>>
                                                               recv_buf_size;
    std::vector<std::vector<std::vector<std::vector<double>>>> rhoRecvd,
      gradRhoRecvd;
    std::vector<std::vector<std::vector<std::vector<dftfe::Int>>>> groupOffsets;
    /**
     * Data sizes and offsets required for MPI scattering and gathering of
     * mapping tables and symmetrized density They have to be data members since
     * the same sizes and offsets are used in both communication mapping tables
     * and symmetrized density
     */
    dftfe::uInt      totPoints;
    std::vector<int> mpi_scatter_offset, send_scatter_size, recv_size,
      mpi_scatterGrad_offset, send_scatterGrad_size;
    std::vector<int>        mpi_offsets0, mpi_offsets1, mpiGrad_offsets1;
    std::vector<dftfe::Int> recvdData0, recvdData2, recvdData3;
    std::vector<std::vector<double>> recvdData1;
    std::vector<int>                 recv_size0, recv_size1, recvGrad_size1;
    //
  };
} // namespace dftfe
#endif
