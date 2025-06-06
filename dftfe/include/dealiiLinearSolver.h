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


#include <linearSolver.h>

#ifndef dealiiLinearSolver_H_
#  define dealiiLinearSolver_H_

namespace dftfe
{
  /**
   * @brief dealii linear solver class wrapper
   *
   * @author Sambit Das
   */
  class dealiiLinearSolver : public linearSolver
  {
  public:
    enum solverType
    {
      CG = 0,
      GMRES
    };

    /**
     * @brief Constructor
     *
     * @param mpi_comm_parent parent mpi communicato
     * @param mpi_comm_domain domain mpi communicator
     * @param type enum specifying the choice of the dealii linear solver
     */
    dealiiLinearSolver(const MPI_Comm  &mpi_comm_parent,
                       const MPI_Comm  &mpi_comm_domain,
                       const solverType type);

    /**
     * @brief Solve linear system, A*x=Rhs
     *
     * @param problem linearSolverProblem object (functor) to compute Rhs and A*x, and preconditioning
     * @param relTolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     */
    void
    solve(dealiiLinearSolverProblem &problem,
          const double               absTolerance,
          const dftfe::uInt          maxNumberIterations,
          const dftfe::Int           debugLevel     = 0,
          bool                       distributeFlag = true);

  private:
    /// enum denoting the choice of the dealii solver
    const solverType d_type;

    /// define some temporary vectors
    distributedCPUVec<double> gvec, dvec, hvec;

    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const dftfe::uInt          n_mpi_processes;
    const dftfe::uInt          this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe

#endif
