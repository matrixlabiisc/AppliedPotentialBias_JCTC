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
// @author Vishal Subramanian
//

#include "excDensityGGAClass.h"
#include "excDensityLDAClass.h"
#include "excDensityLLMGGAClass.h"
#include "ExcDFTPlusU.h"
#include "Exceptions.h"
#include "AuxDensityMatrixFE.h"
#include <dftfeDataTypes.h>

namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  ExcDFTPlusU<ValueType, memorySpace>::ExcDFTPlusU(
    std::shared_ptr<ExcSSDFunctionalBaseClass<memorySpace>> excSSDObjPtr,
    dftfe::uInt                                             numSpins)
    : ExcSSDFunctionalBaseClass<memorySpace>(*(excSSDObjPtr.get()))
  {
    this->d_ExcFamilyType = ExcFamilyType::DFTPlusU;
    d_excSSDObjPtr        = excSSDObjPtr;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  ExcDFTPlusU<ValueType, memorySpace>::~ExcDFTPlusU()
  {}



  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::computeRhoTauDependentXCData(
    AuxDensityMatrix<memorySpace>             &auxDensityMatrix,
    const std::pair<dftfe::uInt, dftfe::uInt> &quadIndexRange,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &xDataOut,
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      &cDataOut) const
  {
    d_excSSDObjPtr->computeRhoTauDependentXCData(auxDensityMatrix,
                                                 quadIndexRange,
                                                 xDataOut,
                                                 cDataOut);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const
  {
    d_excSSDObjPtr->checkInputOutputDataAttributesConsistency(
      outputDataAttributes);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
                                                                      &src,
    dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {
    d_hubbardClassPtr->applyPotentialDueToHubbardCorrection(
      src, dst, inputVecSize, kPointIndex, spinIndex);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::applyWaveFunctionDependentFuncDerWrtPsi(
    const dftfe::linearAlgebra::MultiVector<
      typename dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &src,
    dftfe::linearAlgebra::MultiVector<
      typename dataTypes::singlePrecType<ValueType>::type,
      memorySpace>   &dst,
    const dftfe::uInt inputVecSize,
    const dftfe::uInt kPointIndex,
    const dftfe::uInt spinIndex)
  {
    d_hubbardClassPtr->applyPotentialDueToHubbardCorrection(
      src, dst, inputVecSize, kPointIndex, spinIndex);
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::updateWaveFunctionDependentFuncDerWrtPsi(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrixPtr,
    const std::vector<double>                            &kPointWeights)
  {
    std::shared_ptr<AuxDensityMatrixFE<memorySpace>> auxDensityMatrixFEPtr =
      std::dynamic_pointer_cast<AuxDensityMatrixFE<memorySpace>>(
        auxDensityMatrixPtr);

    d_hubbardClassPtr->computeOccupationMatrix(
      auxDensityMatrixFEPtr->getDensityMatrixComponents_wavefunctions(),
      *(auxDensityMatrixFEPtr->getDensityMatrixComponents_occupancies()));
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::computeWaveFunctionDependentExcEnergy(
    const std::shared_ptr<AuxDensityMatrix<memorySpace>> &auxDensityMatrix,
    const std::vector<double>                            &kPointWeights)
  {
    d_hubbardClassPtr->computeEnergyFromOccupationMatrix();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  ExcDFTPlusU<ValueType, memorySpace>::getWaveFunctionDependentExcEnergy()
  {
    return d_hubbardClassPtr->getHubbardEnergy();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  double
  ExcDFTPlusU<ValueType, memorySpace>::
    getExpectationOfWaveFunctionDependentExcFuncDerWrtPsi()
  {
    return d_hubbardClassPtr->getExpectationOfHubbardPotential();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::initialiseHubbardClass(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain,
    const MPI_Comm &mpi_comm_interPool,
    const MPI_Comm &mpi_comm_interBandGroup,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
      basisOperationsMemPtr,
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      basisOperationsHostPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      BLASWrapperMemPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                            BLASWrapperHostPtr,
    const dftfe::uInt                       matrixFreeVectorComponent,
    const dftfe::uInt                       densityQuadratureId,
    const dftfe::uInt                       sparsityPatternQuadratureId,
    const dftfe::uInt                       numberWaveFunctions,
    const dftfe::uInt                       numSpins,
    const dftParameters                    &dftParam,
    const std::string                      &scratchFolderName,
    const bool                              singlePrecNonLocalOperator,
    const bool                              updateNonlocalSparsity,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &atomLocationsFrac,
    const std::vector<dftfe::Int>          &imageIds,
    const std::vector<std::vector<double>> &imagePositions,
    std::vector<double>                    &kPointCoordinates,
    const std::vector<double>              &kPointWeights,
    const std::vector<std::vector<double>> &domainBoundaries)
  {
    d_hubbardClassPtr = std::make_shared<hubbard<ValueType, memorySpace>>(
      mpi_comm_parent,
      mpi_comm_domain,
      mpi_comm_interPool,
      mpi_comm_interBandGroup);

    d_hubbardClassPtr->init(basisOperationsMemPtr,
                            basisOperationsHostPtr,
                            BLASWrapperMemPtr,
                            BLASWrapperHostPtr,
                            matrixFreeVectorComponent,
                            densityQuadratureId,
                            sparsityPatternQuadratureId,
                            numberWaveFunctions,
                            numSpins,
                            dftParam,
                            scratchFolderName,
                            singlePrecNonLocalOperator,
                            updateNonlocalSparsity,
                            atomLocations,
                            atomLocationsFrac,
                            imageIds,
                            imagePositions,
                            kPointCoordinates,
                            kPointWeights,
                            domainBoundaries);
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  std::shared_ptr<hubbard<ValueType, memorySpace>> &
  ExcDFTPlusU<ValueType, memorySpace>::getHubbardClass()
  {
    return d_hubbardClassPtr;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<ValueType, memorySpace>::reinitKPointDependentVariables(
    dftfe::uInt kPointIndex)
  {
    d_hubbardClassPtr->initialiseOperatorActionOnX(kPointIndex);
  }


  template class ExcDFTPlusU<dataTypes::number,
                             dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class ExcDFTPlusU<dataTypes::number,
                             dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
