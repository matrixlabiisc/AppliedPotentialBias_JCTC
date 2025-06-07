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
#include <eshelbyTensor.h>
#include <eshelbyTensorSpinPolarized.h>

namespace dftfe
{
  // compute nonlinear core correction contribution to stress
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::addENonlinearCoreCorrectionStressContribution(
    FEEvaluationWrapperClass<3>         &forceEval,
    const dealii::MatrixFree<3, double> &matrixFreeData,
    const dftfe::uInt                    cell,
    const dealii::AlignedVector<dealii::VectorizedArray<double>> &vxcQuads,
    const dealii::AlignedVector<
      dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &derExcGradRho,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
      &hessianRhoCoreAtoms)
  {
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      zeroTensor1[idim] = dealii::make_vectorized_array(0.0);

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
        zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);

    const dftfe::uInt numberGlobalAtoms  = dftPtr->atomLocations.size();
    const dftfe::uInt numberImageCharges = dftPtr->d_imageIdsTrunc.size();
    const dftfe::uInt totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
    const dftfe::uInt numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const dftfe::uInt numQuadPoints = forceEval.n_q_points;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      xMinusAtomLoc(numQuadPoints, zeroTensor1);


    bool isGradDensityDataDependent =
      (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
         ->getDensityBasedFamilyType() == densityFamilyType::GGA);


    for (dftfe::uInt iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor2);

        double           atomCharge;
        dftfe::uInt      atomId = iAtom;
        dealii::Point<3> atomLocation;
        if (iAtom < numberGlobalAtoms)
          {
            atomLocation[0] = dftPtr->atomLocations[iAtom][2];
            atomLocation[1] = dftPtr->atomLocations[iAtom][3];
            atomLocation[2] = dftPtr->atomLocations[iAtom][4];
            if (d_dftParams.isPseudopotential)
              atomCharge = dftPtr->atomLocations[iAtom][1];
            else
              atomCharge = dftPtr->atomLocations[iAtom][0];
          }
        else
          {
            const dftfe::Int imageId = iAtom - numberGlobalAtoms;
            atomId                   = dftPtr->d_imageIdsTrunc[imageId];
            atomCharge               = dftPtr->d_imageChargesTrunc[imageId];
            atomLocation[0] = dftPtr->d_imagePositionsTrunc[imageId][0];
            atomLocation[1] = dftPtr->d_imagePositionsTrunc[imageId][1];
            atomLocation[2] = dftPtr->d_imagePositionsTrunc[imageId][2];
          }

        bool isLocalDomainOutsideCoreRhoTail = false;
        if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
          isLocalDomainOutsideCoreRhoTail = true;

        if (isLocalDomainOutsideCoreRhoTail)
          continue;

        bool isCellOutsideCoreRhoTail = true;

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            std::map<dealii::CellId, std::vector<double>>::const_iterator it =
              gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
            if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
              {
                isCellOutsideCoreRhoTail        = false;
                const std::vector<double> &temp = it->second;
                for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                  {
                    gradRhoCoreAtomsQuads[q][0][iSubCell] = temp[q * 3];
                    gradRhoCoreAtomsQuads[q][1][iSubCell] = temp[q * 3 + 1];
                    gradRhoCoreAtomsQuads[q][2][iSubCell] = temp[q * 3 + 2];
                  }
              }

            if (isGradDensityDataDependent && !isCellOutsideCoreRhoTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it2 = hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

                if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                  {
                    const std::vector<double> &temp = it2->second;
                    for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                      {
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                            hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] =
                              temp[q * 3 * 3 + 3 * iDim + jDim];
                      }
                  }
              }

            if (!isCellOutsideCoreRhoTail)
              for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                {
                  const dealii::Point<3, dealii::VectorizedArray<double>>
                    &quadPointVectorized = forceEval.quadrature_point(q);
                  dealii::Point<3> quadPoint;
                  quadPoint[0] = quadPointVectorized[0][iSubCell];
                  quadPoint[1] = quadPointVectorized[1][iSubCell];
                  quadPoint[2] = quadPointVectorized[2][iSubCell];
                  const dealii::Tensor<1, 3, double> dispAtom =
                    quadPoint - atomLocation;
                  for (dftfe::uInt idim = 0; idim < 3; idim++)
                    {
                      xMinusAtomLoc[q][idim][iSubCell] = dispAtom[idim];
                    }
                }
          } // subCell loop

        if (isCellOutsideCoreRhoTail)
          continue;


        dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
          stressContribution = zeroTensor2;

        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
          {
            stressContribution +=
              outer_product(eshelbyTensor::getFNonlinearCoreCorrection(
                              vxcQuads[q], gradRhoCoreAtomsQuads[q]) +
                              eshelbyTensor::getFNonlinearCoreCorrection(
                                derExcGradRho[q], hessianRhoCoreAtomsQuads[q]),
                            xMinusAtomLoc[q]) *
              forceEval.JxW(q);
          }

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              d_stress[idim][jdim] += stressContribution[idim][jdim][iSubCell];
      } // iAtom loop
  }

  // compute nonlinear core correction contribution to stress
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::
    addENonlinearCoreCorrectionStressContributionSpinPolarized(
      FEEvaluationWrapperClass<3>         &forceEval,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const dftfe::uInt                    cell,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin0,
      const dealii::AlignedVector<dealii::VectorizedArray<double>>
        &vxcQuadsSpin1,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin0,
      const dealii::AlignedVector<
        dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
        &derExcGradRhoSpin1,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
        &gradRhoCoreAtoms,
      const std::map<dftfe::uInt, std::map<dealii::CellId, std::vector<double>>>
                &hessianRhoCoreAtoms,
      const bool isXCGGA)
  {
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      zeroTensor1[idim] = dealii::make_vectorized_array(0.0);

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
        zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);

    const dftfe::uInt numberGlobalAtoms  = dftPtr->atomLocations.size();
    const dftfe::uInt numberImageCharges = dftPtr->d_imageIdsTrunc.size();
    const dftfe::uInt totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
    const dftfe::uInt numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const dftfe::uInt numQuadPoints = forceEval.n_q_points;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      xMinusAtomLoc(numQuadPoints, zeroTensor1);

    bool isGradDensityDataDependent =
      (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
         ->getDensityBasedFamilyType() == densityFamilyType::GGA);

    for (dftfe::uInt iAtom = 0; iAtom < totalNumberAtoms; iAtom++)
      {
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          gradRhoCoreAtomsQuads(numQuadPoints, zeroTensor1);
        dealii::AlignedVector<
          dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>
          hessianRhoCoreAtomsQuads(numQuadPoints, zeroTensor2);

        double           atomCharge;
        dftfe::uInt      atomId = iAtom;
        dealii::Point<3> atomLocation;
        if (iAtom < numberGlobalAtoms)
          {
            atomLocation[0] = dftPtr->atomLocations[iAtom][2];
            atomLocation[1] = dftPtr->atomLocations[iAtom][3];
            atomLocation[2] = dftPtr->atomLocations[iAtom][4];
            if (d_dftParams.isPseudopotential)
              atomCharge = dftPtr->atomLocations[iAtom][1];
            else
              atomCharge = dftPtr->atomLocations[iAtom][0];
          }
        else
          {
            const dftfe::Int imageId = iAtom - numberGlobalAtoms;
            atomId                   = dftPtr->d_imageIdsTrunc[imageId];
            atomCharge               = dftPtr->d_imageChargesTrunc[imageId];
            atomLocation[0] = dftPtr->d_imagePositionsTrunc[imageId][0];
            atomLocation[1] = dftPtr->d_imagePositionsTrunc[imageId][1];
            atomLocation[2] = dftPtr->d_imagePositionsTrunc[imageId][2];
          }

        bool isLocalDomainOutsideCoreRhoTail = false;
        if (gradRhoCoreAtoms.find(iAtom) == gradRhoCoreAtoms.end())
          isLocalDomainOutsideCoreRhoTail = true;

        if (isLocalDomainOutsideCoreRhoTail)
          continue;

        bool isCellOutsideCoreRhoTail = true;

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId subCellId = subCellPtr->id();

            std::map<dealii::CellId, std::vector<double>>::const_iterator it =
              gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
            if (it != gradRhoCoreAtoms.find(iAtom)->second.end())
              {
                isCellOutsideCoreRhoTail        = false;
                const std::vector<double> &temp = it->second;
                for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                  {
                    gradRhoCoreAtomsQuads[q][0][iSubCell] = temp[q * 3] / 2.0;
                    gradRhoCoreAtomsQuads[q][1][iSubCell] =
                      temp[q * 3 + 1] / 2.0;
                    gradRhoCoreAtomsQuads[q][2][iSubCell] =
                      temp[q * 3 + 2] / 2.0;
                  }
              }

            if (isGradDensityDataDependent && !isCellOutsideCoreRhoTail)
              {
                std::map<dealii::CellId, std::vector<double>>::const_iterator
                  it2 = hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);

                if (it2 != hessianRhoCoreAtoms.find(iAtom)->second.end())
                  {
                    const std::vector<double> &temp2 = it2->second;
                    for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                      {
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                            hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] =
                              temp2[q * 3 * 3 + 3 * iDim + jDim] / 2.0;
                      }
                  }
              }

            if (!isCellOutsideCoreRhoTail)
              for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
                {
                  const dealii::Point<3, dealii::VectorizedArray<double>>
                    &quadPointVectorized = forceEval.quadrature_point(q);
                  dealii::Point<3> quadPoint;
                  quadPoint[0] = quadPointVectorized[0][iSubCell];
                  quadPoint[1] = quadPointVectorized[1][iSubCell];
                  quadPoint[2] = quadPointVectorized[2][iSubCell];
                  const dealii::Tensor<1, 3, double> dispAtom =
                    quadPoint - atomLocation;
                  for (dftfe::uInt idim = 0; idim < 3; idim++)
                    {
                      xMinusAtomLoc[q][idim][iSubCell] = dispAtom[idim];
                    }
                }
          } // subCell loop

        if (isCellOutsideCoreRhoTail)
          continue;


        dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
          stressContribution = zeroTensor2;

        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
          stressContribution +=
            outer_product(eshelbyTensorSP::getFNonlinearCoreCorrection(
                            vxcQuadsSpin0[q],
                            vxcQuadsSpin1[q],
                            derExcGradRhoSpin0[q],
                            derExcGradRhoSpin1[q],
                            gradRhoCoreAtomsQuads[q],
                            hessianRhoCoreAtomsQuads[q],
                            isXCGGA),
                          xMinusAtomLoc[q]) *
            forceEval.JxW(q);

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              d_stress[idim][jdim] += stressContribution[idim][jdim][iSubCell];
      } // iAtom loop
  }
#include "../force.inst.cc"
} // namespace dftfe
