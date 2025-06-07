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


namespace dftfe
{
  // compute ESmeared contribution stress
  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::addEPhiTotSmearedStressContribution(
    FEEvaluationWrapperClass<3>         &forceEval,
    const dealii::MatrixFree<3, double> &matrixFreeData,
    const dftfe::uInt                    cell,
    const dealii::AlignedVector<
      dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradPhiTotQuads,
    const std::vector<dftfe::uInt> &nonTrivialAtomImageIdsMacroCell,
    const std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtomsImages,
    const dealii::AlignedVector<dealii::VectorizedArray<double>> &smearedbQuads)
  {
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      zeroTensor1[idim] = dealii::make_vectorized_array(0.0);

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
        {
          zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
        }

    const dftfe::uInt numberGlobalAtoms  = dftPtr->atomLocations.size();
    const dftfe::uInt numberImageCharges = dftPtr->d_imageIdsTrunc.size();
    const dftfe::uInt numberTotalAtoms = numberGlobalAtoms + numberImageCharges;
    const dftfe::uInt numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const dftfe::uInt numQuadPoints = forceEval.n_q_points;

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuadsiAtom(
      numQuadPoints, dealii::make_vectorized_array(0.0));

    for (dftfe::Int iAtomNonTrivial = 0;
         iAtomNonTrivial < nonTrivialAtomImageIdsMacroCell.size();
         iAtomNonTrivial++)
      {
        const dftfe::Int iAtom =
          nonTrivialAtomImageIdsMacroCell[iAtomNonTrivial];
        dealii::Point<3, double> atomLocation;
        if (iAtom < numberGlobalAtoms)
          {
            atomLocation[0] = dftPtr->atomLocations[iAtom][2];
            atomLocation[1] = dftPtr->atomLocations[iAtom][3];
            atomLocation[2] = dftPtr->atomLocations[iAtom][4];
          }
        else
          {
            const dftfe::Int imageId = iAtom - numberGlobalAtoms;
            atomLocation[0] = dftPtr->d_imagePositionsTrunc[imageId][0];
            atomLocation[1] = dftPtr->d_imagePositionsTrunc[imageId][1];
            atomLocation[2] = dftPtr->d_imagePositionsTrunc[imageId][2];
          }

        dealii::Point<3, dealii::VectorizedArray<double>> atomLocationVect;
        atomLocationVect[0] = dealii::make_vectorized_array(atomLocation[0]);
        atomLocationVect[1] = dealii::make_vectorized_array(atomLocation[1]);
        atomLocationVect[2] = dealii::make_vectorized_array(atomLocation[2]);

        std::fill(smearedbQuadsiAtom.begin(),
                  smearedbQuadsiAtom.end(),
                  dealii::make_vectorized_array(0.0));

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId                 subCellId = subCellPtr->id();
            const std::vector<dftfe::Int> &bQuadAtomIdsCell =
              bQuadAtomIdsAllAtomsImages.find(subCellId)->second;
            for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
              if (bQuadAtomIdsCell[q] == iAtom)
                smearedbQuadsiAtom[q][iSubCell] = smearedbQuads[q][iSubCell];
          }


        dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
          EPSPStressContribution = zeroTensor2;
        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
          EPSPStressContribution -=
            outer_product(smearedbQuadsiAtom[q] * gradPhiTotQuads[q],
                          forceEval.quadrature_point(q) - atomLocationVect) *
            forceEval.JxW(q);

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              d_stress[idim][jdim] +=
                EPSPStressContribution[idim][jdim][iSubCell];
      } // iAtom loop
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  forceClass<memorySpace>::addEVselfSmearedStressContribution(
    FEEvaluationWrapperClass<3>         &forceEval,
    const dealii::MatrixFree<3, double> &matrixFreeData,
    const dftfe::uInt                    cell,
    const dealii::AlignedVector<
      dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &gradVselfQuads,
    const std::vector<dftfe::uInt> &nonTrivialAtomImageIdsMacroCell,
    const std::map<dealii::CellId, std::vector<dftfe::Int>>
      &bQuadAtomIdsAllAtomsImages,
    const dealii::AlignedVector<dealii::VectorizedArray<double>> &smearedbQuads)
  {
    dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor1;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      zeroTensor1[idim] = dealii::make_vectorized_array(0.0);

    dealii::Tensor<2, 3, dealii::VectorizedArray<double>> zeroTensor2;
    for (dftfe::uInt idim = 0; idim < 3; idim++)
      for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
        {
          zeroTensor2[idim][jdim] = dealii::make_vectorized_array(0.0);
        }

    const dftfe::uInt numSubCells =
      matrixFreeData.n_active_entries_per_cell_batch(cell);
    const dftfe::uInt numQuadPoints = forceEval.n_q_points;

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    const dftfe::uInt numberGlobalAtoms = dftPtr->atomLocations.size();

    dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuadsiAtom(
      numQuadPoints, dealii::make_vectorized_array(0.0));

    for (dftfe::Int iAtomNonTrivial = 0;
         iAtomNonTrivial < nonTrivialAtomImageIdsMacroCell.size();
         iAtomNonTrivial++)
      {
        const dftfe::Int atomId =
          nonTrivialAtomImageIdsMacroCell[iAtomNonTrivial];

        dealii::Point<3, double> atomLocation;
        if (atomId < numberGlobalAtoms)
          {
            atomLocation[0] = dftPtr->atomLocations[atomId][2];
            atomLocation[1] = dftPtr->atomLocations[atomId][3];
            atomLocation[2] = dftPtr->atomLocations[atomId][4];
          }
        else
          {
            const dftfe::Int imageId = atomId - numberGlobalAtoms;
            atomLocation[0] = dftPtr->d_imagePositionsTrunc[imageId][0];
            atomLocation[1] = dftPtr->d_imagePositionsTrunc[imageId][1];
            atomLocation[2] = dftPtr->d_imagePositionsTrunc[imageId][2];
          }

        dealii::Point<3, dealii::VectorizedArray<double>> atomLocationVect;
        atomLocationVect[0] = dealii::make_vectorized_array(atomLocation[0]);
        atomLocationVect[1] = dealii::make_vectorized_array(atomLocation[1]);
        atomLocationVect[2] = dealii::make_vectorized_array(atomLocation[2]);

        std::fill(smearedbQuadsiAtom.begin(),
                  smearedbQuadsiAtom.end(),
                  dealii::make_vectorized_array(0.0));

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
            dealii::CellId                 subCellId = subCellPtr->id();
            const std::vector<dftfe::Int> &bQuadAtomIdsCell =
              bQuadAtomIdsAllAtomsImages.find(subCellId)->second;
            for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
              if (bQuadAtomIdsCell[q] == atomId)
                smearedbQuadsiAtom[q][iSubCell] = smearedbQuads[q][iSubCell];
          }

        dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
          EPSPStressContribution = zeroTensor2;
        for (dftfe::uInt q = 0; q < numQuadPoints; ++q)
          EPSPStressContribution +=
            outer_product(smearedbQuadsiAtom[q] * gradVselfQuads[q],
                          forceEval.quadrature_point(q) - atomLocationVect) *
            forceEval.JxW(q);

        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          for (dftfe::uInt idim = 0; idim < 3; idim++)
            for (dftfe::uInt jdim = 0; jdim < 3; jdim++)
              d_stress[idim][jdim] +=
                EPSPStressContribution[idim][jdim][iSubCell];
      } // iAtom loop
  }
#include "../force.inst.cc"
} // namespace dftfe
