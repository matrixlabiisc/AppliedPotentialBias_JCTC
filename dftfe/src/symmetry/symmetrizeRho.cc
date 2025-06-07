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
//============================================================================================================================================
//============================================================================================================================================
//                        This is the source file for computing density at
//                        symmetry transformed points and
//	             communicating the density back to the processors from which the
// transformed points came from. 	            Only relevant for calculations
// using multiple k-points and when USE GROUP SYMMETRY = true
//
//                                       Author : Krishnendu Ghosh,
//                                       krisg@umich.edu
//
//============================================================================================================================================
//============================================================================================================================================
//
#include <vectorUtilities.h>

#include "../../include/dft.h"
#include "../../include/symmetry.h"
//
namespace dftfe
{
  //=============================================================================================================================================
  //						Get partial occupancy based on Fermi-Dirac statistics
  //=============================================================================================================================================
  double
  getOccupancy(const double &factor)
  {
    return (factor >= 0) ? std::exp(-factor) / (1.0 + std::exp(-factor)) :
                           1.0 / (1.0 + std::exp(factor));
  }
  //=============================================================================================================================================
  //				Following routine computes total density by summing over all the
  // symmetry transformed points
  //=============================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  symmetryClass<memorySpace>::computeAndSymmetrize_rhoOut()
  {
    const dealii::Quadrature<3> &quadrature =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const dftfe::uInt num_quad_points = quadrature.size();
    const dftfe::uInt numCells = dftPtr->matrix_free_data.n_physical_cells();
    //
    dftPtr->d_densityOutQuadValues.resize(
      dftPtr->getParametersObject().spinPolarized == 1 ? 2 : 1);

    bool isGradDensityDataRequired =
      (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
         ->getDensityBasedFamilyType() == densityFamilyType::GGA);
    ;

    if (isGradDensityDataRequired)
      {
        dftPtr->d_gradDensityOutQuadValues.resize(
          dftPtr->getParametersObject().spinPolarized == 1 ? 2 : 1);
      }
    for (dftfe::uInt iComp = 0; iComp < dftPtr->d_densityOutQuadValues.size();
         ++iComp)
      dftPtr->d_densityOutQuadValues[iComp].resize(numCells * num_quad_points);
    for (dftfe::uInt iComp = 0;
         iComp < dftPtr->d_gradDensityOutQuadValues.size();
         ++iComp)
      dftPtr->d_gradDensityOutQuadValues[iComp].resize(3 * numCells *
                                                       num_quad_points);

    std::vector<double> rhoOut(num_quad_points),
      gradRhoOut(3 * num_quad_points), rhoOutSpinPolarized(2 * num_quad_points),
      gradRhoOutSpinPolarized(6 * num_quad_points);
    //=============================================================================================================================================
    //				Loop over cell and quad point and compute density by summing over
    // all the used symmetries
    //=============================================================================================================================================
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell            = (dftPtr->dofHandlerEigen).begin_active(),
      endc            = (dftPtr->dofHandlerEigen).end();
    dftfe::uInt iCell = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            std::fill(rhoOut.begin(), rhoOut.end(), 0.0);
            if (dftPtr->getParametersObject().spinPolarized == 1)
              {
                std::fill(rhoOutSpinPolarized.begin(),
                          rhoOutSpinPolarized.end(),
                          0.0);
              }
            //
            if (isGradDensityDataRequired)
              {
                std::fill(gradRhoOut.begin(), gradRhoOut.end(), 0.0);
                if (dftPtr->getParametersObject().spinPolarized == 1)
                  {
                    std::fill(gradRhoOutSpinPolarized.begin(),
                              gradRhoOutSpinPolarized.end(),
                              0.0);
                  }
              }
            //
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              {
                for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
                  {
                    const dftfe::uInt proc = std::get<0>(
                      mappedGroup[iSymm][globalCellId[cell->id()]][q_point]);
                    const dftfe::uInt group = std::get<1>(
                      mappedGroup[iSymm][globalCellId[cell->id()]][q_point]);
                    const dftfe::uInt point = std::get<2>(
                      mappedGroup[iSymm][globalCellId[cell->id()]][q_point]);
                    //
                    if (dftPtr->getParametersObject().spinPolarized == 1)
                      {
                        rhoOutSpinPolarized[2 * q_point] +=
                          rhoRecvd[iSymm][globalCellId[cell->id()]][proc]
                                  [2 * point];
                        rhoOutSpinPolarized[2 * q_point + 1] +=
                          rhoRecvd[iSymm][globalCellId[cell->id()]][proc]
                                  [2 * point + 1];
                      }
                    else
                      rhoOut[q_point] +=
                        rhoRecvd[iSymm][globalCellId[cell->id()]][proc][point];
                    if (isGradDensityDataRequired)
                      {
                        if (dftPtr->getParametersObject().spinPolarized == 1)
                          {
                            for (dftfe::uInt j = 0; j < 6; ++j)
                              gradRhoOutSpinPolarized[6 * q_point + j] +=
                                gradRhoRecvd[iSymm][globalCellId[cell->id()]]
                                            [proc][6 * point + j];
                          }
                        else
                          {
                            for (dftfe::uInt j = 0; j < 3; ++j)
                              gradRhoOut[3 * q_point + j] +=
                                gradRhoRecvd[iSymm][globalCellId[cell->id()]]
                                            [proc][3 * point + j];
                          }
                      }
                  }
                if (dftPtr->getParametersObject().spinPolarized == 1)
                  {
                    dftPtr->d_densityOutQuadValues[0][iCell * num_quad_points +
                                                      q_point] =
                      rhoOutSpinPolarized[2 * q_point] +
                      rhoOutSpinPolarized[2 * q_point + 1];
                    dftPtr->d_densityOutQuadValues[1][iCell * num_quad_points +
                                                      q_point] =
                      rhoOutSpinPolarized[2 * q_point] -
                      rhoOutSpinPolarized[2 * q_point + 1];
                  }
                else
                  dftPtr->d_densityOutQuadValues[0][iCell * num_quad_points +
                                                    q_point] = rhoOut[q_point];
                //
                if (isGradDensityDataRequired)
                  {
                    if (dftPtr->getParametersObject().spinPolarized == 1)
                      {
                        //
                        for (dftfe::uInt j = 0; j < 3; ++j)
                          dftPtr->d_gradDensityOutQuadValues
                            [0][iCell * num_quad_points * 3 + 3 * q_point + j] =
                            gradRhoOutSpinPolarized[6 * q_point + j] +
                            gradRhoOutSpinPolarized[6 * q_point + j + 3];
                        for (dftfe::uInt j = 0; j < 3; ++j)
                          dftPtr->d_gradDensityOutQuadValues
                            [1][iCell * num_quad_points * 3 + 3 * q_point + j] =
                            gradRhoOutSpinPolarized[6 * q_point + j] -
                            gradRhoOutSpinPolarized[6 * q_point + j + 3];
                      }
                    else
                      {
                        for (dftfe::uInt j = 0; j < 3; ++j)
                          dftPtr->d_gradDensityOutQuadValues
                            [0][iCell * num_quad_points * 3 + 3 * q_point + j] =
                            gradRhoOut[3 * q_point + j];
                      }
                  }
              }
            ++iCell;
          }
      }
    //=============================================================================================================================================
    //			Free up some memory by getting rid of density history beyond what is
    // required by mixing scheme
    //=============================================================================================================================================
  }
  //=============================================================================================================================================
  //=============================================================================================================================================
  //				Following routine computes density at all the transformed points
  // received from other processors 				        and scatters the density
  // back to the corresponding processors
  //=============================================================================================================================================
  //=============================================================================================================================================
  template <dftfe::utils::MemorySpace memorySpace>
  void
  symmetryClass<memorySpace>::computeLocalrhoOut()
  {
    std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
      (1 + dftPtr->getParametersObject().spinPolarized) *
      dftPtr->d_kPointWeights.size());

    const dftfe::uInt localVectorSize =
      dftPtr->matrix_free_data.get_vector_partitioner()->locally_owned_size();

    distributedCPUVec<dataTypes::number> eigenVectorsFlattenedArrayFullBlock;
    vectorTools::createDealiiVector<dataTypes::number>(
      dftPtr->matrix_free_data.get_vector_partitioner(),
      dftPtr->d_numEigenValues,
      eigenVectorsFlattenedArrayFullBlock);

    for (dftfe::uInt kPoint = 0;
         kPoint < (1 + dftPtr->getParametersObject().spinPolarized) *
                    dftPtr->d_kPointWeights.size();
         ++kPoint)
      {
        eigenVectors[kPoint].resize(dftPtr->d_numEigenValues);
        for (dftfe::uInt i = 0; i < dftPtr->d_numEigenValues; ++i)
          eigenVectors[kPoint][i].reinit(dftPtr->d_tempEigenVec);

        for (dftfe::uInt iNode = 0; iNode < localVectorSize; ++iNode)
          for (dftfe::uInt iWave = 0; iWave < dftPtr->d_numEigenValues; ++iWave)
            eigenVectorsFlattenedArrayFullBlock.local_element(
              iNode * dftPtr->d_numEigenValues + iWave) =
              dftPtr->d_eigenVectorsFlattenedHost
                [kPoint * localVectorSize * dftPtr->d_numEigenValues +
                 iNode * dftPtr->d_numEigenValues + iWave];
        eigenVectorsFlattenedArrayFullBlock.update_ghost_values();
        dftPtr->constraintsNoneDataInfo.distribute(
          eigenVectorsFlattenedArrayFullBlock, dftPtr->d_numEigenValues);


#ifdef USE_COMPLEX
        vectorTools::copyFlattenedDealiiVecToSingleCompVec(
          eigenVectorsFlattenedArrayFullBlock,
          dftPtr->d_numEigenValues,
          std::make_pair(0, dftPtr->d_numEigenValues),
          dftPtr->localProc_dof_indicesReal,
          dftPtr->localProc_dof_indicesImag,
          eigenVectors[kPoint]);
#endif
      }
    //
    const dealii::Quadrature<3> &quadrature =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    const dftfe::uInt num_quad_points = quadrature.size();
    totPoints                         = recvdData1[0].size();
    double                              px, py, pz;
    std::vector<dealii::Vector<double>> tempPsiAlpha, tempPsiBeta;
    std::vector<std::vector<dealii::Tensor<1, 3, double>>> tempGradPsi,
      tempGradPsiTempAlpha, tempGradPsiTempBeta;
    std::vector<dealii::Point<3>> quadPointList;
    std::vector<double>           sendData, recvdData;
    //
    std::vector<double> rhoLocal, gradRhoLocal, rhoLocalSpinPolarized,
      gradRhoLocalSpinPolarized;
    std::vector<double> rhoTemp, gradRhoTemp, rhoTempSpinPolarized,
      gradRhoTempSpinPolarized;
    rhoLocal.resize(totPoints, 0.0);
    rhoTemp.resize(totPoints, 0.0);
    //
    if (dftPtr->getParametersObject().spinPolarized == 1)
      {
        rhoLocalSpinPolarized.resize(2 * totPoints, 0.0);
        rhoTempSpinPolarized.resize(2 * totPoints, 0.0);
      }
    //
    bool isGradDensityDataRequired =
      (dftPtr->d_excManagerPtr->getExcSSDFunctionalObj()
         ->getDensityBasedFamilyType() == densityFamilyType::GGA);
    ;

    if (isGradDensityDataRequired)
      {
        gradRhoLocal.resize(3 * totPoints, 0.0);
        gradRhoTemp.resize(3 * totPoints, 0.0);
        if (dftPtr->getParametersObject().spinPolarized)
          {
            gradRhoLocalSpinPolarized.resize(6 * totPoints, 0.0);
            gradRhoTempSpinPolarized.resize(6 * totPoints, 0.0);
          }
      }
    dftfe::uInt numPointsDone = 0, numGroupsDone = 0;
    for (dftfe::uInt proc = 0; proc < n_mpi_processes; ++proc)
      {
        //
        for (dftfe::uInt iGroup = 0; iGroup < recv_size0[proc]; ++iGroup)
          {
            //
            const dftfe::uInt numPoint = recvdData2[numGroupsDone + iGroup];
            const dftfe::uInt cellId   = recvdData0[numGroupsDone + iGroup];
            //
            tempPsiAlpha.resize(numPoint);
            tempPsiBeta.resize(numPoint);
            quadPointList.resize(numPoint);
            //
            if (isGradDensityDataRequired)
              {
                tempGradPsi.resize(numPoint);
                tempGradPsiTempAlpha.resize(numPoint);
                tempGradPsiTempBeta.resize(numPoint);
              }
            for (dftfe::uInt iList = 0; iList < numPoint; ++iList)
              {
                //
                px = recvdData1[0][numPointsDone + iList];
                py = recvdData1[1][numPointsDone + iList];
                pz = recvdData1[2][numPointsDone + iList];
                //
                const dealii::Point<3> pointTemp(px, py, pz);
                quadPointList[iList] = pointTemp;
                tempPsiAlpha[iList].reinit(2);
                tempPsiBeta[iList].reinit(2);
                //
                if (isGradDensityDataRequired)
                  {
                    tempGradPsi[iList].resize(2);
                    tempGradPsiTempAlpha[iList].resize(2);
                    tempGradPsiTempBeta[iList].resize(2);
                  }
              } // loop on points
            //
            //
            dealii::Quadrature<3> quadRule(quadPointList);
            dealii::FEValues<3>   fe_values(dftPtr->FEEigen,
                                          quadRule,
                                          dealii::update_values |
                                            dealii::update_gradients |
                                            dealii::update_JxW_values |
                                            dealii::update_quadrature_points);
            fe_values.reinit(dealIICellId[cellId]);
            const dftfe::uInt iSymm = recvdData3[numGroupsDone + iGroup];
            //
            //
            //=============================================================================================================================================
            //				             Sum over the star of the k point and the
            // bands
            //				  	     Rho(r) = \sum_(n, Sk) | Psi (n, Sr + tau ) |^2
            //=============================================================================================================================================
            for (dftfe::uInt kPoint = 0;
                 kPoint < (dftPtr->d_kPointWeights.size());
                 ++kPoint)
              {
                if (symmUnderGroup[kPoint][iSymm] == 1)
                  {
                    for (dftfe::uInt i = 0; i < (dftPtr->d_numEigenValues); ++i)
                      {
                        double partialOccupancyAlpha =
                          (dftPtr->d_partialOccupancies)[kPoint][i];
                        double partialOccupancyBeta =
                          (dftPtr->d_partialOccupancies)
                            [kPoint]
                            [i + dftPtr->getParametersObject().spinPolarized *
                                   (dftPtr->d_numEigenValues)];
                        //
                        fe_values.get_function_values(
                          (eigenVectors[(1 + dftPtr->getParametersObject()
                                               .spinPolarized) *
                                        kPoint][i]),
                          tempPsiAlpha);
                        if (dftPtr->getParametersObject().spinPolarized == 1)
                          fe_values.get_function_values(
                            (eigenVectors[(1 + dftPtr->getParametersObject()
                                                 .spinPolarized) *
                                            kPoint +
                                          1][i]),
                            tempPsiBeta);
                        //
                        if (isGradDensityDataRequired)
                          {
                            fe_values.get_function_gradients(
                              (eigenVectors[(1 + dftPtr->getParametersObject()
                                                   .spinPolarized) *
                                            kPoint][i]),
                              tempGradPsiTempAlpha);
                            if (dftPtr->getParametersObject().spinPolarized ==
                                1)
                              fe_values.get_function_gradients(
                                (eigenVectors[(1 + dftPtr->getParametersObject()
                                                     .spinPolarized) *
                                                kPoint +
                                              1][i]),
                                tempGradPsiTempBeta);
                          }
                        //
                        for (dftfe::uInt iList = 0; iList < numPoint; ++iList)
                          {
                            //
                            if (dftPtr->getParametersObject().spinPolarized ==
                                1)
                              {
                                rhoTempSpinPolarized[2 *
                                                     (numPointsDone + iList)] +=
                                  1.0 / (double(numSymmUnderGroup[kPoint])) *
                                  partialOccupancyAlpha *
                                  (dftPtr->d_kPointWeights)[kPoint] *
                                  (tempPsiAlpha[iList](0) *
                                     tempPsiAlpha[iList](0) +
                                   tempPsiAlpha[iList](1) *
                                     tempPsiAlpha[iList](1));
                                rhoTempSpinPolarized[2 *
                                                       (numPointsDone + iList) +
                                                     1] +=
                                  1.0 / (double(numSymmUnderGroup[kPoint])) *
                                  partialOccupancyBeta *
                                  (dftPtr->d_kPointWeights)[kPoint] *
                                  (tempPsiBeta[iList](0) *
                                     tempPsiBeta[iList](0) +
                                   tempPsiBeta[iList](1) *
                                     tempPsiBeta[iList](1));
                              }
                            else
                              rhoTemp[numPointsDone + iList] +=
                                1.0 / (double(numSymmUnderGroup[kPoint])) *
                                2.0 * partialOccupancyAlpha *
                                (dftPtr->d_kPointWeights)[kPoint] *
                                (tempPsiAlpha[iList](0) *
                                   tempPsiAlpha[iList](0) +
                                 tempPsiAlpha[iList](1) *
                                   tempPsiAlpha[iList](1));
                            if (isGradDensityDataRequired)
                              {
                                for (dftfe::uInt j = 0; j < 3; ++j)
                                  {
                                    tempGradPsi[iList][0][j] =
                                      tempGradPsiTempAlpha[iList][0][0] *
                                        symmMat[iSymm][0][j] +
                                      tempGradPsiTempAlpha[iList][0][1] *
                                        symmMat[iSymm][1][j] +
                                      tempGradPsiTempAlpha[iList][0][2] *
                                        symmMat[iSymm][2][j];
                                    tempGradPsi[iList][1][j] =
                                      tempGradPsiTempAlpha[iList][1][0] *
                                        symmMat[iSymm][0][j] +
                                      tempGradPsiTempAlpha[iList][1][1] *
                                        symmMat[iSymm][1][j] +
                                      tempGradPsiTempAlpha[iList][1][2] *
                                        symmMat[iSymm][2][j];
                                  }
                                if (dftPtr->getParametersObject()
                                      .spinPolarized == 1)
                                  {
                                    gradRhoTempSpinPolarized
                                      [6 * (numPointsDone + iList) + 0] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * partialOccupancyAlpha *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiAlpha[iList](0) *
                                         tempGradPsi[iList][0][0] +
                                       tempPsiAlpha[iList](1) *
                                         tempGradPsi[iList][1][0]);
                                    gradRhoTempSpinPolarized
                                      [6 * (numPointsDone + iList) + 1] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * partialOccupancyAlpha *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiAlpha[iList](0) *
                                         tempGradPsi[iList][0][1] +
                                       tempPsiAlpha[iList](1) *
                                         tempGradPsi[iList][1][1]);
                                    gradRhoTempSpinPolarized
                                      [6 * (numPointsDone + iList) + 2] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * partialOccupancyAlpha *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiAlpha[iList](0) *
                                         tempGradPsi[iList][0][2] +
                                       tempPsiAlpha[iList](1) *
                                         tempGradPsi[iList][1][2]);
                                    for (dftfe::uInt j = 0; j < 3; ++j)
                                      {
                                        tempGradPsi[iList][0][j] =
                                          tempGradPsiTempBeta[iList][0][0] *
                                            symmMat[iSymm][0][j] +
                                          tempGradPsiTempBeta[iList][0][1] *
                                            symmMat[iSymm][1][j] +
                                          tempGradPsiTempBeta[iList][0][2] *
                                            symmMat[iSymm][2][j];
                                        tempGradPsi[iList][1][j] =
                                          tempGradPsiTempBeta[iList][1][0] *
                                            symmMat[iSymm][0][j] +
                                          tempGradPsiTempBeta[iList][1][1] *
                                            symmMat[iSymm][1][j] +
                                          tempGradPsiTempBeta[iList][1][2] *
                                            symmMat[iSymm][2][j];
                                      }
                                    //
                                    gradRhoTempSpinPolarized
                                      [6 * (numPointsDone + iList) + 3] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * partialOccupancyBeta *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiBeta[iList](0) *
                                         tempGradPsi[iList][0][0] +
                                       tempPsiBeta[iList](1) *
                                         tempGradPsi[iList][1][0]);
                                    gradRhoTempSpinPolarized
                                      [6 * (numPointsDone + iList) + 4] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * partialOccupancyBeta *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiBeta[iList](0) *
                                         tempGradPsi[iList][0][1] +
                                       tempPsiBeta[iList](1) *
                                         tempGradPsi[iList][1][1]);
                                    gradRhoTempSpinPolarized
                                      [6 * (numPointsDone + iList) + 5] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * partialOccupancyBeta *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiBeta[iList](0) *
                                         tempGradPsi[iList][0][2] +
                                       tempPsiBeta[iList](1) *
                                         tempGradPsi[iList][1][2]);
                                  }
                                else
                                  {
                                    gradRhoTemp[3 * (numPointsDone + iList) +
                                                0] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * 2.0 * partialOccupancyAlpha *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiAlpha[iList](0) *
                                         tempGradPsi[iList][0][0] +
                                       tempPsiAlpha[iList](1) *
                                         tempGradPsi[iList][1][0]);
                                    gradRhoTemp[3 * (numPointsDone + iList) +
                                                1] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * 2.0 * partialOccupancyAlpha *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiAlpha[iList](0) *
                                         tempGradPsi[iList][0][1] +
                                       tempPsiAlpha[iList](1) *
                                         tempGradPsi[iList][1][1]);
                                    gradRhoTemp[3 * (numPointsDone + iList) +
                                                2] +=
                                      1.0 /
                                      (double(numSymmUnderGroup[kPoint])) *
                                      2.0 * 2.0 * partialOccupancyAlpha *
                                      (dftPtr->d_kPointWeights)[kPoint] *
                                      (tempPsiAlpha[iList](0) *
                                         tempGradPsi[iList][0][2] +
                                       tempPsiAlpha[iList](1) *
                                         tempGradPsi[iList][1][2]);
                                  }
                              }
                          } // loop on points list
                      }     // loop on eigenValues
                  } // if this symm is part of the group under this kpoint
              }     // loop on k Points
            //
            numPointsDone += numPoint;
          } // loop on group
        //
        numGroupsDone += recv_size0[proc];
      } // loop on proc
    //
    MPI_Allreduce(
      &rhoTemp[0], &rhoLocal[0], totPoints, MPI_DOUBLE, MPI_SUM, interpoolcomm);
    if (isGradDensityDataRequired)
      MPI_Allreduce(&gradRhoTemp[0],
                    &gradRhoLocal[0],
                    3 * totPoints,
                    MPI_DOUBLE,
                    MPI_SUM,
                    interpoolcomm);
    if (dftPtr->getParametersObject().spinPolarized == 1)
      {
        MPI_Allreduce(&rhoTempSpinPolarized[0],
                      &rhoLocalSpinPolarized[0],
                      2 * totPoints,
                      MPI_DOUBLE,
                      MPI_SUM,
                      interpoolcomm);
        if (isGradDensityDataRequired)
          MPI_Allreduce(&gradRhoTempSpinPolarized[0],
                        &gradRhoLocalSpinPolarized[0],
                        6 * totPoints,
                        MPI_DOUBLE,
                        MPI_SUM,
                        interpoolcomm);
      }
    //================================================================================================================================================
    //			      Now first prepare the flattened sending vectors and then MPI
    // scatter. 		         The essential idea here is that each processor
    // sends the computed density on the transformed points
    //                                back to all other processors.from which
    //                                the points came from.
    //================================================================================================================================================
    sendData.resize((1 + dftPtr->getParametersObject().spinPolarized) *
                    totPoints);
    if (dftPtr->getParametersObject().spinPolarized == 1)
      sendData = rhoLocalSpinPolarized;
    else
      sendData = rhoLocal;
    //
    typename dealii::DoFHandler<3>::active_cell_iterator cell;
    typename dealii::DoFHandler<3>::active_cell_iterator endc =
      (dftPtr->dofHandlerEigen).end();
    //
    for (dftfe::Int sendProc = 0; sendProc < n_mpi_processes; ++sendProc)
      {
        recvdData.resize(recv_size[sendProc]);
        MPI_Scatterv(&(sendData[0]),
                     &(recv_size1[0]),
                     &(mpi_offsets1[0]),
                     MPI_DOUBLE,
                     &(recvdData[0]),
                     recv_size[sendProc],
                     MPI_DOUBLE,
                     sendProc,
                     mpi_communicator);
        //
        cell               = (dftPtr->dofHandlerEigen).begin_active();
        dftfe::uInt offset = 0;
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
                  {
                    for (dftfe::uInt i = 0;
                         i < rhoRecvd[iSymm][globalCellId[cell->id()]][sendProc]
                               .size();
                         ++i)
                      rhoRecvd[iSymm][globalCellId[cell->id()]][sendProc][i] =
                        recvdData[offset + i];
                    //
                    offset +=
                      rhoRecvd[iSymm][globalCellId[cell->id()]][sendProc]
                        .size();
                  }
              }
          }
        recvdData.clear();
      }
    //
    if (isGradDensityDataRequired)
      {
        sendData.resize(3 * (1 + dftPtr->getParametersObject().spinPolarized) *
                        totPoints);
        if (dftPtr->getParametersObject().spinPolarized == 1)
          sendData = gradRhoLocalSpinPolarized;
        else
          sendData = gradRhoLocal;
        //
        for (dftfe::Int sendProc = 0; sendProc < n_mpi_processes; ++sendProc)
          {
            recvdData.resize(3 * recv_size[sendProc]);
            MPI_Scatterv(&(sendData[0]),
                         &(recvGrad_size1[0]),
                         &(mpiGrad_offsets1[0]),
                         MPI_DOUBLE,
                         &(recvdData[0]),
                         3 * recv_size[sendProc],
                         MPI_DOUBLE,
                         sendProc,
                         mpi_communicator);
            //
            cell               = (dftPtr->dofHandlerEigen).begin_active();
            dftfe::uInt offset = 0;
            //
            for (; cell != endc; ++cell)
              {
                if (cell->is_locally_owned())
                  {
                    for (dftfe::uInt iSymm = 0; iSymm < numSymm; ++iSymm)
                      {
                        for (dftfe::uInt i = 0;
                             i < gradRhoRecvd[iSymm][globalCellId[cell->id()]]
                                             [sendProc]
                                               .size();
                             ++i)
                          gradRhoRecvd[iSymm][globalCellId[cell->id()]]
                                      [sendProc][i] = recvdData[offset + i];
                        //
                        offset += gradRhoRecvd[iSymm][globalCellId[cell->id()]]
                                              [sendProc]
                                                .size();
                      }
                  }
              }
            recvdData.clear();
          }
      }
  } // end function
} // namespace dftfe
