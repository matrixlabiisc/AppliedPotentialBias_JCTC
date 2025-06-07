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

#include <headers.h>
#include <dftd.h>
#include <excManager.h>
#include "dftParameters.h"
#include <FEBasisOperations.h>
#include <AuxDensityMatrix.h>
#ifndef energyCalculator_H_
#  define energyCalculator_H_

namespace dftfe
{
  namespace internalEnergy
  {
    template <typename T>
    double
    computeFieldTimesDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                                                          &basisOperationsPtr,
      const dftfe::uInt                                    quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValues);

    template <typename T>
    double
    computeModGradPhiSq(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradFieldValues);

    template <typename T>
    double
    computeFieldTimesDensityResidual(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                                                          &basisOperationsPtr,
      const dftfe::uInt                                    quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesIn,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesOut);

    template <typename T>
    double
    computeFieldTimesDensity(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValues);

    template <typename T>
    double
    computeFieldTimesDensityResidual(
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<T, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtr,
      const dftfe::uInt quadratureId,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &fieldValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesIn,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &densityQuadValuesOut);

    void
    printEnergy(const double                      bandEnergy,
                const double                      totalkineticEnergy,
                const double                      totalexchangeEnergy,
                const double                      totalcorrelationEnergy,
                const double                      totalElectrostaticEnergy,
                const double                      dispersionEnergy,
                const double                      totalEnergy,
                const dftfe::uInt                 numberAtoms,
                const dealii::ConditionalOStream &pcout,
                const bool                        reproducibleOutput,
                const bool                        isPseudo,
                const dftfe::uInt                 verbosity,
                const dftParameters              &dftParams);

    double
    localBandEnergy(const std::vector<std::vector<double>> &eigenValues,
                    const std::vector<std::vector<double>> &partialOccupancies,
                    const std::vector<double>              &kPointWeights,
                    const double                            fermiEnergy,
                    const double                            fermiEnergyUp,
                    const double                            fermiEnergyDown,
                    const double                            TVal,
                    const dftfe::uInt                       spinPolarized,
                    const dealii::ConditionalOStream       &scout,
                    const MPI_Comm                         &interpoolcomm,
                    const dftfe::uInt                       lowerBoundKindex,
                    const dftfe::uInt                       verbosity,
                    const dftParameters                    &dftParams);

    double
    nuclearElectrostaticEnergyLocal(
      const distributedCPUVec<double>                     &phiTotRhoOut,
      const std::vector<std::vector<double>>              &localVselfs,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                  &smearedbNonTrivialAtomIds,
      const dealii::DoFHandler<3> &dofHandlerElectrostatic,
      const dealii::Quadrature<3> &quadratureElectrostatic,
      const dealii::Quadrature<3> &quadratureSmearedCharge,
      const std::map<dealii::types::global_dof_index, double>
        &atomElectrostaticNodeIdToChargeMap,
      const std::map<dealii::CellId, std::vector<double>>
                &uExternalQuadValuesNuclear,
      const bool smearedNuclearCharges       = false,
      const bool externalFieldOrSawToothFlag = false);

    double
    nuclearElectrostaticEnergyResidualLocalTemp(
      const distributedCPUVec<double>                     &phiTotRhoIn,
      const distributedCPUVec<double>                     &phiTotRhoOut,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                  &smearedbNonTrivialAtomIds,
      const dealii::DoFHandler<3> &dofHandlerElectrostatic,
      const dealii::Quadrature<3> &quadratureSmearedCharge,
      const std::map<dealii::types::global_dof_index, double>
                &atomElectrostaticNodeIdToChargeMap,
      const bool smearedNuclearCharges = false);
    double
    computeRepulsiveEnergy(
      const std::vector<std::vector<double>> &atomLocationsAndCharge,
      const bool                              isPseudopotential);
  } // namespace internalEnergy

  /**
   * @brief Calculates the ksdft problem total energy and its components
   *
   * @author Sambit Das, Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class energyCalculator
  {
  public:
    /**
     * @brief Constructor
     *
     * @param mpi_comm_parent parent mpi communicator
     * @param mpi_comm_domain mpi communicator of domain decomposition
     * @param interpool_comm mpi interpool communicator over k points
     * @param interBandGroupComm mpi interpool communicator over band groups
     */
    energyCalculator(const MPI_Comm      &mpi_comm_parent,
                     const MPI_Comm      &mpi_comm_domain,
                     const MPI_Comm      &interpool_comm,
                     const MPI_Comm      &interBandGroupComm,
                     const dftParameters &dftParams);

    /**
     * Computes total energy of the ksdft problem in the current state and also
     * prints the individual components of the energy
     *
     * @param dofHandlerElectrostatic p refined DoFHandler object used for re-computing
     * the electrostatic fields using the ground state electron density. If
     * electrostatics is not recomputed on p refined mesh, use
     * dofHandlerElectronic for this argument.
     * @param dofHandlerElectronic DoFHandler object on which the electrostatics for the
     * eigen solve are computed.
     * @param quadratureElectrostatic qudarature object for dofHandlerElectrostatic.
     * @param quadratureElectronic qudarature object for dofHandlerElectronic.
     * @param eigenValues eigenValues for each k point.
     * @param kPointWeights
     * @param fermiEnergy
     * @param funcX exchange functional object.
     * @param funcC correlation functional object.
     * @param phiTotRhoIn nodal vector field of total electrostatic potential using input
     * electron density to an eigensolve. This vector field is based on
     * dofHandlerElectronic.
     * @param phiTotRhoOut nodal vector field of total electrostatic potential using output
     * electron density to an eigensolve. This vector field is based on
     * dofHandlerElectrostatic.
     * @param rhoInValues cell quadrature data of input electron density to an eigensolve. This
     * data must correspond to quadratureElectronic.
     * @param rhoOutValues cell quadrature data of output electron density of an eigensolve. This
     * data must correspond to quadratureElectronic.
     * @param rhoOutValuesElectrostatic cell quadrature data of output electron density of an eigensolve
     * evaluated on a p refined mesh. This data corresponds to
     * quadratureElectrostatic.
     * @param gradRhoInValues cell quadrature data of input gradient electron density
     * to an eigensolve. This data must correspond to quadratureElectronic.
     * @param gradRhoOutValues cell quadrature data of output gradient electron density
     * of an eigensolve. This data must correspond to quadratureElectronic.
     * @param localVselfs peak vselfs of local atoms in each vself bin
     * @param atomElectrostaticNodeIdToChargeMap map between locally processor atom global node ids
     * of dofHandlerElectrostatic to atom charge value.
     * @param numberGlobalAtoms
     * @param lowerBoundKindex global k index of lower bound of the local k point set in the current pool
     * @param if scf is converged
     * @param print
     * @param uExternalQuadValuesNuclear quadrature values of external potential at smeared charge quadrature rule
     * @param uExternalQuadValuesRho quadrature values of external potential at density Quadrature Rule
     * @param smearedNuclearCharges
     * @param externalFieldOrSawToothFlag include the external field/ saw tooth potential in energy computation
     * @return total energy
     */
    double
    computeEnergy(
      const std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &basisOperationsPtr,
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                             &basisOperationsPtrElectro,
      const dftfe::uInt                       densityQuadratureID,
      const dftfe::uInt                       densityQuadratureIDElectro,
      const dftfe::uInt                       smearedChargeQuadratureIDElectro,
      const dftfe::uInt                       lpspQuadratureIDElectro,
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<std::vector<double>> &partialOccupancies,
      const std::vector<double>              &kPointWeights,
      const double                            fermiEnergy,
      const double                            fermiEnergyUp,
      const double                            fermiEnergyDown,
      const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      const dispersionCorrection                    &dispersionCorr,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiTotRhoInValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                      &phiTotRhoOutValues,
      const distributedCPUVec<double> &phiTotRhoOut,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &gradPhiTotRhoOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauOutValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &rhoOutValuesLpsp,
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCInRepresentationPtr,
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        auxDensityXCOutRepresentationPtr,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                             &smearedbNonTrivialAtomIds,
      const std::vector<std::vector<double>> &localVselfs,
      const std::map<dealii::CellId, std::vector<double>> &pseudoLocValues,
      const std::map<dealii::types::global_dof_index, double>
                       &atomElectrostaticNodeIdToChargeMap,
      const dftfe::uInt numberGlobalAtoms,
      const dftfe::uInt lowerBoundKindex,
      const dftfe::uInt scfConverged,
      const bool        print,
      const std::map<dealii::CellId, std::vector<double>>
        &uExternalValuesNuclear,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                &uExternalQuadValuesRho,
      const bool smearedNuclearCharges       = false,
      const bool externalFieldOrSawToothFlag = false);

    double
    computeEnergyResidual(
      const std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &basisOperationsPtr,
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtrElectro,
      const dftfe::uInt densityQuadratureID,
      const dftfe::uInt densityQuadratureIDElectro,
      const dftfe::uInt smearedChargeQuadratureIDElectro,
      const dftfe::uInt lpspQuadratureIDElectro,
      const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &phiTotRhoInValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                      &phiTotRhoOutValues,
      const distributedCPUVec<double> &phiTotRhoIn,
      const distributedCPUVec<double> &phiTotRhoOut,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauOutValues,
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        AuxDensityXCInRepresentationPtr,
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        AuxDensityXCOutRepresentationPtr,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                             &smearedbNonTrivialAtomIds,
      const std::vector<std::vector<double>> &localVselfs,
      const std::map<dealii::types::global_dof_index, double>
                &atomElectrostaticNodeIdToChargeMap,
      const bool smearedNuclearCharges);


    void
    computeTotalElectrostaticsEnergy(
      const std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &basisOperationsPtr,
      const std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                       &basisOperationsPtrElectro,
      const dftfe::uInt densityQuadratureID,
      const dftfe::uInt densityQuadratureIDElectro,
      const dftfe::uInt smearedChargeQuadratureIDElectro,
      const dftfe::uInt lpspQuadratureIDElectro,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                      &phiTotRhoInValues,
      const distributedCPUVec<double> &phiTotRhoIn,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                                          &rhoInValuesLpsp,
      const std::map<dealii::CellId, std::vector<double>> &smearedbValues,
      const std::map<dealii::CellId, std::vector<dftfe::uInt>>
                                             &smearedbNonTrivialAtomIds,
      const std::vector<std::vector<double>> &localVselfs,
      const std::map<dealii::CellId, std::vector<double>> &pseudoLocValues,
      const std::map<dealii::types::global_dof_index, double>
        &atomElectrostaticNodeIdToChargeMap,
      const std::map<dealii::CellId, std::vector<double>>
        &uExternalValuesNuclear,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                &uExternalQuadValuesRho,
      const bool smearedNuclearCharges,
      const bool externalFieldOrSawToothFlag);

    void
    computeXCEnergyTermsSpinPolarized(
      const std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
                                                    &basisOperationsPtr,
      const dftfe::uInt                              quadratureId,
      const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityInValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradDensityOutValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &tauInValues,
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
        AuxDensityXCInRepresentationPtr,
      std::shared_ptr<AuxDensityMatrix<memorySpace>>
              auxDensityXCOutRepresentationPtr,
      double &exchangeEnergy,
      double &correlationEnergy,
      double &excCorrPotentialTimesRho);

    double
    computeEntropicEnergy(
      const std::vector<std::vector<double>> &eigenValues,
      const std::vector<std::vector<double>> &partialOccupancies,
      const std::vector<double>              &kPointWeights,
      const double                            fermiEnergy,
      const double                            fermiEnergyUp,
      const double                            fermiEnergyDown,
      const bool                              isSpinPolarized,
      const bool                              isConstraintMagnetization,
      const double                            temperature) const;



  private:
    const MPI_Comm d_mpiCommParent;
    const MPI_Comm mpi_communicator;
    const MPI_Comm interpoolcomm;
    const MPI_Comm interBandGroupComm;

    const dftParameters &d_dftParams;

    /// parallel message stream
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // energyCalculator_H_
