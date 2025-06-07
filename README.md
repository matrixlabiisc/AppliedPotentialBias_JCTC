# AppliedPotentialBias_JCTC
Input and output files of various results published in the paper. The arxiv version is found at [arxiv](https://arxiv.org/abs/2504.00998).


Pseudopotential data is not present. The pseudopotentials can be obtained from [SPMS database](https://github.com/SPARC-X/SPMS-psps).

The DFT-FE code used here is not par of the release branch. The version of DFT-FE to reproduce the results can be found in this repository. These capabilities will be implemented in future release of DFT-FE. The appropriate changes to the input files will be updated in this repository when the capabilities are merged in the relase branch of DFT-FE. The release branch of DFT-FE is found at [DFT-FE release](https://github.com/dftfeDevelopers/dftfe) and scripts for installation can be found at [DFT-FE install script](https://github.com/dftfeDevelopers/install_DFTFE)


The repository contains 5 folders:
## CPD_LCPDComparison
 This folder contains the input output files for running the CPD and L-CPD corresponding to Figure 9 in the main manuscript.
## dipoleComparison
This folder contains the QE and DFT-FE input and output files corresponding to section 3.1 and Figure 5 in the main manuscript.
## planarAverageComparison
This folder contains the DFT-FE input and output file corresponding to section 3.2 and Figure 6,7 in the main manuscript.
## surfaceEnergyAdsorptionEnergyComparison
The folder contains the DFT-FE input and output files correspoding to section 3.3 in the main manuscript.
## dftfe
The source code to reproduce the DFT-FE results mentioned in the paper. These capabilities will be part of release version in the next couple of months.


