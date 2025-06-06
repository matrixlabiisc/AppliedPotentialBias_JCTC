#!/global/homes/d/dsambit/perlmutter/bin/rc
#SBATCH -A m2360_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name realtests
#SBATCH -t 00:40:00
#SBATCH -n 6
#SBATCH --ntasks-per-node=3
#SBATCH -c 32
#SBATCH --gpus-per-node=3
#SBATCH --gpu-bind=map_gpu:0*1,1*1,2*1,3*0

module load gcc-native/12.3
module load PrgEnv-gnu
module load cudatoolkit
module load cray-libsci
module load cmake
module load nccl

SLURM_CPU_BIND='cores'
OMP_NUM_THREADS=1
MPICH_GPU_SUPPORT_ENABLED=1
LD_LIBRARY_PATH = $LD_LIBRARY_PATH:$WD/env2/lib
LD_LIBRARY_PATH = $LD_LIBRARY_PATH:$WD/env2/lib64

srun ./dftfe parameterFileN2_1.prm > outputN2_1
srun ./dftfe parameterFileN2_2.prm > outputN2_2
srun ./dftfe parameterFileN2_3.prm > outputN2_3
srun ./dftfe parameterFileN2_4.prm > outputN2_4
srun ./dftfe parameterFileN2_5.prm > outputN2_5
srun ./dftfe parameterFileMg2x_8.prm > outputMg2x_8
srun ./dftfe parameterFileMg2x_9.prm > outputMg2x_9
srun ./dftfe parameterFileMg2x_10.prm > outputMg2x_10
srun ./dftfe parameterFileMg2x_11.prm > outputMg2x_11
srun ./dftfe parameterFileMg2x_14.prm > outputMg2x_14
srun ./dftfe parameterFileMg2x_15.prm > outputMg2x_15
srun ./dftfe Pt3Ni_hubbard_spin.prm            > outputPt3Ni_hubbard_spin
srun ./dftfe Pt3Ni_hubbard_spin_mixedPrec.prm > outputPt3Ni_hubbard_spin_mixedPrec
