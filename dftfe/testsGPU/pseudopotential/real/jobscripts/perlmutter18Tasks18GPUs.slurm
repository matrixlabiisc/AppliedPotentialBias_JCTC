#!/global/homes/d/dsambit/perlmutter/bin/rc
#SBATCH -A m2360_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name realtests
#SBATCH -t 1:00:00
#SBATCH -n 18
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

srun  ./dftfe Input_MD_0.prm > output_MD_0
srun  ./dftfe Input_MD_1.prm > output_MD_1
srun  ./dftfe Input_MD_2.prm > output_MD_2
srun  ./dftfe parameterFileMg2x_1.prm > outputMg2x_1
srun  ./dftfe parameterFileMg2x_1_spingpu.prm > outputMg2x_1_spin_gpu
srun  ./dftfe parameterFileMg2x_2.prm > outputMg2x_2
srun  ./dftfe parameterFileMg2x_3.prm > outputMg2x_3
srun  ./dftfe parameterFileMg2x_4.prm > outputMg2x_4
srun  ./dftfe parameterFileMg2x_5.prm > outputMg2x_5
srun  ./dftfe parameterFileMg2x_6.prm > outputMg2x_6
srun  ./dftfe parameterFileMg2x_7.prm > outputMg2x_7
srun  ./dftfe parameterFileMg2x_12.prm > outputMg2x_12
srun  ./dftfe parameterFileMg2x_13.prm > outputMg2x_13
srun  ./dftfe parameterFileMg2x_17.prm > outputMg2x_17
srun  ./dftfe parameterFileBe.prm > outputBe
srun  ./dftfe parameterFile_LLZO.prm > outputLLZO
