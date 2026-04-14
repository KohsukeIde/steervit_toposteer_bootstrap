#!/bin/bash
#SBATCH --job-name=franca_b_IN21k   # job name
#SBATCH -A yic@h100
#SBATCH -C h100                         # to target H100 nodes
#SBATCH --nodes=16                       # number of nodes
#SBATCH --ntasks-per-node=4             # number of MPI tasks per node (= here number of GPUs per node)
#SBATCH --gres=gpu:4                    # number of GPUs per task (max total 4 for H100 nodes)
#SBATCH --cpus-per-task=12              # number of CPUs per task (1/4 of CPUs here)
#SBATCH --hint=nomultithread            # hyperthreading disabled
#SBATCH --time=20:00:00                 # maximum execution time requested (HH:MM:SS)
#SBATCH --output=slurm_jobs_logs/stdout/Imagenet_21k/francaB.out
#SBATCH --error=slurm_jobs_logs/stdout/Imagenet_21k/francaB.err
#SBATCH --qos=qos_gpu_h100-t3

cd $WORK/Franca
mkdir -p slurm_jobs_logs/stdout/Imagenet_21k

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# By convention, use a port number between 10001 and 20000
export MASTER_PORT=18530

export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

export TRITON_CACHE_DIR=$SCRATCH/.triton
export HYDRA_FULL_ERROR=1

export TMPDIR=$JOBSCRATCH

# Cleaning up modules loaded interactively and inherited by default
module purge

# Activating the modules
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONPATH=.

set -x

srun python franca/train/train.py \
  --config-file=franca/configs/train/IN21K/vitb14.yaml  \
  --output-dir= <path to output>/Imagenet21k/francaB/
