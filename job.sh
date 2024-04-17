#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-FILTRO.o%j
#SBATCH --error=submit-FILTRO.e%j
#SBATCH -A cuda
#SBATCH -p cuda
### Se piden 4 GPUs 
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/12.2.2/bin:$PATH



./filtrar.exe IMG01.jpg OutSeq.jpg OutKernel.jpg 

