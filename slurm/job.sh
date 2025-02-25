#!/bin/bash

#SBATCH --gpus=1
#SBATCH --mem=128GB
#SBATCH --constraint=40gb
#SBATCH --cpus-per-task=16
#SBATCH -o ./logs/%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=p.chormai@tu-berlin.de

nvidia-smi

"$@"