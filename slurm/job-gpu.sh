#!/bin/bash

#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=12
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=p.chormai@tu-berlin.de

nvidia-smi

"$@"
