#!/bin/bash

#SBATCH --gpus=1
#SBATCH -o ./logs/%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=p.chormai@tu-berlin.de

nvidia-smi

./runpy "$@"