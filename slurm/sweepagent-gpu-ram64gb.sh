#!/bin/bash

#SBATCH --mem=64GB
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH -o ./logs/array/%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=p.chormai@tu-berlin.de
#SBATCH --exclude=head[073-076]

nvidia-smi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB__SERVICE_WAIT=300

./runpy wandb agent --count 1 "$@"
