#!/bin/bash

#SBATCH --gpus=1
#SBATCH -o ./logs/array/%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=p.chormai@tu-berlin.de


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB__SERVICE_WAIT=300

nvidia-smi

# copy data
if [[ $COPY_DATA -eq 1 ]]; then
    echo "copy data.."
    time cp ~/datasets/imagenet.sqfs /tmp/
fi


./runpy wandb agent "$@"
