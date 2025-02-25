#!/bin/bash

#SBATCH --gpus=1
#SBATCH --mem=128GB
#SBATCH --constraint=40gb
#SBATCH --cpus-per-task=16
#SBATCH -o ./logs/array/%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=p.chormai@tu-berlin.de

nvidia-smi

TASK_VALUE=`sed -n ${SLURM_ARRAY_TASK_ID}p $SEEDFILE`
CMD=${@/\{\}/$TASK_VALUE}

echo "-- Job Array Information ($SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT)"
echo "JOB_ARRAY_SEEDFILE=$SEEDFILE"
echo "TASK_VALUE=$TASK_VALUE"
echo "CMD:"
echo " $CMD"
echo "--"

$CMD