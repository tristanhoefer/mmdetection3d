#!/usr/bin/env bash

set -x

#SBATCH -J pointpillar_simdata
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00


PARTITION="reissdorf"
JOB_NAME="pointpillar_simdata"
CONFIG="configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d_myconfig.py"
WORK_DIR="/globalwork/data/6GEM/2022-05-19_vlp_32_less_pedestrians/Experiments/"
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:1 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=6 \

    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
