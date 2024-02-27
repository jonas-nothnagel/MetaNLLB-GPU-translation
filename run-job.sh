#!/bin/bash

set -xe

srun \
  --gpus=1 \
  --mem=24GB \
  --container-image=/data/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
  --container-workdir=`pwd` \
  --container-mounts=/data/nothnagel/MetaNLLB-GPU-translation:/data/nothnagel/MetaNLLB-GPU-translation \
  ./job.sh