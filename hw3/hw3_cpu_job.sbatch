#!/bin/bash
#SBATCH -J hw3_cpu
#SBATCH -p ct56
#SBATCH -A ACD114118
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:03:00
#SBATCH -o hw3_cpu.%j.out
#SBATCH -e hw3_cpu.%j.err

set -euo pipefail
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "CWD:   $(pwd)"
date

srun ./hw3_cpu -0.522 2.874 1.340 0 0 0 64 64 00.png

date
EOF
