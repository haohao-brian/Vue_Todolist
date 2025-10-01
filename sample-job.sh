#!/bin/bash
#SBATCH --account=ACD114118       # (-A) iService Project ID
#SBATCH --job-name=sbatch             # (-J) Job name
#SBATCH --partition=ctest       # (-p) Slurm partition
#SBATCH --nodes=2                     # (-N) Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=1             # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=10         # Maximum number of tasks on each node
#SBATCH --time=00:30:00               # (-t) Wall time limit (days-hrs:min:sec)
#SBATCH --output=job-%j.out           # (-o) Path to the standard output file
#SBATCH --error=job-%j.err            # (-e) Path to the standard error file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=brian.chunhao@gmail.com  # Where to send mail.  Set this to your email address

./hw1 samples/05.txt > a.output
