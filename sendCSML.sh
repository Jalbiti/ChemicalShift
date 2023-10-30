#!/bin/bash
#SBATCH -c 1
#SBATCH -J CSML
#SBATCH --time=2-00:00:00
#SBATCH -o log2.out
#SBATCH -e log2.err
#SBATCH -p AMDMEM
export PGI_FASTMATH_CPU=sandybridge

#cd $SLURM_SUBMIT_DIR

#__conda_setup="$(/orozco/projects/NMR_i-motif/condaRTX/bin/conda shell.bash hook 2> /dev/null)"; eval "$__conda_setup"
#conda activate python36

module load ANACONDA2
conda activate triplex

python3.6 CSML_agosto2023.py
