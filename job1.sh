#!/bin/bash
#SBATCH --job-name=vasptest
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuyue1995@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=55gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:8
#SBATCH --time=24:00:00

for i in $(seq 12 20)
do
   python -u run.py 
   cp tmp.txt $i.txt 
done 
