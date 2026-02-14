#!/bin/bash
#SBATCH --job-name=DefectSim
#SBATCH -A special00007
#SBATCH --output=%x_%j.out     
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1           
#SBATCH --gres=gpu:a100:1      
#SBATCH --cpus-per-task=16       
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-20:00:00
#SBATCH -C a100

# 1. Setup paths
PROJECT_DIR="/home/dd12hesa/projects/phase_field_grain"
WORK_DIR="/work/scratch/dd12hesa"
JOB_DIR="$WORK_DIR/${SLURM_JOB_ID}_defects"

source /home/dd12hesa/venvs/py311/bin/activate

echo "Creating job directory: $JOB_DIR"
mkdir -p "$JOB_DIR"
cd "$JOB_DIR"

cp "$PROJECT_DIR/multi_grain_v1.py" .
cp -r "$PROJECT_DIR/solver" .

python -B multi_grain_v1.py