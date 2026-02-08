#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=staging
#SBATCH --time=10:00:00
#SBATCH --mem=180GB
#SBATCH --output=script_logging/slurm_finalization_%j.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
base_dir=corpus_datasets/dataset_creation_heydar
output_dir=corpus_datasets/dataset_creation_heydar
run=all

python $HOME/total-recall-rag/c4_post_qrel_generation/dataset_finalization.py \
    --base-dir $base_dir \
    --output-dir $output_dir \
    --run $run
