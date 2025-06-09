#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclude=wn208,wn210,wn220,wn221,wn224
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/rag-project-%J.out
#SBATCH --error=logs/rag-project-%J.err
#SBATCH --job-name="RAG project"

srun singularity exec --nv ./rag.sif python \
  ./src/main.py "Which game engine was Clair Obscur: Expedition 33 developed with?"