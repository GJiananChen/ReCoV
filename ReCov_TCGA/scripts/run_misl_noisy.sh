#!/bin/bash
#for prospero
# SBATCH --gres=gpu:a100_3g.39gb:1   # Request GPU "generic resources"
#SBATCH --gres=gpu:a100:1   # Request GPU "generic resources"
#SBATCH --time=72:00:00      #23:00:00
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=200G          # Memory proportional to GPUs: 32000 Cedar
#SBATCH --output=/home/vramanathan/Projects/TCGA_MIL/logs/tcgamil_%j.log
#SBATCH --error=/home/vramanathan/Projects/TCGA_MIL/logs/tcgamil_error_%j.log
#SBATCH --partition=amgrp

# source activate tiger
source activate graph
cd "/home/vramanathan/Projects/TCGA_MIL/ReCov_TCGA"

export CUDA_VISIBLE_DEVICES=0
# python generate_features.py
# python TCGA_extract_features.py
# python train.py -c /home/vramanathan/Projects/TCGA_MIL/configs/attn_mil_tcga.yml
# python train.py -c /home/vramanathan/Projects/TCGA_MIL/configs/attn_mil_tcga_small.yml
# python train_attnmil_noisy.py
# python main_recov.py
# python main_recov_v2.py --model_type amil
python main_recov_nll.py --model_type amil
