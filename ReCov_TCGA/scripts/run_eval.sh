#!/bin/bash
#for prospero
#SBATCH --gres=gpu:1   # Request GPU "generic resources"
#SBATCH --time=23:00:00      #23:00:00
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=200G          # Memory proportional to GPUs: 32000 Cedar
#SBATCH --output=/home/vramanathan/Projects/TCGA_MIL/logs/tcgaeval_%j.log
#SBATCH --error=/home/vramanathan/Projects/TCGA_MIL/logs/tcgaeval_error_%j.log

# source activate tiger
# cd "/home/vramanathan/Projects/TCGA_MIL"
source activate graph
# cd "/home/vramanathan/Projects/TCGA_MIL/Patch-GCN"
cd "/home/vramanathan/Projects/TCGA_MIL/ReCov_TCGA"

# python generate_features.py
# python TCGA_extract_features.py
# python train.py -c /home/vramanathan/Projects/TCGA_MIL/configs/attn_mil_tcga.yml
# python train.py -c /home/vramanathan/Projects/TCGA_MIL/configs/attn_mil_tcga_small.yml
# python tcga_evaluate_noisy.py
python evaluate_recov.py