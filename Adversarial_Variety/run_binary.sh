#!/bin/bash
#SBATCH --mail-user=daniel.afrimi@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:3g
#SBATCH --mem=5g
#SBATCH --time=2-0
#SBATCH -c2
#SBATCH --output=/cs/labs/daphna/daniel023/Adversarial_Variety/model_logs/sbatch_%J.out

module load cuda
module load torch

source /cs/labs/daphna/daniel023/lab_env/bin/activate
cd /cs/labs/daphna/daniel023/Adversarial_Variety/

python3 main.py

