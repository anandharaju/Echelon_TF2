#!/bin/bash
#SBATCH --account=def-wangk
#SBATCH --time=8:00:00
#SBATCH --job-name=TF2_1

#SBATCH --cpus-per-task=8
#SBATCH --cores-per-socket=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=8G

#SBATCH -o /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.out
#SBATCH -e /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.err
#SBATCH --mail-user=aduraira@sfu.ca
#SBATCH --mail-type=ALL

source ../../tf2/bin/activate
python main.py
