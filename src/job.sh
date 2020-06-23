#!/bin/bash
#SBATCH --account=def-wangk
#SBATCH --time=18:00:00

#SBATCH --cpus-per-task=8
#SBATCH --cores-per-socket=8
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=64GB

#SBATCH -o /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.out
#SBATCH -e /home/aduraira/projects/def-wangk/aduraira/cc_out/job%j.err
#SBATCH --mail-user=aduraira@sfu.ca
#SBATCH --mail-type=ALL

source ../../tf2/bin/activate
python main.py $1
