#!/bin/bash
#SBATCH --job-name=train4-exp1-fold1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=167:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yw3642@nyu.edu
#SBATCH --output=log/%x-%A.out
#SBATCH --error=log/%x-%A.err
#SBATCH --gres=gpu:1
#SBATCH -p aquila
#SBATCH --nodelist=agpu7

module purge
module load anaconda3 cuda/11.1.1

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/fb2/src

echo "START"
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate kaggle
python -u train3.py --ckpt /gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-v3-base \
--epochs 10 --batch_size 2 --lr 2e-5 --lr_head 1e-3 --weight_decay 0 --seed 42 \
--exp 1 --fold 1 --use_pretrained /gpfsnyu/scratch/yw3642/fb2/ckpt/pretrain0/exp4/pretrained_model.pt \
--use_stable_dropout
echo "FINISH"