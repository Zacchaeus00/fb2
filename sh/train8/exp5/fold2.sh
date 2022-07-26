#!/bin/bash
##train8-exp0 + fix
#SBATCH --job-name=train8-exp5-fold2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
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
python -u train8.py --ckpt /gpfsnyu/scratch/yw3642/hf-models/allenai_longformer-base-4096 \
--epochs 10 --batch_size 4 --lr 1e-5 --lr_head 1e-3 --seed -1 --fold 2 --exp 5 \
--use_pretrained /gpfsnyu/scratch/yw3642/fb2/ckpt/pretrain0/exp16/pretrained_model.pt \
--adv_lr 1 --adv_eps 0.0005 --adv_after_epoch 1 --patience 30 --gradient_checkpointing --warmup_ratio 0.1
echo "FINISH"