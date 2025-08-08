#!/bin/bash
#SBATCH -n 48
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --output=./qsub_output/one2any_%j.out
#SBATCH --gres=gpumem:24g
#SBATCH --gpus=8
#SBATCH --tmp=300G

NPROC_PER_NODE=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29400

rsync -aq ./ $TMPDIR
cd $TMPDIR

tar -xvf  ov9d.tar.gz -C $TMPDIR/foundationpose_dataset/

cd $TMPDIR/foundationpose_dataset/ov9d/test/
mv bowl/* ./all/
mv bumbag/* ./all/
mv dumpling/* ./all/
mv facial_cream/* ./all/
mv handbag/* ./all/
mv litchi/* ./all/
mv mouse/* ./all/
mv pineapple/* ./all/
mv toy_truck/* ./all/
mv teddy_bear/* ./all/

cd $TMPDIR

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK   --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py --data_path ./foundationpose_dataset --data_train train --data_val test --exp_name one2any --epochs 20 \
    --log_dir ./logs/  --batch_size 12 --save_model --min_lr 1e-6 --max_lr 1e-4 --dataset combined_dataset