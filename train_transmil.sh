#!/bin/sh


for SEED in 0 1 2
do
python train_transmil.py \
    --output_dir "output/mag40_seed${SEED}" \
    --epochs 30 \
    --seed ${SEED}
done