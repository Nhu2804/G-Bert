#! /bin/bash

# Pre-training với procedure thay vì medication
python run_pretraining.py --model_name GBert-pretraining-proc --num_train_epochs 5 --do_train --graph

# Fine-tuning cho procedure prediction
python run_gbert.py --model_name GBert-predict-proc --use_pretrain --pretrain_dir ../saved/GBert-pretraining-proc --num_train_epochs 5 --do_train --graph

# Lặp lại quá trình alternating training
for i in {1..15}
do
    echo "Iteration $i: Pre-training with procedure data"
    python run_pretraining.py --model_name GBert-pretraining-proc --use_pretrain --pretrain_dir ../saved/GBert-predict-proc --num_train_epochs 5 --do_train --graph
    
    echo "Iteration $i: Fine-tuning for procedure prediction"
    python run_gbert.py --model_name GBert-predict-proc --use_pretrain --pretrain_dir ../saved/GBert-pretraining-proc --num_train_epochs 5 --do_train --graph
done