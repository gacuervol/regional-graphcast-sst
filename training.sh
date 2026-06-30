#!/bin/bash
pyenv_path=~/miniconda3/envs/art1_pyenv/bin/
script_path=~/repo/src/art1_tools/
params_path=~/repo/params/GraphCast
logs_path=~/repo/test/test_8_6_ds1982_2012_log.txt

# Test summary:
# This script runs 8 experiments with lineal function
# Latent size, n messages, mask filter, log file
# 8, 6, None, 56

# TEST 10: 8, 6, None, 56
echo "TEST 10: 8, 6, None, 56" | tee -a $logs_path
n_epochs_train=150
n_epochs_fine=100
M_i=3
msg_steps=6
latent_size=8
batch_size=8
mask_filter=land
n_test=$(($(ls $params_path | grep -oP '(?<=params_)\d+' | sort -nr | head -n1) + 1))
#n_test=$(ls $params_path | grep -oP '(?<=params_)\d+' | sort -nr | head -n1) # sobreescribe el ultimo

params_train_name=params_${n_test}_cycle-${n_epochs_train}_M_i-${M_i}_msg_steps-${msg_steps}_latent_size-${latent_size}_batch_size-${batch_size}.npz

echo PHASE 1: TRAINING 
date >> $logs_path
SECONDS=0
#${pyenv_path}python ${script_path}training.py $n_epochs_train $M_i $msg_steps $latent_size $batch_size $mask_filter --fast_test $n_test 2>> $logs_path
${pyenv_path}python ${script_path}training_pmap.py $n_epochs_train $M_i $msg_steps $latent_size $batch_size $mask_filter $n_test 2>> $logs_path
if [ $? = 0 ]; then
    message="Training test ${n_test} successfully done"
else
    message="Error in training test ${n_test}"
fi
duration=$SECONDS
echo "$((duration / 60)) m $((duration % 60)) s" | tee -a $logs_path
date >> $logs_path
echo $message | tee -a $logs_path
