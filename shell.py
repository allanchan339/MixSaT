import subprocess
import numpy as np
import os
experiment_name = "TwinsSVT_6019_sweep_test_BS"
model_name = "TwinsSVT_1d_group_LoGlo_SE"
ckpt_path = "SoccerViTAC/2o4izlmk/checkpoints/epoch=14-step=10493.ckpt"

for batch in range(268,268+1,67):
    command = f'conda run -n SoccerNet python main.py --batch_size {batch} --model_name {model_name} --experiment_name {experiment_name}_B{batch} --test_only --ckpt_path={ckpt_path} --device 1'
    child = subprocess.Popen(command, shell=True)

    child.wait()    
    
"""
for batch in range(50,250+1, 10):
    command = f'conda run -n SoccerNet python main.py --batch_size {batch} --model_name {model_name} --experiment_name {experiment_name}_B{batch} --test_only --ckpt_path={ckpt_path}'
    child = subprocess.Popen(command, shell=True)

    child.wait()    
"""
# for dim in next_dim:
#     for local_size in local_patch_size:
#         for patch in patch_size:
#             for window in window_size:
#                 print(
#                     f'python main.py --batch_size {batch_size} --model_name {model_name} --experiment_name {experiment_name} --window_size {window} --patch_size "{patch}" --local_patch_size "{local_size}" --next_dim "{dim}"')
#                 # command = f'python main.py --batch_size {batch_size} --model_name {model_name} --experiment_name TwinsSVT_group_sweep --window_size {window} --patch_size "{patch}" --local_patch_size "{local_size}" --next_dim "{dim}"'

#                 # child = subprocess.Popen(list(map(str,['conda','run','-n', 'SoccerNet', 'python', 'main.py', '--batch_size', batch_size, '--model_name', model_name, '--experiment_name', experiment_name, '--window_size', window, '--patch_size', patch, '--local_patch_size', local_size, '--next_dim', dim])))
#                 # child.wait()
