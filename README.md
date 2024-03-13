# To Start Automatic Commentary of Soccer Game with Mixed Spatial and Temporal Attention

<img width="669" alt="image" src="https://user-images.githubusercontent.com/38370649/229306020-20fd11a0-dc46-4ae7-9ad7-5c453853e87c.png">

## How to create environment required 

Run the following command on terminal
```
conda create -n MixSaT  -c conda-forge pytorch-gpu=1.13 pytorch-lightning=1.7 torchmetrics==0.11.4 python=3.10 einops scikit-learn
conda activate MixSaT
pip install wandb SoccerNet
```

The wandb account must be configured in advance to proper output log to wandb experiment logger. More details can be refered to [Wandb](https://docs.wandb.ai/quickstart)
## How to train
```
python main.py 
```

## How to interference with pretrained checkpoint

Download pretrained weight on [drive](), and run following command on terminal
```
python main.py --test_only --ckpt_path [LOCATION OF CHECKPOINT]
```

## Citation
If you find our work useful for your research, please do not hesitate to cite our paper
```
@INPROCEEDINGS{9978078,
  author={Chan, Cheuk-Yiu and Hui, Chun-Chuen and Siu, Wan-Chi and Chan, Sin-wai and Chan, H. Anthony},
  booktitle={TENCON 2022 - 2022 IEEE Region 10 Conference (TENCON)}, 
  title={To Start Automatic Commentary of Soccer Game with Mixed Spatial and Temporal Attention}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/TENCON55691.2022.9978078}}
```
