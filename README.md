# MixSaT

## How to create environment required 

Run the following command on terminal
```
conda create -n MixSaT  -c conda-forge pytorch-gpu=1.13 pytorch_lightning=1.7 python=3.10 einops
conda activate MixSaT
pip install wandb
```

The wandb account must be configured in advance to proper output log to wandb experiment logger. More details can be refered to [Wandb](https://docs.wandb.ai/quickstart)
## How to train
```
python main.py 
```

## How to interference with pretrained model

Download pretrained weight on [drive](), and run following command on terminal
```

```
