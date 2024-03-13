import pytorch_lightning as pl
import torch
import argparse
import numpy as np
from training import LitModel, SpotMatchingModel
from dataLoading import LitDataModule, SpotMatchingDataModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, ModelSummary, Timer, \
    RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model_twins_1d import TwinsSVT_1d  # 0
from model_twins_1d_group import TwinsSVT_1d_group  # 1
from model_twins_1d_SE import TwinsSVT_1d_SE  # 3
from model_twins_1d_group_SE import TwinsSVT_1d_group_SE  # 13
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from freegpu import find_gpus
import os
from pytorch_lightning.loggers import WandbLogger
from parseAction import ParseStr2List

parser = argparse.ArgumentParser()

# to fix 4090 NCCL P2P bug in driver
from gpuoption import gpuoption
if gpuoption():
    print('NCCL P2P is configured to disabled, new driver should fix this bug')

# train spot matching route
parser.add_argument('--SpotMatching', default=False, required=False, action='store_true', help='to switch to spotmatching training path')
parser.add_argument('--backbone_path', default='MixSaT/Focal2fps/epoch=11-step=12324.ckpt', help='to describe the backbone weighting')

# pytorch_lightning trainer args
parser.add_argument('--deterministic', default=True, required=False,
                    action='store_true', help='determinstic mode to ensure re-production, performance impeded')
parser.add_argument('--benchmark', dest='benchmark', type=bool,
                    default=True, help='use CUDnn benchmark to accelarate')
parser.add_argument('--min_epochs', dest='min_epochs', type=int,
                    default=1, help='minimum epochs to train')
parser.add_argument('--max_epochs', dest='max_epochs', type=int,
                    default=100, help='maximum epochs to train')
parser.add_argument('--check_val_every_n_epoch', dest='check_val_every_n_epoch', type=int,
                    default=1, help='every n epoch, go for validation once')
parser.add_argument('--devices',
                    default=-1, help='Devices to train')
parser.add_argument('--accelerator', dest='accelerator', type=str,
                    default='gpu', help='indicate accelarator to train')
parser.add_argument('--strategy', dest='strategy', type=str,
                    default='ddp', help='training method used for accelarator')
parser.add_argument('--precision', dest='precision', type=int,
                    default=32, help='precision for model (16//bf16//32//64)')
parser.add_argument('--detect_anomaly',
                    dest='detect_anomaly', default=True, type=bool, )
parser.add_argument('--sync_batchnorm', default=True, type=bool,
                    help='Enable synchronization between batchnorm layers across all GPUs')
parser.add_argument('--log_every_n_steps', type=int, default=10,
                    help='log every n step, It may slow down training to log on every single batch. ')

# Data loading args
parser.add_argument('--batch_size', dest='batch_size',
                    type=int, default=134, help='number of samples in one batch')
parser.add_argument('--SoccerNet_path', required=False, type=str,
                    default="/hdda/Datasets/SoccerNet",
                    help='directory for dataset')
parser.add_argument('--features', required=False, type=str,
                    default="baidu_ResNET_concat.npy", help='Video features [baidu_soccer_embeddings.npy, baidu_ResNET_concat.npy]')  # baidu_soccer_embeddings.npy
parser.add_argument('--split_train', nargs='+',
                    default=["train"], help='list of split for training')
parser.add_argument('--split_valid', nargs='+',
                    default=["valid"], help='list of split for validation')
parser.add_argument('--split_test', nargs='+',
                    default=["test", "challenge"], help='list of split for testing')
parser.add_argument('--max_num_worker', required=False,
                    type=int, default=(
                        os.cpu_count() // torch.cuda.device_count()),
                    help='number of worker to load data')

# optimizer args
parser.add_argument('--lr', required=False,
                    type=float, default=1e-04, help='Learning Rate')
parser.add_argument('--lrE', required=False,
                    type=float, default=1e-08, help='Learning Rate end')
parser.add_argument('--patience', required=False, type=int, default=5,
                    help='Patience ƒpabefore earlier stopping')
parser.add_argument('--seed', required=False, type=int,
                    default=42, help='seed for reproducibility')
parser.add_argument('--weight_decay', required=False,
                    type=float, default=0.0, help='weight decay')
parser.add_argument('--warmup', required=False, type=int,
                    default=(1000 // torch.cuda.device_count()), help='warmup epoch for scheduler')
parser.add_argument('--max_iters', required=False, type=int,
                    default=(1000 // torch.cuda.device_count()), help='number of iter//step for warmup')

# model args
parser.add_argument('--model_name', required=False, type=str,
                    default="TwinsSVT_1d_group_SE", help='named of the model to save')
parser.add_argument('--version', required=False, type=int,
                    default=2, help='Version of the dataset')
parser.add_argument('--feature_dim', required=False, type=int,
                    default=None, help='Number of input features')
parser.add_argument('--framerate', required=False, type=int,
                    default=2, help='Framerate of the input features')
parser.add_argument('--window_size', required=False, type=int,
                    default=3, help='Size of the chunk (in seconds)')
parser.add_argument('--window_shift', required=False, type=int,
                    default=0, help='Shift window RHS when slide by slide window data loading')
parser.add_argument('--window_stride', required=False, type=int,
                    default=3, help='1: Load data frame by frame')
parser.add_argument('--NMS_window', required=False,
                    type=int, default=6, help='NMS window in second')
parser.add_argument('--NMS_threshold', required=False, type=float,
                    default=0.0, help='NMS threshold for positive results')

# loss function args
parser.add_argument('--criterion', required=False, type=str,
                    default="BinaryFocalLoss", help='loss function: [SigmoidFocalLoss, BCELoss, BinaryFocalLoss]')
parser.add_argument('--weight', default=None, type=float,
                    help='loss function reduction on None class; None = not use')

# control arguments, affect performance and training process on development
parser.add_argument('--test_only', default=False, required=False,
                    action='store_true', help='Perform testing only')
parser.add_argument('--ckpt_path', required=False,
                    type=str, default=None, help='weights to load')
parser.add_argument('--limit_val_batches', dest='limit_val_batches', type=float,
                    default=1.0,
                    help='To limit the sample_size on validation step, can input 1.0 for whole set, [0,1] means percentage, int 1,2,..,n means num of data pair')
parser.add_argument('--limit_train_batches', dest='limit_val_batches',
                    type=float,
                    default=1.0,
                    help='To limit the sample_size on validation step, can input 1.0 for whole set, [0,1] means percentage, int 1,2,..,n means num of data pair')
parser.add_argument('--reload_dataloaders_every_n_epochs',
                    dest='reload_dataloaders_every_n_epochs', default=100, type=int)
parser.add_argument('--fast_dev_run', default=False, type=bool,
                    help='control trainer to run train/valid/test 1 epoch only, development purpose')
parser.add_argument('--profiler', default=None, type=str,
                    help='profiler to find bottleneck, None/simple/advanced, performance impeded')
parser.add_argument('--fast_dev', default=False, action='store_true',
                    help='This flag will set training dataset to head(5)')

# logger args
parser.add_argument('--logger', default='wandb', type=str,
                    help='logger to be selected')
parser.add_argument('--project', default='MixSaT',
                    type=str, help='project for wandb')
parser.add_argument('--entity', default='cihe-cis',
                    type=str, help='organization used in wandb')
parser.add_argument('--experiment_name', default='Valid mAP Test + BinFocal + SWA',
                    type=str, help='experiment name for wandb')

# model TwinsSVT args
parser.add_argument('--s1_next_dim', default=60, type=int)
parser.add_argument('--s1_patch_size', default=8, type=int)
parser.add_argument('--s1_local_patch_size', default=16, type=int)
parser.add_argument('--s1_global_k', default=20, type=int)
parser.add_argument('--s1_depth', default=1, type=int)
parser.add_argument('--s2_next_dim', default=720, type=int)
parser.add_argument('--s2_patch_size', default=4, type=int)
parser.add_argument('--s2_local_patch_size', default=4, type=int)
parser.add_argument('--s2_global_k', default=20, type=int)
parser.add_argument('--s2_depth', default=2, type=int)
parser.add_argument('--peg_kernel_size', default=9, type=int)
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--Post_norm', required=False, type=bool,
                    default=False, help='False to use PreNorm, True to use PostNorm')


def main(args):
    logger = WandbLogger(name=args.experiment_name,
                         project=args.project)

    if args.model_name == 'TwinsSVT_1d':
        model = TwinsSVT_1d(num_classes=18, frames_size=args.framerate * args.window_size,
                            s1_next_dim=args.s1_next_dim,
                            s1_patch_size=args.s1_patch_size,
                            s1_local_patch_size=args.s1_local_patch_size,
                            s1_global_k=args.s1_global_k,
                            s1_depth=args.s1_depth,
                            s2_next_dim=args.s2_next_dim,
                            s2_patch_size=args.s2_patch_size,
                            s2_local_patch_size=args.s2_local_patch_size,
                            s2_global_k=args.s2_global_k,
                            s2_depth=args.s2_depth,
                            peg_kernel_size=args.peg_kernel_size,
                            dropout=args.dropout,
                            Post_norm=args.Post_norm
                            )
    elif args.model_name == 'TwinsSVT_1d_SE':  # 3
        model = TwinsSVT_1d_SE(num_classes=18, frames_size=args.framerate * args.window_size,
                               s1_next_dim=args.s1_next_dim,
                               s1_patch_size=args.s1_patch_size,
                               s1_local_patch_size=args.s1_local_patch_size,
                               s1_global_k=args.s1_global_k,
                               s1_depth=args.s1_depth,
                               s2_next_dim=args.s2_next_dim,
                               s2_patch_size=args.s2_patch_size,
                               s2_local_patch_size=args.s2_local_patch_size,
                               s2_global_k=args.s2_global_k,
                               s2_depth=args.s2_depth,
                               peg_kernel_size=args.peg_kernel_size,
                               dropout=args.dropout,
                               Post_norm=args.Post_norm)
    elif args.model_name == 'TwinsSVT_1d_group':  # 1
        model = TwinsSVT_1d_group(num_classes=18, frames_size=args.framerate * args.window_size,
                                  s1_next_dim=args.s1_next_dim,
                                  s1_patch_size=args.s1_patch_size,
                                  s1_local_patch_size=args.s1_local_patch_size,
                                  s1_global_k=args.s1_global_k,
                                  s1_depth=args.s1_depth,
                                  s2_next_dim=args.s2_next_dim,
                                  s2_patch_size=args.s2_patch_size,
                                  s2_local_patch_size=args.s2_local_patch_size,
                                  s2_global_k=args.s2_global_k,
                                  s2_depth=args.s2_depth,
                                  peg_kernel_size=args.peg_kernel_size,
                                  dropout=args.dropout,
                                  Post_norm=args.Post_norm
                                  )
    elif args.model_name == 'TwinsSVT_1d_group_SE':  # 13
        model = TwinsSVT_1d_group_SE(num_classes=18, frames_size=args.framerate * args.window_size,
                                     s1_next_dim=args.s1_next_dim,
                                     s1_patch_size=args.s1_patch_size,
                                     s1_local_patch_size=args.s1_local_patch_size,
                                     s1_global_k=args.s1_global_k,
                                     s1_depth=args.s1_depth,
                                     s2_next_dim=args.s2_next_dim,
                                     s2_patch_size=args.s2_patch_size,
                                     s2_local_patch_size=args.s2_local_patch_size,
                                     s2_global_k=args.s2_global_k,
                                     s2_depth=args.s2_depth,
                                     peg_kernel_size=args.peg_kernel_size,
                                     dropout=args.dropout,
                                     Post_norm=args.Post_norm)
    
    if args.SpotMatching:
        litModel = LitModel.load_from_checkpoint(model = model, checkpoint_path=args.backbone_path)
        model = litModel.model # weighting loaded
        model.eval() # declare not train
        litModel = SpotMatchingModel(model, args)
    else:
        litModel = LitModel(model, args)

    if args.SpotMatching:
        dataModule = SpotMatchingDataModule(args)
    else:
        dataModule = LitDataModule(args)

    callbacks = [
        # our model checkpoint callback
        ModelCheckpoint(
            monitor='Valid/mAP', mode='max'),
        ModelSummary(max_depth=-1),
        LearningRateMonitor(),
        RichProgressBar(),
        EarlyStopping(monitor='Valid/mAP', mode='max', patience=args.patience),
        StochasticWeightAveraging(swa_lrs=args.lrE)
    ]

    if not args.test_only:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger if args.logger == 'wandb' else True,
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            limit_train_batches=0,
            limit_val_batches=0,
        )

    if not args.test_only:
        # may consider trainer.tune before training
        if args.ckpt_path is None:
            trainer.fit(model=litModel, datamodule=dataModule)
        else:
            # resume checkpoint and training
            trainer.fit(model=litModel, datamodule=dataModule,
                        ckpt_path=args.ckpt_path)

    # test_branch
    if not args.test_only:
        # if not defined ckpt_path
        # auto select best checkpoint in the current experiment
        trainer.test(datamodule=dataModule)  # for test set
        trainer.predict(datamodule=dataModule)  # for challege set
    else:
        trainer.test(model=litModel, datamodule=dataModule,
                     ckpt_path=args.ckpt_path)
        trainer.predict(model=litModel, datamodule=dataModule,
                        ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(
            args.SoccerNet_path) or  torch.cuda.device_count() > 2:
        args.SoccerNet_path = '/code/Datasets/SoccerNet'

    if (args.ckpt_path is not None) and (not os.path.exists(args.ckpt_path)):
        import glob
        path = glob.glob(f"SoccerViTAC/{args.ckpt_path}/checkpoints/*.ckpt")
        args.ckpt_path = path[0]
    # seed setting
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)
    main(args)
