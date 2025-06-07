import pytorch_lightning as pl
from torch import nn
from .loss import NLLLoss, SigmoidFocalLoss, BinaryFocalLoss
import numpy as np
import torch
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.ActionSpotting import evaluate
from operator import itemgetter
import os
from einops import rearrange
from einops.layers.torch import Rearrange

from ..utils.utils import format_results_to_json
from .scheduler import CosineWarmupScheduler  # Added missing import
from .callbacks import OutputManagementCallback  # Added missing import


class LitModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        # save hyper-parameters to loggers
        self.save_hyperparameters(ignore=['model'])
        
        # Storage for validation and test outputs (for PyTorch Lightning v2.0+ compatibility)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if self.args.criterion == 'NLLLoss':
            self.criterion = NLLLoss()
        elif self.args.criterion == 'BCELoss':
            if self.args.weight is not None:
                weight = torch.ones(18,)
                weight[0] = self.args.weight
                self.criterion = nn.BCELoss(weight=weight)
            else:
                self.criterion = nn.BCELoss()

        elif self.args.criterion == 'SigmoidFocalLoss':
            self.criterion = SigmoidFocalLoss()

        elif self.args.criterion == 'BinaryFocalLoss':
            self.criterion = BinaryFocalLoss()
    
    def forward(self, x):  # used to write pipeline from input to output
        y = self.model(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=self.args.weight_decay, amsgrad=False)
        # eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.args.warmup, max_iters=self.args.max_iters
        )
        return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Valid/mAP",
                "frequency": 1*self.args.check_val_every_n_epoch
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def optimizer_step(self, *args, **kwargs):
        # we update lr per iter
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteratio

    def training_step(self, batch, batch_idx):
        feat, label = batch
        output = self.model(feat)

        loss = self.criterion(output, label)
        self.log("Train/Loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        # Clear any leftover validation outputs from previous epoch
        self.validation_step_outputs.clear()

    def on_test_epoch_start(self):
        # Clear any leftover test outputs from previous epoch
        self.test_step_outputs.clear()

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        
        def _valid_epoch_end(validation_step_outputs):

            # AP refers to average_precision_score in torchmetrics
            # [2048,18] in batch_size = 1024, [3979, 18] in batch_size = 2048
            all_labels = list(
                map(itemgetter('label'), validation_step_outputs))
            all_labels = torch.cat(all_labels).cpu().detach().to(torch.float32).numpy()
            all_outputs = list(
                map(itemgetter('output'), validation_step_outputs))
            all_outputs = torch.cat(all_outputs).cpu().detach().to(torch.float32).numpy()

            AP = []
            for i in range(1, 17+1):
                AP.append(np.nan_to_num(average_precision_score(all_labels
                                                                [:, i], all_outputs[:,  i])))
            mAP = np.mean(AP)  # sanity check 0.03242630150236116
            tmp = EVENT_DICTIONARY_V2.copy()
            for k, v in tmp.items():
                #dict["kick-off"] = AP[0]
                tmp[k] = AP[v]

            self.log('Valid/mAP', mAP, logger=True, prog_bar=True,
                     )

            label_cls = (list(EVENT_DICTIONARY_V2.keys()))
            zip_iterator = zip(label_cls, AP)
            AP_dictionary = dict(zip_iterator)
            
            # Log each AP value individually since PyTorch Lightning cannot log dictionaries
            for key, value in AP_dictionary.items():
                self.log(f'Valid/AP/{key}', value, logger=True, prog_bar=False)

        def _valid_epoch_end_ddp(validation_step_outputs):

            # AP refers to average_precision_score in torchmetrics
            all_labels = list(
                map(itemgetter('label'), validation_step_outputs))

            all_outputs = list(
                map(itemgetter('output'), validation_step_outputs))
            all_labels_tmp = list()
            all_outputs_tmp = list()
            for i in range(len(all_labels)):
                all_labels_tmp.append(
                    rearrange(all_labels[i], "g b c -> (g b) c"))
                all_outputs_tmp.append(
                    rearrange(all_outputs[i], "g b c -> (g b) c"))

            all_labels = torch.cat(all_labels_tmp).cpu().detach().to(torch.float32).numpy()
            all_outputs = torch.cat(
                all_outputs_tmp).cpu().detach().to(torch.float32).numpy()

            AP = []
            for i in range(1, 17+1):
                AP.append(np.nan_to_num(average_precision_score(all_labels
                                                                [:, i], all_outputs[:,  i])))
            mAP = np.mean(AP)
            tmp = EVENT_DICTIONARY_V2
            tmp = tmp.copy()
            for k, v in tmp.items():
                #dict["kick-off"] = AP[0]
                tmp[k] = AP[v]

            self.log('Valid/mAP', mAP, logger=True, prog_bar=True,
                     )

            label_cls = (list(EVENT_DICTIONARY_V2.keys()))
            zip_iterator = zip(label_cls, AP)
            AP_dictionary = dict(zip_iterator)
            
            # Log each AP value individually since PyTorch Lightning cannot log dictionaries
            for key, value in AP_dictionary.items():
                self.log(f'Valid/AP/{key}', value, logger=True, prog_bar=False)

        if self.args.strategy in ['ddp', 'ddp_sharded', 'ddp_find_unused_parameters_true']:
            validation_step_outputs = self.all_gather(validation_step_outputs)
            _valid_epoch_end_ddp(validation_step_outputs)
        else:
            _valid_epoch_end(validation_step_outputs)
            
        # Clear the outputs for the next epoch
        self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # use valid_metric
        feat, label = batch  # DP cuda 0 [512,18] #DDP cuda 0 [1024,18]
        output = self.model(feat)
        
        # Store outputs for on_validation_epoch_end (PyTorch Lightning v2.0+ compatibility)
        step_output = {"label": label, "output": output}
        self.validation_step_outputs.append(step_output)
        # return step_output

    def _get_output_management_callback(self, trainer: pl.Trainer) -> 'OutputManagementCallback':
        """Helper to retrieve the OutputManagementCallback from the trainer."""
        for callback in trainer.callbacks:
            if isinstance(callback, OutputManagementCallback): # Make sure to import OutputManagementCallback if not done globally
                return callback
        # self.print("Warning: OutputManagementCallback not found in trainer.callbacks.")
        return None

    def _test_predict_share_step(self, game_ID, feat_half1, feat_half2, label_half1, label_half2, split, trainer: pl.Trainer):
        # must be here as we need to load the _split
        
        output_mgmt_callback = self._get_output_management_callback(trainer)
        if not output_mgmt_callback:
            self.print("Error: OutputManagementCallback not found. Cannot determine output path.")
            # Potentially raise an error or handle this case gracefully
            return -1 # Indicate failure

        # The specific path for this game's JSON, e.g., lightning_logs/VERSION/results/output_test
        # This path is where the individual game JSONs (inside their game_ID folder) will be saved.
        current_output_path_for_split = output_mgmt_callback.get_output_path_for_split(self, trainer)
        
        # The old self.args.output_results_path is now managed by the callback.
        # We pass current_output_path_for_split to format_results_to_json,
        # which will then create game_ID subdirectories within it.

        # Batch size of 1
        game_ID = game_ID[0]
        feat_half1 = feat_half1.squeeze(0)
        label_half1 = label_half1.float().squeeze(0)
        feat_half2 = feat_half2.squeeze(0)
        label_half2 = label_half2.float().squeeze(0)

        BS = self.args.batch_size
        timestamp_long_half_1 = []
        for b in range(int(np.ceil(len(feat_half1)/BS))):  # range(0,180)
            start_frame = BS*b  # 0,256,512, ...
            end_frame = BS*(b+1) if BS * \
                (b+1) < len(feat_half1) else len(feat_half1)  # 256,512, 768, ...
            feat = feat_half1[start_frame:end_frame]
            output = self.model(feat).cpu().detach().to(torch.float32).numpy()
            timestamp_long_half_1.append(output)
        timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

        timestamp_long_half_2 = []
        for b in range(int(np.ceil(len(feat_half2)/BS))):
            start_frame = BS*b
            end_frame = BS*(b+1) if BS * \
                (b+1) < len(feat_half2) else len(feat_half2)
            feat = feat_half2[start_frame:end_frame]
            output = self.model(feat).cpu().detach().to(torch.float32).numpy()
            timestamp_long_half_2.append(output)
        timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)

        # cut the null class
        timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
        # cut the null class
        timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

        # Call the utility function to format and save JSON
        # Pass all necessary arguments, including the path from the callback
        flag = format_results_to_json(
            game_ID=game_ID,
            timestamp_long_half_1=timestamp_long_half_1,
            timestamp_long_half_2=timestamp_long_half_2,
            output_results_path=current_output_path_for_split, # This is the key change
            framerate=self.args.framerate,
            NMS_window=self.args.NMS_window,
            NMS_threshold=self.args.NMS_threshold,
            INVERSE_EVENT_DICTIONARY_V2=INVERSE_EVENT_DICTIONARY_V2
        )
        return flag

    def test_step(self, batch, batch_idx):
        game_ID, feat_half1, feat_half2, label_half1, label_half2, split = batch
        self.args._split = split[0]  # test

        flag = self._test_predict_share_step(
            game_ID, feat_half1, feat_half2, label_half1, label_half2, split, self.trainer) # Pass trainer
        
        # Store outputs for on_test_epoch_end (PyTorch Lightning v2.0+ compatibility)
        step_output = {"flag": flag, "split": split[0]}
        self.test_step_outputs.append(step_output)
        
        return flag

    def on_test_epoch_end(self):
        # Use the stored test outputs (PyTorch Lightning v2.0+ compatibility)
        test_step_outputs = self.test_step_outputs
        
        # Zipping is now handled by OutputManagementCallback.on_test_epoch_end
        
        if not self.trainer.is_global_zero:
            # Clear the outputs even if we exit early
            self.test_step_outputs.clear()
            return
            
        output_mgmt_callback = self._get_output_management_callback(self.trainer)
        if not output_mgmt_callback:
            self.print("Error: OutputManagementCallback not found in test_epoch_end. Cannot proceed with evaluation.")
            # Clear the outputs even if we exit early
            self.test_step_outputs.clear()
            return

        # The path for evaluation should be the directory where JSONs (and the zip) are stored for the current split.
        # e.g., lightning_logs/VERSION/results/output_test
        predictions_path_for_evaluation = output_mgmt_callback.get_output_path_for_split(self, self.trainer)

        # Evaluation logic remains here, but uses the path from the callback
        for metric in ['loose', 'tight']:
            results = evaluate(SoccerNet_path=self.args.SoccerNet_path,
                               Predictions_path=predictions_path_for_evaluation, # Use callback provided path
                               split=self.args._split, # Ensure _split is correctly set, e.g. 'test'
                               prediction_file="results_spotting.json",
                               version=self.args.version, metric=metric)

            self.log(f'Test/{metric}/average_mAP',
                     results['a_mAP'], prog_bar=True, logger=True, rank_zero_only=True)
            self.log(f'Test/{metric}/a_mAP_visible',
                     results['a_mAP_visible'], prog_bar=True, logger=True, rank_zero_only=True)
            self.log(f'Test/{metric}/a_mAP_unshown',
                     results['a_mAP_unshown'], prog_bar=True, logger=True, rank_zero_only=True)

            label_cls = (list(EVENT_DICTIONARY_V2.keys()))
            a_mAP_per_class = dict(
                zip(label_cls, results['a_mAP_per_class']))
            self.log(f'Test/{metric}/mAP_per_class',
                     a_mAP_per_class, logger=True, prog_bar=False, rank_zero_only=True)

            a_mAP_per_class_visible = dict(
                zip(label_cls, results['a_mAP_per_class_visible']))
            self.log(f'Test/{metric}/mAP_per_class_visible)', a_mAP_per_class_visible,
                     logger=True, prog_bar=False, rank_zero_only=True)

            a_mAP_per_class_unshown = dict(
                zip(label_cls, results['a_mAP_per_class_unshown']))
            self.log(f'Test/{metric}/a_mAP_per_class_unshown)', a_mAP_per_class_unshown,
                     logger=True, prog_bar=False, rank_zero_only=True)
        
        # Clear the outputs for the next epoch
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        game_ID, feat_half1, feat_half2, label_half1, label_half2, split = batch
        self.args._split = split[0]

        # expect to create and write data to output_challege folder
        flag = self._test_predict_share_step(
            game_ID, feat_half1, feat_half2, label_half1, label_half2, split, self.trainer) # Pass trainer

        return flag

    def on_predict_epoch_end(self, predict_step_outputs):
        # Zipping is now handled by OutputManagementCallback.on_predict_epoch_end
        # The original zipResults call is removed from here.
        # If there was any other logic here, it would remain.
        pass # Placeholder if no other logic was present
