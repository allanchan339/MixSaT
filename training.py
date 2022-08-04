import pytorch_lightning as pl
from torch import nn
from loss import NLLLoss, SigmoidFocalLoss, BinaryFocalLoss
import numpy as np
import torch
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.ActionSpotting import evaluate
from operator import itemgetter
import os
import json
from scheduler import CosineWarmupScheduler
from einops import rearrange
from einops.layers.torch import Rearrange


class LitModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        # save hyper-parameters to loggers
        self.save_hyperparameters(ignore=['model'])

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

    def validation_epoch_end(self, validation_step_outputs):
        def _valid_epoch_end(validation_step_outputs):

            # AP refers to average_precision_score in torchmetrics
            # [2048,18] in batch_size = 1024, [3979, 18] in batch_size = 2048
            all_labels = list(
                map(itemgetter('label'), validation_step_outputs))
            all_labels = torch.cat(all_labels).cpu().detach().numpy()
            all_outputs = list(
                map(itemgetter('output'), validation_step_outputs))
            all_outputs = torch.cat(all_outputs).cpu().detach().numpy()

            AP = []
            for i in range(1, 17+1):
                AP.append(np.nan_to_num(average_precision_score(all_labels
                                                                [:, i], all_outputs[:,  i])))
            mAP = np.mean(AP)  # sanity check 0.03242630150236116
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
            self.log('Valid/AP', AP_dictionary, logger=True, prog_bar=False,
                     )

        def _valid_epoch_end_ddp(validation_step_outputs):

            enable_Flag = True
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

            all_labels = torch.cat(all_labels_tmp).cpu().detach().numpy()
            all_outputs = torch.cat(
                all_outputs_tmp).cpu().detach().numpy()

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
            self.log('Valid/AP', AP_dictionary, logger=True, prog_bar=False,
                     )

        if self.args.strategy in ['ddp', 'ddp_sharded']:

            validation_step_outputs = self.all_gather(validation_step_outputs)

            _valid_epoch_end_ddp(validation_step_outputs)

        else:
            _valid_epoch_end(validation_step_outputs)

    def validation_step(self, batch, batch_idx):
        # use valid_metric
        feat, label = batch  # DP cuda 0 [512,18] #DDP cuda 0 [1024,18]
        output = self.model(feat)

        return {"label": label, "output": output}

    def _test_predict_share_step(self, game_ID, feat_half1, feat_half2, label_half1, label_half2, split):
        # must be here as we need to load the _split
        if self.args.logger == 'wandb':
            self.args.output_results_path = os.path.join(
                f'lightning_logs/version_None/results', f'output_{self.args._split}')
        else:
            self.args.output_results_path = os.path.join(
                f'lightning_logs/version_{self.trainer.logger.version}/results', f'output_{self.args._split}')


        def result_to_json(timestamp_long_half_1, timestamp_long_half_2, output_results_path):

            def get_spot_from_NMS(Input, window=60, thresh=0.0):
                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index, 0))
                    nms_to = int(np.minimum(
                        max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1
                return np.transpose([indexes, MaxValues])

            framerate = self.args.framerate
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()

            for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(17):
                    spots = get_spot(
                        timestamp[:, l], window=self.args.NMS_window*framerate, thresh=self.args.NMS_threshold)
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        # confidence = predictions_half_1[frame_index, l]

                        seconds = int((frame_index//framerate) % 60)
                        minutes = int((frame_index//framerate)//60)

                        prediction_data = dict()
                        prediction_data["gameTime"] = str(
                            half+1) + " - " + str(minutes) + ":" + str(seconds)

                        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]

                        prediction_data["position"] = str(
                            int((frame_index/framerate)*1000))
                        prediction_data["half"] = str(half+1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)

            # lightning_logs/version_25/results/output_test/2021-10-5 18:00 A vs B
            os.makedirs(os.path.join(
                output_results_path, game_ID), exist_ok=True)
            with open(os.path.join(output_results_path, game_ID, 'results_spotting.json'), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)
            return 0

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
            output = self.model(feat).cpu().detach().numpy()
            timestamp_long_half_1.append(output)
        timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

        timestamp_long_half_2 = []
        for b in range(int(np.ceil(len(feat_half2)/BS))):
            start_frame = BS*b
            end_frame = BS*(b+1) if BS * \
                (b+1) < len(feat_half2) else len(feat_half2)
            feat = feat_half2[start_frame:end_frame]
            output = self.model(feat).cpu().detach().numpy()
            timestamp_long_half_2.append(output)
        timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)

        # cut the null class
        timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
        # cut the null class
        timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

        # lightning_logs/version_25/results/output_test/
        flag = result_to_json(timestamp_long_half_1,
                       timestamp_long_half_2, self.args.output_results_path)
        return flag

    def test_step(self, batch, batch_idx):
        game_ID, feat_half1, feat_half2, label_half1, label_half2, split = batch
        self.args._split = split[0]  # test

        flag = self._test_predict_share_step(
            game_ID, feat_half1, feat_half2, label_half1, label_half2, split)
        
        # try to let all process wait 
        if self.args.strategy in ['ddp', 'ddp_sharded']:
            self.trainer.strategy.barrier()
        return flag

    def test_epoch_end(self, test_step_outputs):
        def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            import zipfile
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])

        if self.trainer.is_global_zero or self.args.strategy == 'dp':
            zipResults(zip_path=os.path.join(self.args.output_results_path, f'result_spotting_{self.args._split}.zip'),
                       target_dir=self.args.output_results_path,
                       filename="results_spotting.json")

            for metric in ['loose', 'tight']:
                results = evaluate(SoccerNet_path=self.args.SoccerNet_path,
                                   Predictions_path=self.args.output_results_path,
                                   split="test",
                                   prediction_file="results_spotting.json",
                                   version=self.args.version, metric=metric)

                self.log(f'Test/{metric}/average_mAP',
                         results['a_mAP'], prog_bar=True, logger=True, rank_zero_only=True if self.args.strategy != 'dp' else False)
                self.log(f'Test/{metric}/a_mAP_visible',
                         results['a_mAP_visible'], prog_bar=True, logger=True, rank_zero_only=True if self.args.strategy != 'dp' else False)
                self.log(f'Test/{metric}/a_mAP_unshown',
                         results['a_mAP_unshown'], prog_bar=True, logger=True, rank_zero_only=True if self.args.strategy != 'dp' else False)

                label_cls = (list(EVENT_DICTIONARY_V2.keys()))
                a_mAP_per_class = dict(
                    zip(label_cls, results['a_mAP_per_class']))
                self.log(f'Test/{metric}/mAP_per_class',
                         a_mAP_per_class, logger=True, prog_bar=False, rank_zero_only=True if self.args.strategy != 'dp' else False)

                a_mAP_per_class_visible = dict(
                    zip(label_cls, results['a_mAP_per_class_visible']))
                self.log(f'Test/{metric}/mAP_per_class_visible)', a_mAP_per_class_visible,
                         logger=True, prog_bar=False, rank_zero_only=True if self.args.strategy != 'dp' else False)

                a_mAP_per_class_unshown = dict(
                    zip(label_cls, results['a_mAP_per_class_unshown']))
                self.log(f'Test/{metric}/a_mAP_per_class_unshown)', a_mAP_per_class_unshown,
                         logger=True, prog_bar=False, rank_zero_only=True if self.args.strategy != 'dp' else False)

    def predict_step(self, batch, batch_idx):
        game_ID, feat_half1, feat_half2, label_half1, label_half2, split = batch
        self.args._split = split[0]  # test

        # expect to create and write data to output_challege folder
        flag = self._test_predict_share_step(
            game_ID, feat_half1, feat_half2, label_half1, label_half2, split)

        #TODO: use barrier but no waiting too long ???? why wait? 
        # if self.args.strategy in ['ddp', 'ddp_sharded']:
        #     self.trainer.strategy.barrier()

        return flag

    def on_predict_epoch_end(self, predict_step_outputs):
        def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            import zipfile
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])
        if self.trainer.is_global_zero or self.args.strategy == 'dp':
            # should zip as result_spotting_challege.zip
            zipResults(zip_path=os.path.join(self.args.output_results_path, f'result_spotting_{self.args._split}.zip'),
                       target_dir=self.args.output_results_path,
                       filename="results_spotting.json")

    def on_predict_end(self):

        if self.args.logger == 'wandb':
            if self.trainer.is_global_zero or self.args.strategy == 'dp':
                # rename version_None to {version}
                os.rename(f'lightning_logs/version_None/results',
                        f'lightning_logs/{self.trainer.logger.version}')
                import shutil
                # move data recursively to checkpoint folder
                shutil.move(f'lightning_logs/{self.trainer.logger.version}',
                        f'SoccerViTAC/{self.trainer.logger.version}')


class SpotMatchingModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        backbone = nn.Sequential(*list(model.children())).requires_grad_(False)
        backbone[0][4] = nn.Identity()
        backbone[0][5] = nn.Identity()
        backbone.eval()
        self.model = nn.Sequential(backbone, nn.Linear(720, 120*args.framerate), nn.Linear(
            120*args.framerate, 18*args.window_size*args.framerate), Rearrange('b (t cls) -> b t cls', t=args.window_size*args.framerate, cls=18), nn.Sigmoid())
        
        self.args = args
        # save hyper-parameters to loggers
        self.save_hyperparameters(ignore=['model'])

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
        feat, label = batch #[6, 10624], [6, 18]
        output = self.model(feat)

        loss = self.criterion(output, label)
        self.log("Train/Loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        def _valid_epoch_end(validation_step_outputs):

            # AP refers to average_precision_score in torchmetrics
            # [2048,18] in batch_size = 1024, [3979, 18] in batch_size = 2048
            all_labels = list(
                map(itemgetter('label'), validation_step_outputs))
            all_labels = torch.cat(all_labels).cpu().detach().numpy()
            all_outputs = list(
                map(itemgetter('output'), validation_step_outputs))
            all_outputs = torch.cat(all_outputs).cpu().detach().numpy()

            AP = []
            for i in range(1, 17+1):
                AP.append(np.nan_to_num(average_precision_score(all_labels
                                                                [:, i], all_outputs[:,  i])))
            mAP = np.mean(AP)  # sanity check 0.03242630150236116
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
            self.log('Valid/AP', AP_dictionary, logger=True, prog_bar=False,
                     )

        def _valid_epoch_end_ddp(validation_step_outputs):

            enable_Flag = True
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

            all_labels = torch.cat(all_labels_tmp).cpu().detach().numpy()
            all_outputs = torch.cat(
                all_outputs_tmp).cpu().detach().numpy()

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
            self.log('Valid/AP', AP_dictionary, logger=True, prog_bar=False,
                     )

        if self.args.strategy in ['ddp', 'ddp_sharded']:

            validation_step_outputs = self.all_gather(validation_step_outputs)

            _valid_epoch_end_ddp(validation_step_outputs)

        else:
            _valid_epoch_end(validation_step_outputs)

    def validation_step(self, batch, batch_idx):
        # use valid_metric
        feat, label = batch  # DP cuda 0 [512,18] #DDP cuda 0 [1024,18]
        output = self.model(feat)

        return {"label": label, "output": output}

    def _test_predict_share_step(self, game_ID, feat_half1, feat_half2, label_half1, label_half2, split):
        # must be here as we need to load the _split
        if self.args.logger == 'wandb':
            self.args.output_results_path = os.path.join(
                f'lightning_logs/version_None/results', f'output_{self.args._split}')
        else:
            self.args.output_results_path = os.path.join(
                f'lightning_logs/version_{self.trainer.logger.version}/results', f'output_{self.args._split}')

        def result_to_json(timestamp_long_half_1, timestamp_long_half_2, output_results_path):

            def get_spot_from_NMS(Input, window=60, thresh=0.0):
                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index, 0))
                    nms_to = int(np.minimum(
                        max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1
                return np.transpose([indexes, MaxValues])

            framerate = self.args.framerate
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()

            for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                for l in range(17):
                    spots = get_spot(
                        timestamp[:, l], window=self.args.NMS_window*framerate, thresh=self.args.NMS_threshold)
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        # confidence = predictions_half_1[frame_index, l]

                        seconds = int((frame_index//framerate) % 60)
                        minutes = int((frame_index//framerate)//60)

                        prediction_data = dict()
                        prediction_data["gameTime"] = str(
                            half+1) + " - " + str(minutes) + ":" + str(seconds)

                        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]

                        prediction_data["position"] = str(
                            int((frame_index/framerate)*1000))
                        prediction_data["half"] = str(half+1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)

            # lightning_logs/version_25/results/output_test/2021-10-5 18:00 A vs B
            os.makedirs(os.path.join(
                output_results_path, game_ID), exist_ok=True)
            with open(os.path.join(output_results_path, game_ID, 'results_spotting.json'), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)
            return 0

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
            output = self.model(feat).cpu().detach().numpy()
            timestamp_long_half_1.append(output)
        timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

        timestamp_long_half_2 = []
        for b in range(int(np.ceil(len(feat_half2)/BS))):
            start_frame = BS*b
            end_frame = BS*(b+1) if BS * \
                (b+1) < len(feat_half2) else len(feat_half2)
            feat = feat_half2[start_frame:end_frame]
            output = self.model(feat).cpu().detach().numpy()
            timestamp_long_half_2.append(output)
        timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)

        # cut the null class
        timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
        # cut the null class
        timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

        # lightning_logs/version_25/results/output_test/
        flag = result_to_json(timestamp_long_half_1,
                              timestamp_long_half_2, self.args.output_results_path)
        return flag

    def test_step(self, batch, batch_idx):
        game_ID, feat_half1, feat_half2, label_half1, label_half2, split = batch
        self.args._split = split[0]  # test

        flag = self._test_predict_share_step(
            game_ID, feat_half1, feat_half2, label_half1, label_half2, split)

        # try to let all process wait
        if self.args.strategy in ['ddp', 'ddp_sharded']:
            self.trainer.strategy.barrier()
        return flag

    def test_epoch_end(self, test_step_outputs):
        def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            import zipfile
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])

        if self.trainer.is_global_zero or self.args.strategy == 'dp':
            zipResults(zip_path=os.path.join(self.args.output_results_path, f'result_spotting_{self.args._split}.zip'),
                       target_dir=self.args.output_results_path,
                       filename="results_spotting.json")

            for metric in ['loose', 'tight']:
                results = evaluate(SoccerNet_path=self.args.SoccerNet_path,
                                   Predictions_path=self.args.output_results_path,
                                   split="test",
                                   prediction_file="results_spotting.json",
                                   version=self.args.version, metric=metric)

                self.log(f'Test/{metric}/average_mAP',
                         results['a_mAP'], prog_bar=True, logger=True, rank_zero_only=True if self.args.strategy != 'dp' else False)
                self.log(f'Test/{metric}/a_mAP_visible',
                         results['a_mAP_visible'], prog_bar=True, logger=True, rank_zero_only=True if self.args.strategy != 'dp' else False)
                self.log(f'Test/{metric}/a_mAP_unshown',
                         results['a_mAP_unshown'], prog_bar=True, logger=True, rank_zero_only=True if self.args.strategy != 'dp' else False)

                label_cls = (list(EVENT_DICTIONARY_V2.keys()))
                a_mAP_per_class = dict(
                    zip(label_cls, results['a_mAP_per_class']))
                self.log(f'Test/{metric}/mAP_per_class',
                         a_mAP_per_class, logger=True, prog_bar=False, rank_zero_only=True if self.args.strategy != 'dp' else False)

                a_mAP_per_class_visible = dict(
                    zip(label_cls, results['a_mAP_per_class_visible']))
                self.log(f'Test/{metric}/mAP_per_class_visible)', a_mAP_per_class_visible,
                         logger=True, prog_bar=False, rank_zero_only=True if self.args.strategy != 'dp' else False)

                a_mAP_per_class_unshown = dict(
                    zip(label_cls, results['a_mAP_per_class_unshown']))
                self.log(f'Test/{metric}/a_mAP_per_class_unshown)', a_mAP_per_class_unshown,
                         logger=True, prog_bar=False, rank_zero_only=True if self.args.strategy != 'dp' else False)

    def predict_step(self, batch, batch_idx):
        game_ID, feat_half1, feat_half2, label_half1, label_half2, split = batch
        self.args._split = split[0]  # test

        # expect to create and write data to output_challege folder
        flag = self._test_predict_share_step(
            game_ID, feat_half1, feat_half2, label_half1, label_half2, split)

        #TODO: use barrier but no waiting too long ???? why wait?
        # if self.args.strategy in ['ddp', 'ddp_sharded']:
        #     self.trainer.strategy.barrier()

        return flag

    def on_predict_epoch_end(self, predict_step_outputs):
        def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            import zipfile
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])
        if self.trainer.is_global_zero or self.args.strategy == 'dp':
            # should zip as result_spotting_challege.zip
            zipResults(zip_path=os.path.join(self.args.output_results_path, f'result_spotting_{self.args._split}.zip'),
                       target_dir=self.args.output_results_path,
                       filename="results_spotting.json")

    def on_predict_end(self):

        if self.args.logger == 'wandb':
            if self.trainer.is_global_zero or self.args.strategy == 'dp':
                # rename version_None to {version}
                os.rename(f'lightning_logs/version_None/results',
                          f'lightning_logs/{self.trainer.logger.version}')
                import shutil
                # move data recursively to checkpoint folder
                shutil.move(f'lightning_logs/{self.trainer.logger.version}',
                            f'SoccerViTAC/{self.trainer.logger.version}')

