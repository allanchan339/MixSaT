import pytorch_lightning as pl
from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsNoCache, SoccerNetClips_v2, \
    SoccerNetClipsNoCache_SlidingWindow
import torch


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['model'])

    def prepare_data(self):
        '''called only once and on 1 GPU, before DDP'''
        # download data (train/val and test sets)
        pass

    def setup(self, stage=None):
        """called by each rank (DDP)"""
        if stage == 'fit' or None:
            self.dataset_Train = SoccerNetClipsNoCache_SlidingWindow(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_train,
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size, 
                fast_dev=self.args.fast_dev)
            self.dataset_Valid = SoccerNetClipsNoCache_SlidingWindow(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_valid,
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size, 
                fast_dev=self.args.fast_dev)

        elif stage == 'test':
            self.dataset_Test = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_test[0],  # here we only use 'Test'
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size)

        elif stage == 'predict':
            self.dataset_Predict = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_test[1],  # here we only use 'Challege'
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size)

    def train_dataloader(self):
        print(f'batch_size = %d' % self.args.batch_size)
        train_loader = torch.utils.data.DataLoader(self.dataset_Train,
                                                   batch_size=self.args.batch_size, shuffle=True,
                                                   num_workers=self.args.max_num_worker, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.dataset_Valid,
                                                 batch_size=self.args.batch_size, shuffle=False,
                                                 num_workers=self.args.max_num_worker, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.dataset_Test,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)
        return test_loader

    def predict_dataloader(self):
        predict_loader = torch.utils.data.DataLoader(self.dataset_Predict,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1, pin_memory=True)
        return predict_loader

    # def teardown(self, stage=None):
    #     if stage == 'fit':
    #         print('teardown on fit is called')
    #         del self.dataset_Train, self.dataset_Valid


class LitDataModule_test_replace_valid(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['model'])

    def prepare_data(self):
        '''called only once and on 1 GPU, before DDP'''
        # download data (train/val and test sets)
        pass

    def setup(self, stage=None):
        """called by each rank (DDP)"""
        if stage == 'fit' or None:
            self.dataset_Train = SoccerNetClipsNoCache(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_train,
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size,
                fast_dev=self.args.fast_dev)
            self.dataset_Valid = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_valid,
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size,
                fast_dev=self.args.fast_dev)

        elif stage == 'test':
            self.dataset_Test = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_test[0],  # here we only use 'Test'
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size)

        elif stage == 'predict':
            self.dataset_Predict = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_test[1],  # here we only use 'Challege'
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size)

    def train_dataloader(self):
        print(f'batch_size = %d' % self.args.batch_size)
        train_loader = torch.utils.data.DataLoader(self.dataset_Train,
                                                   batch_size=self.args.batch_size, shuffle=True,
                                                   num_workers=self.args.max_num_worker, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.dataset_Valid,
                                                 batch_size=1, shuffle=False,
                                                 num_workers=self.args.max_num_worker, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.dataset_Test,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)
        return test_loader

    def predict_dataloader(self):
        predict_loader = torch.utils.data.DataLoader(self.dataset_Predict,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1, pin_memory=True)
        return predict_loader

    # def teardown(self, stage=None):
    #     if stage == 'fit':
    #         print('teardown on fit is called')
    #         del self.dataset_Train, self.dataset_Valid


class LitDataModule_v2(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['model'])

    def prepare_data(self):
        '''called only once and on 1 GPU, before DDP'''
        # download data (train/val and test sets)
        pass

    def setup(self, stage=None):
        """called by each rank (DDP)"""
        if stage == 'fit' or None:
            self.dataset_Train = SoccerNetClips_v2(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_train,
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size, 
                fast_dev=self.args.fast_dev)
            self.dataset_Valid = SoccerNetClips_v2(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_valid,
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size, 
                fast_dev=self.args.fast_dev)

        elif stage == 'test':
            self.dataset_Test = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_test[0],  # here we only use 'Test'
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size)

        elif stage == 'predict':
            self.dataset_Predict = SoccerNetClipsTesting(
                path=self.args.SoccerNet_path,
                features=self.args.features,
                split=self.args.split_test[1],  # here we only use 'Challege'
                version=self.args.version,
                framerate=self.args.framerate,
                window_size=self.args.window_size)

    def train_dataloader(self):

        train_loader = torch.utils.data.DataLoader(self.dataset_Train,
                                                   batch_size=self.args.batch_size, shuffle=True,
                                                   num_workers=self.args.max_num_worker, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.dataset_Valid,
                                                 batch_size=self.args.batch_size, shuffle=False,
                                                 num_workers=self.args.max_num_worker, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.dataset_Test,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)
        return test_loader

    def predict_dataloader(self):
        predict_loader = torch.utils.data.DataLoader(self.dataset_Predict,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1, pin_memory=True)
        return predict_loader

    def teardown(self, stage=None):
        if stage == 'fit':
            print('teardown on fit is called')
            del self.dataset_Train, self.dataset_Valid
