import math

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
import os
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


def feats2clip(feats, stride, clip_length, padding="replicate_last", off=0, off_shift=0):
    if padding == "zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0] / stride) * stride
        print("pad need to be", clip_length - pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length - pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    # To control idx can control feature clip
    # [0,30,60,.] #shape = [180]
    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    # [0,clip_length=29]
    for i in torch.arange(-off + off_shift, clip_length - off + off_shift):
        idxs.append(idx + i)
        # 00: [0,30,60,...] #shape = [180]
        # 01: [1,31,61,...]
        # 02: [2,32,62,...]
        # ...
        # 29 [29,59,89,...]
    idx = torch.stack(idxs, dim=1)  # shape = [180,30]
    # [0,1,2,3....],[30,31,32,...]

    if padding == "replicate_last":
        # make sure idx range [0, frame_num]
        idx = idx.clamp(0, feats.shape[0] - 1)
    # print(idx)
    return feats[idx, ...]  # arrange data based on idx, shape = [180,30,2048]


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"],
                 version=2, framerate=2, window_size=3):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.version = version
        self.split = split
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels = "Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels = "Labels-v2.json"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(
            self.path, self.listGames[index], "1_" + self.features))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
        feat_half2 = np.load(os.path.join(
            self.path, self.listGames[index], "2_" + self.features))
        feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

        # Load labels
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        # check if annoation exists
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(
                open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * (seconds + 60 * minutes)

                if self.version == 1:
                    if "card" in event:
                        label = 0
                    elif "subs" in event:
                        label = 1
                    elif "soccer" in event:
                        label = 2
                    else:
                        continue
                elif self.version == 2:
                    if event not in self.dict_event:
                        continue
                    label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0] - 1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0] - 1)
                    label_half2[frame][label] = value

        feat_half1 = feats2clip(torch.from_numpy(feat_half1),
                                stride=1, off=int(self.window_size_frame / 2),
                                clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2),
                                stride=1, off=int(self.window_size_frame / 2),
                                clip_length=self.window_size_frame)

        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2, self.split

    def __len__(self):
        return len(self.listGames)


class SoccerNetClipsNoCache_SlidingWindow(Dataset):
    def __init__(self, path, features="baidu_ResNET_concat.npy", split=["train"],
                 version=2, stride=1,
                 framerate=2, window_size=3, fast_dev=False):
        self.path = path
        self.listGames = getListGames(split)[:5] if fast_dev else getListGames(split)
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.split = split
        self.version = version
        self.stride = stride
        if version == 1:
            self.num_classes = 3
            self.labels = "Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels = "Labels-v2.json"

        self.save_clip = []
        self.all_labels = []
        self.all_feats = list()
        self.save_label_position = []
        for game in tqdm(self.listGames):
            # Load features
            len_half1 = np.load(os.path.join(
                self.path, game, "1_" + "ResNET_TF2.npy")).shape[0]
            len_half1 = len(np.arange(0,  len_half1 - 1, self.stride))
            len_half2 = np.load(os.path.join(
                self.path, game, "2_" + "ResNET_TF2.npy")).shape[0]
            len_half2 = len(np.arange(0, len_half2 - 1, self.stride))
            # self.game_length.append([len_half1, len_half2])

            label_half1 = np.zeros((len_half1, self.num_classes + 1))
            label_half1[:, 0] = 1  # those are BG classes
            label_half2 = np.zeros((len_half2, self.num_classes + 1))
            label_half2[:, 0] = 1  # those are BG classes

            if os.path.exists(os.path.join(self.path, game, self.labels)):
                labels = json.load(
                    open(os.path.join(self.path, game, self.labels)))
                for annotation in labels["annotations"]:
                    time = annotation["gameTime"]
                    event = annotation["label"]
                    half = int(time[0])
                    minutes = int(time[-5:-3])
                    seconds = int(time[-2::])
                    frame = self.framerate * (seconds + 60 * minutes)

                    if self.version == 1:
                        if "card" in event:
                            label = 0
                        elif "subs" in event:
                            label = 1
                        elif "soccer" in event:
                            label = 2
                        else:
                            continue
                    elif self.version == 2:
                        if event not in self.dict_event:
                            continue
                        label = self.dict_event[event]

                    if half == 1 and frame // self.window_size_frame >= label_half1.shape[0]:
                        continue  # skip loop if condition meets
                    if half == 2 and frame // self.window_size_frame >= label_half2.shape[0]:
                        continue

                    # Ignore non-visibility label
                    # if "visibility" in annotation.keys():
                    #     if annotation["visibility"] == "not shown":
                    #         continue

                    if half == 1:  # if on label.json
                        frame = min(frame, len_half1 - 1)
                        label_half1[frame // self.stride][0] = 0
                        label_half1[frame // self.stride][label + 1] = 1

                    if half == 2:
                        frame = min(frame, len_half2 - 1)
                        label_half2[frame // self.stride][0] = 0
                        label_half2[frame // self.stride][label + 1] = 1

                for i in range(label_half1.shape[0]):
                    self.all_labels.append((label_half1[i]))  # label_half1 = np.delete(label_half1, i)
                    self.save_label_position.append([game, '1_', i])

                for i in range(label_half2.shape[0]):
                    self.all_labels.append((label_half2[i]))
                    self.save_label_position.append([game, '2_', i])

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels for the segmentation.
                clip_targets (np.array): clip of targets for the spotting.
            """
        game = self.save_label_position[index][0]
        half = self.save_label_position[index][1]
        position = self.save_label_position[index][2]

        # Load features
        feat = np.load(os.path.join(
            self.path, game, half + self.features), mmap_mode='r')
        feat = feat.reshape(-1, feat.shape[-1])
        idxs = np.arange(position * self.window_size_frame, (position + 1) * self.window_size_frame)
        idxs = np.clip(idxs, position * self.window_size_frame, feat.shape[0] - 1)
        feat = feat[idxs, ...]

        return feat, self.all_labels[index].astype(np.float32)

    def __len__(self):
        return len(self.all_labels)
