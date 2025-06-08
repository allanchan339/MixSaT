import torch
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
import os
import multiprocessing # Added for parallel processing
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


# Worker function for multiprocessing - defined at the module level
def _process_game_static(worker_args):
    game_name, base_path, feature_filename_template, game_stride, \
    labels_filename_template, game_framerate, num_event_classes, \
    game_version, game_dict_event = worker_args

    # Local version of _get_label_for_event logic
    def get_label_for_event_local(event_text_local):
        label_local = None
        if game_version == 1:
            if "card" in event_text_local: label_local = 0
            elif "subs" in event_text_local: label_local = 1
            elif "soccer" in event_text_local: label_local = 2
        elif game_version == 2:
            if event_text_local in game_dict_event:
                label_local = game_dict_event[event_text_local]
        return label_local

    game_specific_labels = []
    game_specific_positions = []

    label_array_dim = num_event_classes + 1  # +1 for background class

    try:
        path_1_feat = os.path.join(base_path, game_name, "1_" + feature_filename_template)
        path_2_feat = os.path.join(base_path, game_name, "2_" + feature_filename_template)

        if not (os.path.exists(path_1_feat) and os.path.exists(path_2_feat)):
            # print(f"Warning: Feature file(s) missing for game {game_name}. Skipping.")
            return [], []

        len_half1_raw = np.load(path_1_feat, mmap_mode='r').shape[0]
        len_half1 = len(np.arange(0,  len_half1_raw - 1, game_stride))

        len_half2_raw = np.load(path_2_feat, mmap_mode='r').shape[0]
        len_half2 = len(np.arange(0, len_half2_raw - 1, game_stride))

    except FileNotFoundError:
        # print(f"Warning: Feature file not found during shape load for game {game_name}. Skipping.")
        return [], []
    except Exception as e:
        # print(f"Warning: Error loading feature shapes for game {game_name}: {e}. Skipping.")
        return [], []

    label_h1_data = np.zeros((len_half1, label_array_dim))
    label_h1_data[:, 0] = 1  # Initialize with background class
    label_h2_data = np.zeros((len_half2, label_array_dim))
    label_h2_data[:, 0] = 1  # Initialize with background class

    path_labels_file = os.path.join(base_path, game_name, labels_filename_template)
    if os.path.exists(path_labels_file):
        try:
            with open(path_labels_file, 'r') as f:
                labels_json_data = json.load(f)
        except Exception as e:
            # print(f"Warning: Error loading/parsing labels JSON for game {game_name}: {e}. Proceeding without annotations for this game.")
            labels_json_data = {"annotations": []}

        for annotation in labels_json_data.get("annotations", []):
            time = annotation.get("gameTime")
            event = annotation.get("label")

            if not time or not event: # Basic check for malformed annotation
                continue

            try:
                half = int(time[0])
                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
            except (ValueError, IndexError):
                # print(f"Warning: Malformed gameTime '{time}' in game {game_name}. Skipping annotation.")
                continue
                
            frame = game_framerate * (seconds + 60 * minutes)
            label_idx = get_label_for_event_local(event)

            if label_idx is None:
                continue

            target_label_col = label_idx + 1 # Map event label to column index (0 is BG)

            current_frame_idx = frame // game_stride
            if half == 1:
                if current_frame_idx < label_h1_data.shape[0]:
                    label_h1_data[current_frame_idx, 0] = 0  # Not background
                    label_h1_data[current_frame_idx, target_label_col] = 1
            elif half == 2:
                if current_frame_idx < label_h2_data.shape[0]:
                    label_h2_data[current_frame_idx, 0] = 0  # Not background
                    label_h2_data[current_frame_idx, target_label_col] = 1
    
    for i in range(label_h1_data.shape[0]):
        game_specific_labels.append(label_h1_data[i])
        game_specific_positions.append([game_name, '1_', i])

    for i in range(label_h2_data.shape[0]):
        game_specific_labels.append(label_h2_data[i])
        game_specific_positions.append([game_name, '2_', i])

    return game_specific_labels, game_specific_positions


class SoccerNetDatasetBase(Dataset):
    def __init__(self, version=2):
        super().__init__()
        self.version = version
        if version == 1:
            # For version 1, the label derivation is custom (card, subs, soccer)
            # and num_classes is 3. EVENT_DICTIONARY_V1 is also size 3.
            self.dict_event = EVENT_DICTIONARY_V1 # Retain for consistency or other uses
            self.labels_filename = "Labels.json"
            self.num_classes = 3 # Explicitly 3 as per original logic's output for event types
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.labels_filename = "Labels-v2.json"
            self.num_classes = len(self.dict_event) # This is 17
        else:
            raise ValueError(f"Unsupported version: {version}. Must be 1 or 2.")

    def _get_label_for_event(self, event_text):
        label = None
        if self.version == 1:
            if "card" in event_text:
                label = 0
            elif "subs" in event_text:
                label = 1
            elif "soccer" in event_text: # "soccer ball" like events
                label = 2
        elif self.version == 2:
            if event_text in self.dict_event:
                label = self.dict_event[event_text]
        return label # Returns None if event should be skipped (e.g. not in dict or not matched)


class SoccerNetClipsTesting(SoccerNetDatasetBase):
    def __init__(self, path, features="ResNET_PCA512.npy", split=["test"],
                 version=2, framerate=2, window_size=3):
        super().__init__(version=version)
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.split = split

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
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels_filename)):
            labels_data = json.load(
                open(os.path.join(self.path, self.listGames[index], self.labels_filename)))

            for annotation in labels_data["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * (seconds + 60 * minutes)

                label = self._get_label_for_event(event)

                if label is None:
                    continue
                
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

class SoccerNetClipsNoCache_SlidingWindow(SoccerNetDatasetBase):
    def __init__(self, path, features="baidu_ResNET_concat.npy", split=["train"],
                 version=2, stride=3,
                 framerate=2, window_size=3, fast_dev=False):
        super().__init__(version=version)
        self.path = os.path.expanduser(path) if '~' in path else path
        
        self.listGames = getListGames(split)[:20] if fast_dev else getListGames(split)
        self.features = features # This is for __getitem__ to load the actual feature data
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.split = split # Keep self.split for potential use elsewhere or logging
        self.stride = stride
        
        # Determine the feature_name for loading shapes in __init__
        # self.feature_name is used for loading feature shapes, self.features for loading actual data in __getitem__
        if features == "baidu_ResNET_concat.npy":
            self.feature_name_for_shape_loading = "ResNET_TF2.npy"
        else:
            self.feature_name_for_shape_loading = features
        
        self.all_labels = []
        self.save_label_position = []
        # self.all_feats = list() # Not modified in the original highlighted section
        # self.save_clip = [] # Not modified in the original highlighted section

        worker_args_list = []
        for game_name_iter in self.listGames:
            worker_args_list.append((
                game_name_iter, self.path, self.feature_name_for_shape_loading, self.stride,
                self.labels_filename, self.framerate, self.num_classes, # Pass actual num_classes
                self.version, self.dict_event
            ))

        desc_split_part = self.split[0] if self.split and len(self.split) > 0 else "data"
        
        if worker_args_list: # Only run multiprocessing if there are games
            # Determine number of processes, fallback to 1 if os.cpu_count() is unavailable or 0
            num_processes = os.cpu_count()
            if not num_processes or num_processes < 1:
                num_processes = 1
            # Cap processes to avoid overwhelming system, e.g., max(1, min(num_processes, 8))
            # For this example, we'll use num_processes directly or a simple cap.
            # num_processes = min(num_processes, 8) # Optional: cap number of processes

            with multiprocessing.Pool(processes=num_processes) as pool:
                all_results = list(tqdm(pool.imap(_process_game_static, worker_args_list),
                                        total=len(worker_args_list),
                                        desc=f"{desc_split_part} Loading features and labels"))
            
            for single_game_labels, single_game_positions in all_results:
                self.all_labels.extend(single_game_labels)
                self.save_label_position.extend(single_game_positions)

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
        idxs = np.arange(position * self.stride, position * self.stride + self.window_size_frame)
        idxs = np.clip(idxs, position * self.stride, feat.shape[0] - 1)
        feat = feat[idxs, ...]

        return feat, self.all_labels[index].astype(np.float32)

    def __len__(self):
        return len(self.all_labels)

