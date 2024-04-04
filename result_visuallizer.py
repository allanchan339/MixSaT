from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ast import arg
from datetime import timedelta
import re
from SoccerNet.Downloader import getListGames

from simplejson import load
from torch.utils.data import Dataset
import json
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip


class ModelResult2Video():
    # Each video must provide VLAD result, VTN result, Label
    # Else error must come
    def readLabels(self):
        def readGroundTruth(label):
            label = label['annotations']
            label = [
                i for i in label if f'{self.half} -' in i["gameTime"]]
            label = [{k: int(v) if v.isnumeric() else v for k,
                      v in i.items()} for i in label]  # O(n^2)
            label = pd.DataFrame(label)
            # force to int(second)
            label['second'] = label['position']//1000
            label['second'] = pd.to_timedelta(label['second'], unit='s')
            label = label.set_index(label['second'])
            return label

        def thresholdsFilter(label):
            label = label["predictions"]
            # filter by half
            label = [i for i in label if f'{self.half} -' in i["gameTime"]]
            label = [{k: int(v) if v.isnumeric() else v for k,
                      v in i.items()} for i in label]  # change str to value
            label = pd.DataFrame(label)
            label['confidence'] = label['confidence'].astype(float)
            # force to int(second)
            label['second'] = label['position']//1000
            label['second'] = pd.to_timedelta(label['second'], unit='s')
            label = label.set_index(label['second'])
            label = label[label['confidence'] >= self.thresholds]
            return label

        self.labels = json.load(
            open(os.path.join(self.path, self.listGames[self.index], self.labels)))  # Soccerpath + listgame + label_v2.json
        self.vlad = json.load(
            open(os.path.join(self.model_path, "NetVLAD++", "outputs_test", self.listGames[self.index], self.model_base_name)))
        self.vtn = json.load(
            open(os.path.join(self.model_path, "VTN", "outputs_test", self.listGames[self.index], self.model_base_name)))

        self.labels = readGroundTruth(self.labels)
        self.vlad = thresholdsFilter(self.vlad)
        self.vtn = thresholdsFilter(self.vtn)

        self.table = pd.merge(self.labels['label'], self.vlad['label'], left_index=True,
                              right_index=True, how='outer', suffixes=["_gt", "_vlad"])
        self.table = pd.merge(self.table, self.vtn['label'], left_index=True, right_index=True, how='outer').rename(
            columns={'label': 'label_vtn'})

    def readVideo(self):
        def getDuration(vidcap, fps_video):
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps_video
            return duration
        if self.HQ:
            video_path = os.path.join(self.path, getListGames(self.split)[
                self.index], f'{self.half}_720p.mkv')  # 720p, 25fps
        else:
            video_path = os.path.join(self.path, getListGames(self.split)[
                self.index], f'{self.half}.mkv')  # 224p, 25fps
        # read video
        self.vidcap = cv2.VideoCapture(video_path)
        # self.vidcap = VideoFileClip(video_path)
        # read FPS
        self.fps_video = self.vidcap.get(cv2.CAP_PROP_FPS)
        # self.fps_video = int(self.vidcap.fps)
        # read duration
        self.time_second = getDuration(self.vidcap, self.fps_video)
        self.video_FourCC = int(self.vidcap.get(cv2.CAP_PROP_FOURCC))

    def parseJSON(self):
        def cleaningTable(df):
            tmp = pd.DataFrame(
                columns=['second', 'label_gt', 'label_vlad', 'label_vtn'])
            for index, row in df.iterrows():
                if row.all() == set():
                    # skip if 3 rows are empty
                    continue
                tmp2 = [index]
                for i in [row['label_gt'], row['label_vlad'], row['label_vtn']]:
                    if np.nan in i:  # del nan if exist
                        i.remove(np.nan)

                    if i == set():  # if set is empty
                        i = None
                    else:
                        i = ','.join(i)

                    tmp2.append(i)

                tmp1 = pd.DataFrame(
                    columns=['second', 'label_gt', 'label_vlad', 'label_vtn'])
                tmp1.loc[0] = tmp2
                tmp = pd.concat(
                    [tmp, tmp1], ignore_index=True)
            tmp = tmp.set_index('second')
            tmp = tmp.dropna(how='all')
            tmp = tmp.fillna('Null')
            return tmp

        def resample():
            final_df = pd.DataFrame()
            final_df['label_gt'] = self.table.groupby(pd.Grouper(
                freq=f'{int(self.time_windows)}s',))['label_gt'].apply(set)
            final_df['label_vlad'] = self.table.groupby(pd.Grouper(freq=f'{int(self.time_windows)}s',))[
                'label_vlad'].apply(set)
            final_df['label_vtn'] = self.table.groupby(pd.Grouper(freq=f'{int(self.time_windows)}s',))[
                'label_vtn'].apply(set)

            return final_df
        self.final_df = resample()
        self.final_df = cleaningTable(self.final_df)
        # print('Final result is as follows')
        # print(self.final_df)

    def createVideoWriter(self):
        output_path = os.path.join(self.output_path, getListGames(self.split)[
            self.index])  # src/demo/3-0 XX vs YY

        from pathlib import Path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.width = int(self.vidcap.get(
            cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        self.height = int(self.vidcap.get(
            cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        if self.HQ:
            self.video_frame = cv2.VideoWriter(
                os.path.join(output_path, f'{self.half}_processed_HQ_{self.thresholds}_{self.time_windows}.mp4'), cv2.VideoWriter_fourcc(*'avc1'), self.fps_video, (self.width, self.height))
        else:
            self.video_frame = cv2.VideoWriter(
                os.path.join(output_path, f'{self.half}_processed_{self.thresholds}_{self.time_windows}.mp4'), self.video_FourCC, self.fps_video, (self.width, self.height))

    def runMoviePY(self):
        self.readLabels()
        self.readVideo()
        self.parseJSON()

        def generator(txt):
            return TextClip(txt, font='Arial', fontsize=24, color='white')

        subtitles = SubtitlesClip(self.subs, generator)
        result = CompositeVideoClip(
            [self.vidcap, subtitles.set_pos(('center', 'bottom'))])

        output_path = os.path.join(self.output_path, getListGames(self.split)[
            self.index])  # src/demo/3-0 XX vs YY

        from pathlib import Path
        Path(output_path).mkdir(parents=True, exist_ok=True)

        result.write_videofile(os.path.join(
            output_path, f'{self.half}_processed.mp4'), fps=self.fps_video, audio=True)
        # failed as Magick cannot be installed in this machine

    def prepare(self):
        self.readLabels()
        # self.readVideo()
        # self.parseJSON()

    def run(self):
        self.createVideoWriter()  # only useful when cv2 writer is using

        def addLabel(frame, sub):
            fontScale = 1.5 if self.HQ else 0.5
            shift = -80 if self.HQ else -30
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2 if self.HQ else 1

            compare = False  # Compare our model with NetVLAD++
            if compare:
                cv2.putText(frame,
                            'NetVLAD++: ' + sub['label_vlad'],
                            (10, self.height-10+shift),  # origin
                            font,  # font
                            fontScale,  # fontScale
                            (0, 255, 255),  # color
                            thickness,  # thickness
                            cv2.LINE_AA)
            
            else:
                # if GT_Label & VTN also show null
                if sub['label_gt'] == 'null' and sub['label_vtn'] == 'null':
                    return frame

            cv2.putText(frame,
                        'GT Label: ' + sub['label_gt'],
                        (10, self.height-10),  # origin
                        font,  # font
                        fontScale,  # fontScale
                        (0, 255, 255),  # color
                        thickness,  # thickness
                        cv2.LINE_AA)

            cv2.putText(frame,
                        'Ours: ' + sub['label_vtn'],
                        (10, self.height-10+2*shift),  # origin
                        font,  # font
                        fontScale,  # fontScale
                        (0, 255, 255),  # color
                        thickness,  # thickness
                        cv2.LINE_AA)
            # cv2.LINE_4 4-connected line
            # cv2.LINE_8
            # cv2.LINE_AA Anti-Alisaed
            return frame

        i_frame = 0
        ret = True
        pbar = tqdm(total=self.time_second)
        pbar.set_description('Processing Video Demo')

        present = False
        while ret:  # ret: bool, indicate have frame to read or not
            ret, frame = self.vidcap.read()
            second = i_frame//self.fps_video  # [100-124]//5 = 4s
            second_windows = int(divmod(second, self.time_windows)[
                                 0] * self.time_windows)
            second_windows = timedelta(seconds=second_windows)
            # if second in subs_CV.keys():
            #     frame = addLabel(frame, second, subs_CV)

            if second_windows in self.final_df.index:
                sub = self.final_df.loc[second_windows]
                frame = addLabel(frame, sub)

            self.video_frame.write(frame)

        #     # update counter and progress bar
            i_frame += 1
            if i_frame % self.fps_video == 0:
                pbar.update(1)

        self.video_frame.release()

    def analysis(self):
        # get only a lucky time windows where our model perform well
        tmp = self.table['label_gt']
        tmp = tmp.dropna()
        tmp = tmp[~tmp.index.duplicated(keep='first')]
        tmp = tmp.index.to_list()
        table = self.table.loc[tmp]
        del table['label_vlad']
        table['correct'] = np.where(table['label_gt'] == table['label_vtn'], True, False)
        table['correct_rolling'] = table['correct'].rolling(3).mean()
        if 1.0 in table['correct_rolling'].values.tolist():
            print(
                f"Analysing for {getListGames(self.split)[self.index]} - half {self.half}, index = {self.index}")

            output_path = os.path.join(self.output_path, getListGames(self.split)[
                self.index])
            from pathlib import Path
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            table.to_csv(os.path.join(
                    output_path, f'{self.half}_Analysis_{self.time_windows}.csv'))
            
            self.table.to_csv(os.path.join(
                output_path, f'{self.half}_Result_{self.time_windows}.csv'))

        

    def statistics(self):
        def confusionMatrix2DataFrame(label_cls, stat, model_name):
            stat_result = pd.DataFrame(columns=[
                'Model', 'Class', 'True Negative', 'False Positive', 'False Negative', 'True Positive', ])
            # we skip the null class
            for label, confu_mat in zip(label_cls[1:], stat[1:]):
                data = [model_name, label, confu_mat[0][0],
                        confu_mat[0][1], confu_mat[1][0], confu_mat[1][1]]
                tmp = pd.DataFrame([data], columns=[
                    'Model', 'Class', 'True Negative', 'False Positive', 'False Negative', 'True Positive'])
                stat_result = pd.concat(
                    [stat_result, tmp], ignore_index=True)

            for i in ['False Positive', 'False Negative', 'True Positive']:
                stat_result.loc['Total', i] = stat_result[i].sum()

            stat_result.loc['Total', 'Class'] = 'Total'
            stat_result.loc['Total', 'Model'] = model_name
            stat_result = stat_result.set_index('Class')
            return stat_result

        from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
        from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
        label_cls = ["Null"] + (list(EVENT_DICTIONARY_V2.keys()))
        stat_vlad = multilabel_confusion_matrix(
            self.final_df['label_gt'], self.final_df['label_vlad'], labels=label_cls)
        stat_vtn = multilabel_confusion_matrix(
            self.final_df['label_gt'], self.final_df['label_vtn'], labels=label_cls)
        stat_result_vlad = confusionMatrix2DataFrame(
            label_cls, stat_vlad, 'NetVLAD++')
        stat_result_vtn = confusionMatrix2DataFrame(label_cls, stat_vtn, 'VTN')

        output_path = os.path.join(self.output_path, getListGames(self.split)[
            self.index])

        from pathlib import Path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.final_df.to_csv(os.path.join(
            output_path, f'{self.half}_Subtitle_{self.time_windows}.csv'))
        stat_result_vlad.to_csv(os.path.join(
            output_path, f'{self.half}_Statistic_NetVLAD++_{self.time_windows}.csv'))
        stat_result_vtn.to_csv(os.path.join(
            output_path, f'{self.half}_Statistic_VTN_{self.time_windows}.csv'))

    def __init__(self, path, model_path, output_path, half=1, thresholds=0.3, HQ=True, time_windows=5, index=0):
        self.path = path
        self.model_path = model_path
        self.output_path = output_path
        self.split = 'test'
        self.listGames = getListGames(self.split)
        self.half = half
        self.labels = "Labels-v2.json"
        self.model_base_name = "results_spotting.json"
        self.index = index
        self.thresholds = thresholds
        self.HQ = HQ
        self.time_windows = time_windows


def main(args):
    # from SoccerNet_path get the label
    # from model_path get the model result
    # rerun the model for a certain thresholds
    # sort all labels and get ready to read video frame
    # according video frame with certain frame rate, embedd results by cv2
    # append all video frame, back to video
    # output to output path
    # to find a game in test split, in england
    for i in range(100):
        for half in range(1,3):
            embedder = ModelResult2Video(args.SoccerNet_path,
                                        args.Model_path, args.Output_path, half, thresholds=args.thresholds, HQ=args.HQ, time_windows=args.display_windows, index = i)
            embedder.prepare()
            # embedder.runMoviePY()
            embedder.analysis()

        # embedder.parseJSON()
        # embedder.statistics()
        # embedder.run()


if __name__ == '__main__':
    parser = ArgumentParser(description='VTN Result Visualization',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path',   required=False, type=str,
                        default="/home/user/Desktop/Code/SoccerNetv2-DevKit/Task1-ActionSpotting/data",     help='Path for SoccerNet')
    parser.add_argument('--Model_path', required=False, type=str,
                        default="/home/user/Desktop/Code/SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling/src/models", help='indicate the path that store model result')
    parser.add_argument('--Output_path', required=False, type=str,
                        default='/home/user/Desktop/Code/SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling/src/models/demo', help='indicate the path that output processed video')
    parser.add_argument('--thresholds', default=0.6, type=float,
                        help='threshold for prediction filter')
    parser.add_argument('--HQ', action='store_true',
                        default=True, help='use HQ video')
    parser.add_argument('--display_windows',       required=False, type=int,
                        default=3, help='Time windows for display in second, eg 5s')
    parser.add_argument('--half', default=1, type=int,
                        help='Indicate the first half  or second half competition to process')
    args = parser.parse_args()

    main(args)
