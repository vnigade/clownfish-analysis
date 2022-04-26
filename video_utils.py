
import csv
import re
from pathlib import Path
import json
import ml_utils as ml
import numpy as np
# from . import map


def process_video_string(line):
    words = re.split('/', line)
    video = words[-1]
    video = re.split('\.', video)
    return video[0].strip()


def get_scores(score_file, use_softmax=False, flow=False):

    if Path(score_file).exists() == False:
        raise FileNotFoundError('Video file does not exist ' + score_file)

    p_window = []
    with open(score_file) as json_data:
        # try:
        window_scores = json.load(json_data)
        # except:
        #    print("Score file cannot be parsed: ", score_file)
        #    return

        for key in window_scores:
            scores = window_scores[key]
            rgb_scores = np.array(scores["rgb_scores"])
            if use_softmax:
                rgb_scores = ml.softmax(rgb_scores)

            if flow:
                flow_scores = np.array(scores["flow_scores"])
                if use_softmax:
                    flow_scores = ml.softmax(flow_scores)
                p_scores = (rgb_scores + flow_scores) / 2
            else:
                p_scores = rgb_scores
            p_window.append(p_scores)

    return np.array(p_window)


# code source: https://github.com/gsig/charades-algorithms
def parse_charades_csv(filename):
    labels = {}
    durations = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
            durations[vid] = row['length']
    return labels, durations


def cls2int_charades(x):
    return int(x[1:])


def action_per_window_charades(actions, nclass, fps, start_frame, end_frame):
    target = np.zeros(nclass, dtype=int)
    for act in actions:
        action = cls2int_charades(act['class'])
        start = act['start']
        end = act['end']
        for frame in range(start_frame, end_frame, 1):
            frame_sec = frame/fps
            if frame_sec > start and frame_sec < end:
                target[action] = 1
    return target


# PKUMMD dataset helper functions
def parse_pkummd_json(label_file):
    labels = {}
    with open(label_file, 'r') as data_file:
        data = json.load(data_file)
        for key, value in data['database'].items():
            video = key.split("-clip")[0]
            annotation = value['annotations']
            # if value['subset'] == 'validation':
            if not video in labels:
                labels[video] = []
            clip = {
                'label': int(annotation['label']),
                'start_frame': annotation['start_frame'],
                'end_frame': annotation['end_frame']}
            labels[video].append(clip)
    return labels


def action_per_window_pkummd(actions, nclass, start_frame, end_frame):
    _FRAME_COUNT_PERCENT = 0.50
    target = np.zeros(nclass, dtype=int)
    possible_actions = []
    for act in actions:
        action = act['label']
        start = act['start_frame']
        end = act['end_frame']
        for frame in range(start_frame, end_frame, 1):
            if frame in list(range(start, end)):
                # target[action] = 1
                possible_actions.append(action)

    window_size = end_frame - start_frame
    valid_frame_count = int(_FRAME_COUNT_PERCENT * window_size)
    d = {i: possible_actions.count(i) for i in possible_actions}
    if d:
        action = sorted(d, reverse=True)[0]
        frames_with_action = d[action]
        if frames_with_action >= valid_frame_count:
            # for action in possible_actions: # Either this action or that
            # action number starts from zero. Threfore, not need to substract -1
            target[action] = 1
        # print("Window_size", window_size, "valid frames ", valid_frame_count,
        #     "possible actions:", possible_actions, "Dictionary:", d, "frames_with_action:", frames_with_action, "target:", np.argmax(target))
    return target


def action_per_frame_pkummd(actions):
    max_end_frame = -1
    for act in actions:
        if max_end_frame < act['end_frame']:
            max_end_frame = act['end_frame']
    assert max_end_frame != -1, "Max end frame index not found"
    n_frames = max_end_frame + 1
    targets = [-1 for i in range(n_frames)]
    for frame in range(n_frames):
        for act in actions:
            action = act['label']
            start = act['start_frame']
            end = act['end_frame']
            # [start, end). This is same for training as well.
            if frame in list(range(start, end+1)):
                targets[frame] = action + 1
    # We do not consider start_frame=0. This is consistent with training. Therefore, won't matter much.
    # del targets[0]
    return targets


def action_per_frame_thumos2014(actions):
    MAX_FRAMES = 100000
    targets = np.full(MAX_FRAMES, int(actions), dtype=int)
    return targets


class VideoLabels:
    def __init__(self, label_file, class_idx, dataset):
        self.label_file = label_file
        self.class_idx = class_idx
        self.dataset = dataset
        self.labels = self.get_labels()
        self.labels_key = list(self.labels.keys())

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.labels_key):
            raise StopIteration
        key = self.labels_key[self.index]
        value = self.labels[key]
        self.index += 1
        return key, value

    def next(self):
        return self.__next__()

    def get_classes(self):
        with open(self.class_idx, 'r') as fin:
            reader = csv.reader(fin, delimiter=' ')
            data_list = list(reader)

        classes = {}
        for item in data_list:
            classes[item[1]] = item[0]
        return classes

    def get_labels(self):

        if self.dataset == 'CHARADES':
            labels, durations = parse_charades_csv(self.label_file)
            self.durations = durations
            return labels
        elif self.dataset == 'PKUMMD':
            labels = parse_pkummd_json(self.label_file)
            return labels

        delim = ','
        if self.dataset == 'UCF101' or self.dataset == 'THUMOS2014':
            delim = ' '
        with open(self.label_file, 'r') as fin:
            reader = csv.reader(fin, delimiter=delim)
            data_list = list(reader)

        labels = {}
        classes = self.get_classes()
        for item in data_list:
            if self.dataset == 'UCF101' or self.dataset == 'THUMOS2014':
                class_label = re.split('/', item[0])
                class_label = class_label[0]
                video = process_video_string(item[0])
            elif self.dataset == 'HMDB51':
                # print(item)
                class_label = item[1]
                video = item[2]
            labels[video] = classes[class_label]
        return labels

    @staticmethod
    def get_action_vector(dataset, actions, nclass):
        target = np.zeros(nclass, dtype=int)
        if dataset == 'CHARADES':
            for x in actions:
                target[cls2int_charades(x['class'])] = 1
        else:
            target[int(actions) - 1] = 1

        return target

    def get_action_vector_window(self, video, nclass, total_frames, start_frame, end_frame):
        _FPS_CHARADES = 24
        actions = self.labels[video]
        if self.dataset == 'PKUMMD':
            target = action_per_window_pkummd(
                actions, nclass, start_frame, end_frame)
        elif self.dataset == 'CHARADES':
            # fps = total_frames / self.durations[video]
            fps = _FPS_CHARADES
            target = action_per_window_charades(
                actions, nclass, fps, start_frame, end_frame)
        else:
            return self.get_action_vector(self.dataset, actions, nclass)

        return target

    def get_action_per_frame(self, video):
        actions = self.labels[video]
        if self.dataset == 'PKUMMD':
            targets = action_per_frame_pkummd(actions)
        elif self.dataset == "THUMOS2014":
            targets = action_per_frame_thumos2014(actions)
        else:
            raise NotImplemented()
        return targets
