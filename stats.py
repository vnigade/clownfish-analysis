import numpy as np
import ml_utils as ml
import siminet
import random
import itertools


class FusionStats:
    def __init__(self, opts):
        self.opts = opts

        # Basic stats
        self.videos_cnt = 0
        self.windows_cnt = []

        # Scores per window
        self.scores_per_window_local = []
        self.scores_per_window_remote = []
        self.scores_per_window_gts = []
        self.max_windows = 200

        # Corr per window
        self.corr_per_window_local = []
        self.corr_per_window_remote = []
        self.corr_per_window_gts = []

        # Corr at transition
        self.corr_at_transition_cosine = []
        self.corr_at_transition_euclidean = []
        self.corr_at_transition_bhatta = []
        self.corr_at_transition_siminet = []
        self.corr_at_transition_gts = []

        # Frames per action
        self.frames_per_action_lst = []

        # Filter windows sent
        self.filter_windows_sent_cnt = 0
        self.filter_windows_sent_transition_cnt = 0

    def basic_video_stats(self, video_cnt, windows_cnt):
        self.videos_cnt += video_cnt
        self.windows_cnt.append(windows_cnt)

    def _save_video_stats(self):
        windows_cnt = np.array(self.windows_cnt)
        print("VideoCount: ", self.videos_cnt,
              "WindowsCount (AVG/MEDIAN): {}/{}".format(windows_cnt.mean(), np.median(windows_cnt)))

    def scores_per_window(self, video, p_scores_local, p_scores_remote, gts):
        if not self.opts.scores_per_window:
            return
        gts_arr = np.asarray(gts)
        self.scores_per_window_local.append(
            p_scores_local[:self.max_windows, :])
        self.scores_per_window_remote.append(
            p_scores_remote[:self.max_windows, :])
        self.scores_per_window_gts.append(gts_arr[:self.max_windows, :])

    def _save_scores_per_window(self):
        score_local = np.array(list(itertools.zip_longest(
            *self.scores_per_window_local, fillvalue=0))).T
        score_remote = np.array(list(itertools.zip_longest(
            *self.scores_per_window_remote, fillvalue=0))).T
        score_gts = np.array(list(itertools.zip_longest(
            *self.scores_per_window_gts, fillvalue=0))).T
        print("Score shape:", score_local.shape,
              score_remote.shape, score_gts.shape)
        np.save(self.opts.stats_dir + "/" + 'local_' +
                self.opts.datasets + '_score', score_local)
        np.save(self.opts.stats_dir + "/" + 'remote_' +
                self.opts.datasets + '_score', score_remote)
        np.save(self.opts.stats_dir + "/" + 'gts_' +
                self.opts.datasets + '_score', score_gts)

    def corr_per_window(self, video, p_scores_local, p_scores_remote, gts, p_local_feature_window):
        if not self.opts.corr_per_window:
            return
        alphas_local, alphas_remote, alphas_gts = [], [], []
        for i in range(1, len(p_scores_local)):
            alpha_cosine_local = ml.cosine_similarity(
                p_scores_local[i-1], p_scores_local[i])
            alpha_cosine_remote = ml.cosine_similarity(
                p_scores_remote[i-1], p_scores_remote[i])
            gt = -1
            if np.max(gts[i]) > 0.0:
                gt = np.argmax(gts[i])
            alpha_siminet_local = siminet.siminet_similarity(prev_vec=p_local_feature_window[i-1],
                                                             cur_vec=p_local_feature_window[i])
            alphas_local.append(alpha_siminet_local)
            alphas_remote.append(alpha_cosine_remote)
            alphas_gts.append(gt)

        self.corr_per_window_local.append(alphas_local)
        self.corr_per_window_remote.append(alphas_remote)
        self.corr_per_window_gts.append(alphas_gts)

    def _save_corr_per_window(self):
        corr_local = np.array(list(itertools.zip_longest(
            *self.corr_per_window_local, fillvalue=0))).T
        corr_remote = np.array(list(itertools.zip_longest(
            *self.corr_per_window_remote, fillvalue=0))).T
        corr_gts = np.array(list(itertools.zip_longest(
            *self.corr_per_window_gts, fillvalue=0))).T
        print("Corr shape:", corr_local.shape,
              corr_remote.shape, corr_gts.shape)
        np.save(self.opts.stats_dir + "/" + 'local_' +
                self.opts.datasets + '_corr', corr_local)
        np.save(self.opts.stats_dir + "/" + 'remote_' +
                self.opts.datasets + '_corr', corr_remote)
        np.save(self.opts.stats_dir + "/" + 'gts_' +
                self.opts.datasets + '_corr', corr_gts)

    def corr_at_transition(self, idx, video, p_scores_local, p_local_feature_window, gts):
        if not self.opts.corr_at_transition:
            return

        if not ((idx - 10) >= 0 and (idx + 10) < len(p_scores_local)):
            return

        alphas_cosine = []
        alphas_euclidean = []
        alphas_bhatta = []
        alphas_siminet = []
        for i in range(idx-10, idx+10+1):
            alpha_cosine = ml.cosine_similarity(
                p_scores_local[i-1], p_scores_local[i])
            alpha_euclidean = ml.euclidean_similarity(
                p_scores_local[i-1], p_scores_local[i])
            alpha_bhatta = ml.bhatta_similarity(
                p_scores_local[i-1], p_scores_local[i])
            alpha_siminet = siminet.siminet_similarity(
                p_local_feature_window[i-1], p_local_feature_window[i])
            alphas_cosine.append(alpha_cosine)
            alphas_euclidean.append(alpha_euclidean)
            alphas_bhatta.append(alpha_bhatta)
            alphas_siminet.append(alpha_siminet)
        _gts_list = []
        for i in range(1, 11):
            gt = 1 if np.max(gts[-i]) == 1 else -1
            _gts_list.append(gt)

        self.corr_at_transition_cosine.append(alphas_cosine)
        self.corr_at_transition_euclidean.append(alphas_euclidean)
        self.corr_at_transition_siminet.append(alphas_siminet)
        self.corr_at_transition_bhatta.append(alphas_bhatta)
        self.corr_at_transition_gts.append(_gts_list)

    def _save_corr_at_transition(self):
        corr_cosine = np.array(self.corr_at_transition_cosine)
        corr_euclidean = np.array(self.corr_at_transition_euclidean)
        corr_bhatta = np.array(self.corr_at_transition_bhatta)
        corr_siminet = np.array(self.corr_at_transition_siminet)
        corr_gts = np.array(self.corr_at_transition_gts)
        idx = random.sample(range(corr_cosine.shape[0]), min(
            corr_cosine.shape[0], 10000))
        corr_cosine = corr_cosine[idx]
        corr_euclidean = corr_euclidean[idx]
        corr_bhatta = corr_bhatta[idx]
        corr_siminet = corr_siminet[idx]
        corr_gts = corr_gts[idx]
        np.save(self.opts.stats_dir + "/" + 'cosine_' +
                self.opts.datasets + '_transition_corr', corr_cosine)
        np.save(self.opts.stats_dir + "/" + 'euclidean_' +
                self.opts.datasets + '_transition_corr', corr_euclidean)
        np.save(self.opts.stats_dir + "/" + 'bhatta_' +
                self.opts.datasets + '_transition_corr', corr_bhatta)
        np.save(self.opts.stats_dir + "/" + 'siminet_' +
                self.opts.datasets + '_transition_corr', corr_siminet)
        np.save(self.opts.stats_dir + "/" + 'gts_' +
                self.opts.datasets + '_transition_corr', corr_gts)

    def frames_per_action(self, frame_count):
        if not self.opts.frames_per_action:
            return
        self.frames_per_action_lst.append(frame_count)

    def _save_frames_per_action(self):
        frames_count = np.array(self.frames_per_action_lst)
        np.save(self.opts.stats_dir + "/" + 'frames_per_action', frames_count)

    def filter_windows_sent(self, transition_point):
        if not self.opts.filter_windows_sent:
            return
        self.filter_windows_sent_cnt += 1
        if transition_point:
            self.filter_windows_sent_transition_cnt += 1

    def _save_filter_windows_sent(self):
        print("FilterWindowsSent: Total={}, Transition={}".format(self.filter_windows_sent_cnt,
                                                                  self.filter_windows_sent_transition_cnt))

    def save(self):
        if self.opts.scores_per_window:
            self._save_scores_per_window()
        elif self.opts.corr_per_window:
            self._save_corr_per_window()
        elif self.opts.corr_at_transition:
            self._save_corr_at_transition()
        elif self.opts.frames_per_action:
            self._save_frames_per_action()
        elif self.opts.filter_windows_sent:
            self._save_filter_windows_sent()

        self._save_video_stats()
