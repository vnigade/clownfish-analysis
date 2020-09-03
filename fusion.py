import numpy as np
import siminet


class ScoreFusionModule:
    def __init__(self, opts, p_local_window, p_remote_window,
                 p_local_feature_window, video=None, labels=None, stats=None):
        self.opts = opts
        self.p_local_window = p_local_window
        self.p_remote_window = p_remote_window
        self.p_local_feature_window = p_local_feature_window
        self.p_perceived_window = np.zeros(p_local_window.shape, dtype=float)
        self.p_history_window = np.zeros(p_local_window.shape, dtype=float)
        self.accumulated_alpha = np.zeros(p_local_window.shape[0], dtype=float)
        # Filter variables
        self.filter_queue = []
        self.filter_timeout = 0
        self.prev_alpha = 0.0

        self.labels = labels
        self.video = video
        self.stats = stats

        self.ground_truths()
        self.ensemble_scores()
        self.temporal_correction()

    def ensemble_scores(self):
        '''
        Pre-cache the ensemble scores
        '''
        weight = 0.33
        self.p_ensemble_window = (
            weight * self.p_local_window + (1.0-weight) * self.p_remote_window)

    def get_ensemble_score(self, idx):
        score = self.p_ensemble_window[idx]
        threshold = 0.0  # No thresholding when 0.0
        if np.max(score) < threshold:
            return None
        return score

    def ground_truths(self):
        def ground_truth(i):
            start_frame = i * self.opts.window_stride + 1
            end_frame = start_frame + self.opts.window_size
            gt = self.labels.get_action_vector_window(
                self.video, self.opts.n_classes, None, start_frame, end_frame)
            return gt

        self.p_gts_window = np.zeros(self.p_local_window.shape, dtype=float)
        for i in range(len(self.p_gts_window)):
            gt = ground_truth(i)
            if np.sum(gt) > 0.0:
                self.p_gts_window[i] = gt
            # else:
            #    self.p_gts_window[i] = self.p_remote_window[i]

    def get_similarity(self, i):
        if self.opts.sim_method == "euclidean":
            alpha = ml.euclidean_similarity(
                self.p_local_window[i-1], self.p_local_window[i])
        elif self.opts.sim_method == "cosine":
            alpha = ml.cosine_similarity(
                self.p_local_window[i-1], self.p_local_window[i])
        elif self.opts.sim_method == "opt_sim":
            gt_prev = self.p_gts_window[i-1]
            gt_curr = self.p_gts_window[i]
            alpha = 0.0
            if np.max(gt_curr) > 0 and np.max(gt_prev) > 0 and np.argmax(gt_curr) == np.argmax(gt_prev):
                alpha = 1.0
        elif self.opts.sim_method == "fix_ma":
            alpha = 0.5
        elif self.opts.sim_method == "siminet":
            alpha = siminet.siminet_similarity(prev_vec=self.p_local_feature_window[i-1],
                                               cur_vec=self.p_local_feature_window[i])
        else:
            print("Not implemented")
            exit()

        return alpha

    def get_corelation_coefficent(self, i):
        alpha = self.get_similarity(i)

        threshold = 0.5
        if self.accumulated_alpha[i-1] < threshold:
            # That means the remote contribution has been lowered.
            # Switch to constant moving average.
            alpha = 0.5
        else:
            # High similarity and more contribution from the remote score.
            # print("High similarity {0} and accumulation {1}".format(
            #    alpha, self.accumulated_alpha[i-1]))
            pass
        self.accumulated_alpha[i] = self.accumulated_alpha[i-1] * alpha
        # print("Alpha idx:", i, alpha, self.accumulated_alpha[i-1])
        return alpha

    def exponential_smoothing(self, idx):
        def filter(idx, alpha):
            send = False
            if self.opts.send_at_transition:
                # transition_score = abs(self.prev_alpha - alpha)
                transition_score = alpha - self.prev_alpha  # Only send on uptrend
            else:
                transition_score = 0
            self.prev_alpha = alpha

            if self.filter_timeout == 0 or transition_score >= self.opts.transition_threshold:
                send = True

                # STATS: Filter windows sent
                transition_point = False
                if self.filter_timeout != 0:
                    transition_point = True
                self.stats.filter_windows_sent(transition_point)

            # print("Transition score: ", idx, transition_score)
            if send:
                self.filter_queue.append(idx)
                self.filter_timeout = self.opts.filter_interval - 1
            else:
                self.filter_timeout -= 1

        def reinforce(idx):
            if self.opts.remote_lag == 0:
                return

            back_window = idx - self.opts.remote_lag
            # print("Back window", back_window, "Filter queue:", self.filter_queue)
            if len(self.filter_queue) > 0 and back_window == self.filter_queue[0]:
                # print("Update back window", idx, back_window, "Filter queue:", self.filter_queue)
                del self.filter_queue[0]
                # We received feedback from the remote
                back_window = idx - self.opts.remote_lag
                # self.p_history_window[back_window] = self.p_ensemble_window[back_window]
                ensemble_score = self.get_ensemble_score(back_window)
                if ensemble_score is None:
                    return
                self.p_history_window[back_window] = ensemble_score
                # self.accumulated_alpha[back_window] = 1.0
                self.accumulated_alpha[back_window] = np.max(
                    self.p_history_window[back_window])

                for i in range(back_window + 1, idx):
                    alpha = self.get_corelation_coefficent(i)
                    self.p_history_window[i] = alpha * self.p_history_window[i -
                                                                             1] + (1.0 - alpha) * self.p_local_window[i]

        def fuse(idx):
            alpha = self.get_corelation_coefficent(idx)
            if self.opts.remote_lag == 0 and len(self.filter_queue) > 0 and idx == self.filter_queue[0]:
                # print("Using remote score:", idx)
                del self.filter_queue[0]
                # p_perceived_window = self.p_ensemble_window[idx]
                ensemble_score = self.get_ensemble_score(idx)
                if ensemble_score is not None:
                    p_perceived_window = ensemble_score
                    # self.accumulated_alpha[idx] = 1.0
                    self.accumulated_alpha[idx] = np.max(p_perceived_window)
                else:
                    p_perceived_window = alpha * \
                        self.p_history_window[idx - 1] + \
                        (1.0 - alpha) * self.p_local_window[idx]
            else:
                p_perceived_window = alpha * \
                    self.p_history_window[idx - 1] + \
                    (1.0 - alpha) * self.p_local_window[idx]

            return p_perceived_window

        sim = self.get_similarity(idx)
        filter(idx, sim)
        # print("Filter Queue: ", idx, self.filter_queue)
        reinforce(idx)

        return fuse(idx)

    def smoothing_prediction(self, idx):
        # First local prediction
        if idx == 0:
            return self.p_local_window[idx]
        p_perceived_window = self.exponential_smoothing(idx)
        return p_perceived_window

    def temporal_correction(self):
        tot_windows = len(self.p_local_window)
        for i in range(tot_windows):

            self.p_perceived_window[i] = self.smoothing_prediction(i)

            # print('Perceived scores sum: %f' %
            #      (np.sum(self.p_perceived_window[i])))
            self.p_history_window[i] = self.p_perceived_window[i]
        return

    def get_perceived_score(self):
        return self.p_perceived_window

    def print_window_actions(self):
        for i in range(len(self.p_perceived_window)):
            action = np.argmax(self.p_perceived_window[i])
            print("Preceived actions for window ", action, i)
