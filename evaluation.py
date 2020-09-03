from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import ml_utils as ml
import siminet
SOLUTION_TYPES = 3
LOCAL = 0
REMOTE = 1
PERCEIVED = 2


class EvaluationMetric:
    """
    Evaluate scores based on the different metrics
    """

    def __init__(self, opts, avg=True, labels=False, stats=None):

        self.opts = opts
        # Variables for per window accuracy
        self.y_pred_per_window = [[] for i in range(SOLUTION_TYPES)]
        self.y_true_per_window = []

        # Variables for for mAP
        self.y_pred = [[] for i in range(SOLUTION_TYPES)]
        self.y_true = []

        # Variables for per frame accuracy
        self.y_pred_per_frame = [[] for i in range(SOLUTION_TYPES)]
        self.y_true_per_frame = []
        self.y_pred_per_frame_lst = [[] for i in range(SOLUTION_TYPES)]
        self.y_true_per_frame_lst = []

        # Variables for per context accuracy. Context is consecutive windows encompassing an action.
        self.per_context_accuracy = [[] for i in range(SOLUTION_TYPES)]

        # some stats
        self.total_actions = 0
        self.avg = avg

        self.labels = labels
        self.stats = stats

    def update_per_frame(self, window_idx, local_action, remote_action, perceived_action):
        # print("-window_{}".format(window_idx), "RGB action", local_action - 1)
        if window_idx == 0:  # First window
            for i in range(self.opts.window_size):
                self.y_pred_per_frame[LOCAL].append(local_action)
                self.y_pred_per_frame[REMOTE].append(remote_action)
                self.y_pred_per_frame[PERCEIVED].append(perceived_action)
            self.prev_action = [0 for i in range(SOLUTION_TYPES)]
        else:
            # First fill the gap of frames due to stride, use prev action
            for i in range(self.opts.window_stride-1):
                self.y_pred_per_frame[LOCAL].append(self.prev_action[LOCAL])
                self.y_pred_per_frame[REMOTE].append(self.prev_action[REMOTE])
                self.y_pred_per_frame[PERCEIVED].append(
                    self.prev_action[PERCEIVED])

            # Now append for the latest frame
            self.y_pred_per_frame[LOCAL].append(local_action)
            self.y_pred_per_frame[REMOTE].append(remote_action)
            self.y_pred_per_frame[PERCEIVED].append(perceived_action)

        # Update the previous action
        self.prev_action[LOCAL] = local_action
        self.prev_action[REMOTE] = remote_action
        self.prev_action[PERCEIVED] = perceived_action

    def append_per_frame_lst(self, video):
        def get_action_range(idx, max_idx):
            start_action = idx
            action_class = target_per_frame[idx]
            assert action_class != -1, "Target action cannot be negative"
            while idx < max_idx:
                target = target_per_frame[idx]
                if target == -1 or target != action_class:
                    if target != -1:
                        print("Actions are contigous", video, action_class, target)
                    break
                idx += 1
            self.stats.frames_per_action((idx - start_action))
            return start_action, idx

        def context_accuracy(pred, true):
            pred = np.array(pred)
            true = np.array(true)
            matched = (true == pred)
            tp = float(matched.sum())
            total = len(true)
            result = tp / total
            return result

        target_per_frame = self.labels.get_action_per_frame(video)
        frame_len = len(self.y_pred_per_frame[LOCAL])
        cur_frame = self.opts.window_size
        target_frame_len = len(target_per_frame)
        # print(video, "Target frame len", target_frame_len, "Predicted frame length", frame_len)

        while cur_frame < target_frame_len and cur_frame < frame_len:
            target = target_per_frame[cur_frame]
            # print("Target", i+1, target)
            if target == -1:  # Only consider valid actions
                # print("TEST", video, i+1, target+1, self.y_pred_per_frame[LOCAL][i])
                cur_frame += 1
                continue
            else:
                start_action, end_action = get_action_range(
                    cur_frame, min(target_frame_len, frame_len))

            # Skip first few frames, for fair comparison we use max window size.
            # If we consider model can be too random even for the single unseen frame,
            # we should use the factor of 1.
            factor = 2
            start_action += (self.opts.window_size // factor)
            if start_action < end_action:
                pred = [[] for i in range(SOLUTION_TYPES)]
                true = []
                for j in range(start_action, end_action):
                    local_action = self.y_pred_per_frame[LOCAL][j]
                    remote_action = self.y_pred_per_frame[REMOTE][j]
                    perceived_action = self.y_pred_per_frame[PERCEIVED][j]
                    target = target_per_frame[j]

                    assert target != - \
                        1 and target != 0, "Target is {}".format(target)

                    self.y_pred_per_frame_lst[LOCAL].append(local_action)
                    self.y_pred_per_frame_lst[REMOTE].append(remote_action)
                    self.y_pred_per_frame_lst[PERCEIVED].append(
                        perceived_action)
                    self.y_true_per_frame_lst.append(target)
                    # self.y_true_per_frame_lst.append(remote_action)
                    # print("TEST", video, j+1, target, local_action)
                    # for context calculations
                    pred[LOCAL].append(local_action)
                    pred[REMOTE].append(remote_action)
                    pred[PERCEIVED].append(perceived_action)
                    true.append(target)

                # context calculation
                # print("{}-{}".format(start_action, end_action))
                for model_type in range(SOLUTION_TYPES):
                    acc = context_accuracy(pred=pred[model_type], true=true)
                    self.per_context_accuracy[model_type].append(acc)

            if end_action + 1 == target_frame_len or end_action + 1 == frame_len:
                end_action += 1
            cur_frame = end_action

        # reset per frame list
        self.y_pred_per_frame = [[] for i in range(SOLUTION_TYPES)]
        self.y_true_per_frame = []

    def update_accuracy(self, video, expected_action, p_scores_local, p_scores_remote, p_scores_perceived, p_local_feature_window):

        if self.avg:  # aggregate score at video level
            actions = [None for i in range(SOLUTION_TYPES)]
            actions[REMOTE] = np.mean(p_scores_remote, axis=0)
            actions[LOCAL] = np.mean(p_scores_local, axis=0)
            actions[PERCEIVED] = np.mean(p_scores_perceived, axis=0)
            gt = video_utils.VideoLabels.get_action_vector(
                self.opts.datasets, expected_action, self.opts.n_classes)
        else:
            actions = [[] for i in range(SOLUTION_TYPES)]
            gts = []
            last_avg_scores_local = p_scores_local[0]
            last_avg_scores_remote = p_scores_remote[0]
            prev_background = -1
            for i in range(len(p_scores_local)):
                if self.opts.fixma_on_models:
                    last_avg_scores_local = 0.5 * \
                        last_avg_scores_local + 0.5 * p_scores_local[i]
                    last_avg_scores_remote = 0.5 * \
                        last_avg_scores_remote + 0.5 * p_scores_remote[i]
                else:
                    last_avg_scores_local = p_scores_local[i]
                    last_avg_scores_remote = p_scores_remote[i]

                actions[REMOTE].append(last_avg_scores_remote)
                actions[LOCAL].append(last_avg_scores_local)
                actions[PERCEIVED].append(p_scores_perceived[i])
                if self.opts.datasets == 'PKUMMD':
                    start_frame = i * self.opts.window_stride + 1  # consitent with the training
                else:
                    start_frame = i * self.opts.window_stride
                end_frame = start_frame + self.opts.window_size
                gt = self.labels.get_action_vector_window(
                    video, self.opts.n_classes, None, start_frame, end_frame)
                gts.append(gt)

                # Note: We are missing a proper/standard metric on video stream classification.
                background = np.argmax(gt) if np.max(gt) == 1 else -1
                if background != -1:  # valid action
                    local_action = np.argmax(last_avg_scores_local) + 1
                    remote_action = np.argmax(last_avg_scores_remote) + 1
                    perceived_action = np.argmax(p_scores_perceived[i]) + 1
                    y_true = np.argmax(gt) + 1
                    self.y_pred_per_window[LOCAL].append(local_action)
                    self.y_pred_per_window[REMOTE].append(remote_action)
                    self.y_pred_per_window[PERCEIVED].append(perceived_action)
                    self.y_true_per_window.append(y_true)
                    self.total_actions += 1
                    # print("Video", video, i, "Local action", np.argmax(p_scores_local[i]), "Remote action", np.argmax(p_scores_remote[i]),
                    #      "Ground truth", np.argmax(gt))

                    # update per frame action
                    self.update_per_frame(
                        i, local_action, remote_action, perceived_action)

                    if prev_background == -1:
                        # context transition at this point
                        self.stats.corr_at_transition(
                            i, video, p_scores_local, p_local_feature_window, gts)
                        # go one more level back
                        # if (i-2) > 0:
                        #    stat_alpha = ml.cosine_similarity(p_scores_local[i-1], p_scores_local[i-2])
                        #    print("CORR_HIST", stat_alpha)

                else:
                    local_action = np.argmax(last_avg_scores_local) + 1
                    remote_action = np.argmax(last_avg_scores_remote) + 1
                    perceived_action = np.argmax(p_scores_perceived[i]) + 1
                    # per frame action update
                    # self.update_per_frame(i, -1, -1, -1)
                    self.update_per_frame(
                        i, local_action, remote_action, perceived_action)
                prev_background = background

                # STATS: print similarity alphas for all scores, whether action is present or not
                if i > 0:
                    cosine_alpha = ml.cosine_similarity(
                        p_scores_local[i-1], p_scores_local[i])
                    kl_alpha = ml.kl_similarity(
                        p_scores_local[i-1], p_scores_local[i])
                    euclidean_alpha = ml.euclidean_similarity(
                        p_scores_local[i-1], p_scores_local[i])
                    bhatta_alpha = ml.bhatta_similarity(
                        p_scores_local[i-1], p_scores_local[i])

                    siminet_alpha = 0
                    if self.opts.sim_method == "siminet":
                        siminet_alpha = siminet.siminet_similarity(prev_vec=p_local_feature_window[i-1],
                                                                   cur_vec=p_local_feature_window[i])
                    now_action = 1 if background != -1 else 0
                    # print(i, "SIMILARITY", now_action, "siminet", siminet_alpha, "cosine", cosine_alpha, "euclidean", euclidean_alpha, "bhatta", bhatta_alpha)
                    # print(i, "Probability", background + 1, "Local:[{},{}]".format(local_action, p_scores_local[i][local_action-1]), "Remote:[{},{}]".format(remote_action, p_scores_remote[i][remote_action-1]), "Perceived:[{},{}]".format(perceived_action, p_scores_perceived[i][perceived_action - 1]))

        for i in range(SOLUTION_TYPES):
            if self.avg:
                self.y_pred[i].append(actions[i])
            else:
                self.y_pred[i].extend(actions[i])

        if self.avg:
            self.y_true.append(gt)
        else:
            self.y_true.extend(gts)

            # append per frame score
            self.append_per_frame_lst(video)

        # STATS: correlation and gts
        self.stats.corr_per_window(video, p_scores_local,
                                   p_scores_remote, gts, p_local_feature_window)
        self.stats.scores_per_window(
            video, p_scores_local, p_scores_remote, gts)

    def get_top1_accuracy_window(self):
        acc = [0.0 for i in range(SOLUTION_TYPES)]
        for i in range(SOLUTION_TYPES):
            acc[i] = accuracy_score(
                self.y_true_per_window, self.y_pred_per_window[i])

        return acc

    def get_f1_score_window(self):
        acc = [0.0 for i in range(SOLUTION_TYPES)]
        labels = list(range(1, self.opts.n_classes+1))
        for i in range(SOLUTION_TYPES):
            acc[i] = f1_score(
                self.y_true_per_window, self.y_pred_per_window[i], average='weighted', labels=labels)
        return acc

    def get_mAP(self):
        acc = [0.0 for i in range(SOLUTION_TYPES)]
        gt = self.y_true
        for i in range(SOLUTION_TYPES):
            pred = self.y_pred[i]
            # acc[i], _, _ = map.charades_map(np.vstack(pred), np.vstack(gt))
        return acc

    def get_top1_accuracy_frame(self):
        acc = [0.0 for i in range(SOLUTION_TYPES)]
        for i in range(SOLUTION_TYPES):
            acc[i] = accuracy_score(
                self.y_true_per_frame_lst, self.y_pred_per_frame_lst[i])

        return acc

    def get_f1_score_frame(self):
        acc = [0.0 for i in range(SOLUTION_TYPES)]
        labels = list(range(1, self.opts.n_classes+1))
        for i in range(SOLUTION_TYPES):
            acc[i] = f1_score(
                # self.y_true_per_frame_lst, self.y_pred_per_frame_lst[i], average='macro')
                # 'Macro' is the unweighted mean of f1 score of all classes. Doesn't take into account the class imbalance
                self.y_true_per_frame_lst, self.y_pred_per_frame_lst[i], average='weighted', labels=labels)

        return acc

    def print_mAP(self, msg=""):
        metric_values = self.get_mAP()
        print("mAP",
              "Local=", metric_values[LOCAL],
              "Remote=", metric_values[REMOTE],
              "Perceived=", metric_values[PERCEIVED],
              "Message=", msg)

    def print_accuracy_window(self, msg=""):
        metric_type = 'PerWindowAccuracy'
        metric_values = self.get_top1_accuracy_window()
        for _ in 0, 1:
            print(metric_type,
                  "Local=", metric_values[LOCAL],
                  "Remote=", metric_values[REMOTE],
                  "Perceived=", metric_values[PERCEIVED],
                  "Message=", msg)
            metric_type = 'PerWindowF1Score'
            metric_values = self.get_f1_score_window()

    def print_accuracy_frame(self, msg=""):
        metric_type = 'PerFrameAccuracy'
        metric_values = self.get_top1_accuracy_frame()
        for _ in 0, 1:
            print(metric_type,
                  "Local=", metric_values[LOCAL],
                  "Remote=", metric_values[REMOTE],
                  "Perceived=", metric_values[PERCEIVED],
                  "Message=", msg)
            metric_type = 'PerFrameF1Score'
            metric_values = self.get_f1_score_frame()
