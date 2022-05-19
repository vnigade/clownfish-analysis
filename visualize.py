import csv
import evaluation as eval
import siminet
import video_utils

from fusion import ScoreFusionModule
from opts import parse_opts
from stats import FusionStats
from visualizer import Visualizer

_FUSION_METHOD = "exponential_smoothing"
_VIDEO_LEVEL_METRIC = False
# _VIDEOS_LST = ["0296-M", "0309-L", "312-L", "0322-L"]
_VIDEOS_LST = ["0322-L"]
_VIDEO_EXTENSION: str = "avi"
_TARGET_FPS: int = 30
_USE_QT: bool = True


def read_txt_labels(class_idx):
    with open(class_idx, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        next(reader)
        data_list = list(reader)

        txt_lables = {}
        for item in data_list[:-1]:
            txt_lables[int(item[0])] = item[1]

    return txt_lables


def get_predicted_actions(opts, video, labels):
    use_siminet = opts.sim_method == "siminet" or opts.corr_at_transition or opts.corr_per_window

    # Find file paths where scores are dumped
    local_score_file = opts.local_scores_dir + "/" + video
    remote_score_file = opts.remote_scores_dir + "/" + video
    p_local_feature_window = []  # for siminet

    try:
        # Read prediction scores from the dump file
        p_local_window = video_utils.get_scores(local_score_file, use_softmax=False, flow=False)
        p_remote_window = video_utils.get_scores(remote_score_file, use_softmax=False, flow=False)

        if use_siminet:
            local_feature_file = opts.local_scores_dir + "/raw_features/" + video
            p_local_feature_window = video_utils.get_scores(local_feature_file, use_softmax=False, flow=False)
            assert len(p_local_window) == len(p_local_feature_window), "Softmax and Raw features windows does ot match"
    except:
        # todo: Never use an empty except.
        return None

    # Get fusion prediction scores
    stats = FusionStats(opts)
    fusion_module = ScoreFusionModule(opts,
                                      p_local_window,
                                      p_remote_window,
                                      p_local_feature_window,
                                      video=video,
                                      labels=labels, stats=stats)
    p_perceived_window = fusion_module.get_perceived_score()

    # Get per frame actions
    eval_metric = eval.EvaluationMetric(opts=opts,
                                        avg=_VIDEO_LEVEL_METRIC,
                                        labels=labels, stats=stats)
    eval_metric.update_accuracy(video, None, p_local_window,
                                p_remote_window, p_perceived_window,
                                p_local_feature_window)
    metric = eval_metric.get_top1_accuracy_frame()
    print(f"Accuracy: {video}: {metric[eval.LOCAL]}, {metric[eval.REMOTE]}, {metric[eval.PERCEIVED]}")
    local_predicted_actions = eval_metric.y_pred_per_frame_lst[eval.LOCAL]
    remote_predicted_actions = eval_metric.y_pred_per_frame_lst[eval.REMOTE]
    fusion_predicted_actions = eval_metric.y_pred_per_frame_lst[eval.PERCEIVED]
    true_actions = eval_metric.y_true_per_frame_lst

    return local_predicted_actions, remote_predicted_actions, fusion_predicted_actions, true_actions


def main():
    # Parse arguments (options)
    opts = parse_opts()

    # Read labels
    # label_file = opts.datasets_dir + '/' + opts.datasets + '/splits/pkummd.json'
    label_file = opts.datasets_dir + '/' + opts.datasets + '/splits/pkummd_cross_subject_background.json'
    class_file = opts.datasets_dir + '/' + opts.datasets + '/splits/classInd.txt'
    labels = video_utils.VideoLabels(label_file, class_file, opts.datasets)
    txt_labels = read_txt_labels(class_file)

    assert _FUSION_METHOD == "exponential_smoothing", "Fusion method NotImplemented"

    use_siminet = (opts.sim_method ==
                   "siminet" or opts.corr_at_transition or opts.corr_per_window)
    if use_siminet:
        assert opts.siminet_path != '', "Siminet model path cannot be empty"
        siminet.load_siminet_model(
            opts.n_classes, opts.siminet_path)

    # _VIDEOS_LST = labels.labels_key
    for video in _VIDEOS_LST:
        ret = get_predicted_actions(opts=opts, video=video, labels=labels)
        if ret is not None:
            local_predicted_actions, remote_predicted_actions, fusion_predicted_actions, true_actions = ret
            predictions = list(zip(local_predicted_actions, remote_predicted_actions, fusion_predicted_actions))
            assert len(predictions) == len(true_actions)

            video_file = opts.datasets_dir + "/" + opts.datasets + f"/videos/{video}.{_VIDEO_EXTENSION}"
            window_size = opts.window_size

            try:
                visualizer = Visualizer(video_file, window_size, predictions, true_actions, txt_labels, target_fps=_TARGET_FPS, use_qt=_USE_QT)
                visualizer.display()
            except Visualizer.VideoError as e:
                print(f"Cannot open video file: {e.filename}")


if __name__ == '__main__':
    main()
