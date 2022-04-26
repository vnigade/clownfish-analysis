from opts import parse_opts
import video_utils
import siminet
from fusion import ScoreFusionModule
import evaluation as eval
from stats import FusionStats
import time
import csv

_FUSION_METHOD = "exponential_smoothing"
_VIDEO_LEVEL_METRIC = False
_EXTENSION = "avi"
# _VIDEOS_LST = ["0296-M", "0309-L", "312-L", "0322-L"]
_VIDEOS_LST = ["0322-L"]


def read_txt_labels(class_idx):
    with open(class_idx, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        next(reader)
        data_list = list(reader)

        txt_lables = {}
        for item in data_list[:-1]:
            txt_lables[int(item[0])] = item[1]

    return txt_lables


def show(opts, video, local_pred_actions, remote_pred_actions,
         fusion_pred_actions, true_actions, txt_labels):
    import cv2 as cv
    video_file = opts.datasets_dir + '/' + opts.datasets + \
        f'/videos/{video}.{_EXTENSION}'

    cap = cv.VideoCapture(video_file)
    if not(cap.isOpened()):
        print("Cannot open ", video_file)
        return

    frame_id = 0
    left_margin = 10
    gap_margin = 75
    while (True):
        try:
            ret, frame = cap.read()
            if ret is False:
                break

            if frame_id >= (opts.window_size // 2) < len(local_pred_actions):
                local_action_txt = txt_labels[local_pred_actions[frame_id]]
                cv.putText(frame, f"Local: {local_action_txt}", (left_margin, gap_margin*1),
                           cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
                remote_action_txt = txt_labels[remote_pred_actions[frame_id]]
                cv.putText(frame, f"Remote: {remote_action_txt}", (left_margin, gap_margin*2),
                           cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)
                fusion_action_txt = txt_labels[fusion_pred_actions[frame_id]]
                cv.putText(frame, f"Clownfish: {fusion_action_txt}", (left_margin, gap_margin*3),
                           cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)
                true_action_txt = txt_labels[true_actions[frame_id]]
                cv.putText(frame, f"GT: {true_action_txt}", (left_margin, gap_margin*4),
                           cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)
            else:
                # The number of frames in the videos could be higher than the number of predictions
                # because we do not take the very last background action into consideration in the
                # ground truth actions itself. That's how the ground truth json file is generated.
                # But the clownfish, local and remote predict all the frames. So, not an issue.
                # break
                pass

            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
            time.sleep(30 * 1e-3)
        except Exception as e:
            print(f"Exception {str(e)}")
            break
    print(
        f"Total frames: {frame_id}, {len(local_pred_actions)}, {len(true_actions)}")
    cap.release()
    cv.destroyAllWindows()


def get_pred_actions(opts, video, labels):
    use_siminet = (opts.sim_method ==
                   "siminet" or opts.corr_at_transition or opts.corr_per_window)

    # Find file paths where scores are dumped
    local_score_file = opts.local_scores_dir + "/" + video
    remote_score_file = opts.remote_scores_dir + "/" + video
    p_local_window = []
    p_remote_window = []
    p_local_feature_window = []  # for siminet
    if use_siminet:
        local_feature_file = opts.local_scores_dir + "/raw_features/" + video
    try:
        # Read prediction scores from the dump file
        p_local_window = video_utils.get_scores(
            local_score_file, use_softmax=False, flow=False)
        p_remote_window = video_utils.get_scores(
            remote_score_file, use_softmax=False, flow=False)

        if use_siminet:
            p_local_feature_window = video_utils.get_scores(
                local_feature_file, use_softmax=False, flow=False)
            assert len(p_local_window) == len(
                p_local_feature_window), "Softmax and Raw features windows does ot match"
    except:
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
    print(
        f"Accuracy: {video}: {metric[eval.LOCAL]}, {metric[eval.REMOTE]}, {metric[eval.PERCEIVED]}")
    local_pred_actions = eval_metric.y_pred_per_frame_lst[eval.LOCAL]
    remote_pred_actions = eval_metric.y_pred_per_frame_lst[eval.REMOTE]
    fusion_pred_actions = eval_metric.y_pred_per_frame_lst[eval.PERCEIVED]
    true_actions = eval_metric.y_true_per_frame_lst

    return (local_pred_actions, remote_pred_actions, fusion_pred_actions, true_actions)


def main():
    opts = parse_opts()

    # label_file = opts.datasets_dir + '/' + opts.datasets + '/splits/pkummd.json'
    label_file = opts.datasets_dir + '/' + opts.datasets + \
        '/splits/pkummd_cross_subject_background.json'
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
        ret = get_pred_actions(opts=opts, video=video, labels=labels)
        if ret is not None:
            (local_pred_actions,
             remote_pred_actions,
             fusion_pred_actions,
             true_actions) = ret

            show(opts, video, local_pred_actions, remote_pred_actions,
                 fusion_pred_actions, true_actions, txt_labels)


if __name__ == '__main__':
    main()
