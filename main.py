'''
Fuse and analyze scores.
'''
import os.path
from opts import parse_opts
import numpy as np
import ml_utils as ml
import video_utils
import siminet
from fusion import ScoreFusionModule
from evaluation import EvaluationMetric
from stats import FusionStats


_FUSION_METHOD = "exponential_smoothing"
_VIDEO_LEVEL_METRIC = False


def main():
    opts = parse_opts()

    label_file = opts.datasets_dir + '/' + opts.datasets + '/splits/pkummd.json'
    class_file = opts.datasets_dir + '/' + opts.datasets + '/splits/classInd.txt'
    labels = video_utils.VideoLabels(label_file, class_file, opts.datasets)

    assert _FUSION_METHOD == "exponential_smoothing", "Fusion method NotImplemented"

    use_siminet = (opts.sim_method ==
                   "siminet" or opts.corr_at_transition or opts.corr_per_window)
    if use_siminet:
        assert opts.siminet_path != '', "Siminet model path cannot be empty"
        siminet.load_siminet_model(
            opts.n_classes, opts.siminet_path)

    stats = FusionStats(opts)
    eval_metric = EvaluationMetric(
        opts=opts, avg=_VIDEO_LEVEL_METRIC, labels=labels, stats=stats)

    for video, action in labels:
        local_score_file = opts.local_scores_dir + "/" + video
        remote_score_file = opts.remote_scores_dir + "/" + video
        p_local_window = []
        p_remote_window = []
        p_local_feature_window = []  # for siminet
        if use_siminet:
            local_feature_file = opts.local_scores_dir + "/raw_features/" + video
        try:
            if opts.datasets == 'CHARADES':
                use_softmax = True
                flow = False
            elif opts.datasets == 'THUMOS2014' or opts.datasets == 'UCF101':
                use_softmax = True
                flow = True
            elif opts.datasets == 'PKUMMD':
                use_softmax = False
                flow = False
            p_local_window = video_utils.get_scores(
                local_score_file, use_softmax=use_softmax, flow=False)
            p_remote_window = video_utils.get_scores(
                remote_score_file, use_softmax=use_softmax, flow=flow)

            if use_siminet:
                p_local_feature_window = video_utils.get_scores(
                    local_feature_file, use_softmax=False, flow=False)
                assert len(p_local_window) == len(
                    p_local_feature_window), "Softmax and Raw features windows does ot match"
                # print("Siminet", len(p_local_window), len(p_local_feature_window))

        except:
            # print("Video {} cannot be parsed".format(video))
            continue

        fusion_module = ScoreFusionModule(
            opts, p_local_window, p_remote_window, p_local_feature_window, video=video, labels=labels, stats=stats)

        # fusion_module.print_window_actions()
        p_perceived_window = fusion_module.get_perceived_score()
        eval_metric.update_accuracy(video, None, p_local_window,
                                    p_remote_window, p_perceived_window, p_local_feature_window)

        # Basic stats update
        stats.basic_video_stats(video_cnt=1, windows_cnt=len(p_local_window))

    # Final accuracy calculation on entire test sets
    # eval_metric.print_accuracy_window()
    eval_metric.print_accuracy_frame()

    # Dump stats
    stats.save()

    print("Total actions", eval_metric.total_actions)


if __name__ == '__main__':
    main()
