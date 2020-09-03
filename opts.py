import argparse


def parse_opts():
    parser = argparse.ArgumentParser(
        description="Analyze Clownfish fusion performance")

    parser.add_argument("--datasets", type=str,
                        default='PKUMMD', help='Dataset to use')
    parser.add_argument("--n_classes", type=int,
                        default=51, help='Number of actions in the datasets')
    parser.add_argument("--datasets_dir", type=str, default='./',
                        help='Root directory to find datasets label/class files')
    parser.add_argument("--local_scores_dir",
                        help='Local scores dump directory')
    parser.add_argument("--remote_scores_dir",
                        help='Remote scores dump directory')
    parser.add_argument("--filter_interval", type=int, default=6,
                        help="Period interval in terms of number of windows")
    parser.add_argument("--remote_lag", type=int, default=0,
                        help="Remote lag in terms of number of windows")
    parser.add_argument("--sim_method", type=str, default='fix_ma',
                        choices=['fix_ma', 'cosine', 'opt_sim', 'siminet'])
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--siminet_path", type=str, default='')
    parser.add_argument("--fixma_on_models", action='store_true',
                        help="Apply fix_ma on local and remote models")
    parser.add_argument("--send_at_transition", action='store_false',
                        help="Send windows at transition point")
    parser.add_argument("--transition_threshold", type=float, default=0.5)
    #===============================STATS=============================================#
    parser.add_argument("--basic_video_stats", action='store_false',
                        help="Collect total video and window count")
    parser.add_argument("--corr_at_transition", action='store_true',
                        help="Collect correlation value at transition point")
    parser.add_argument("--frames_per_action", action='store_true',
                        help="Count number of frames per action")
    parser.add_argument("--scores_per_window",
                        action='store_true', help="Collect scores per window")
    parser.add_argument("--corr_per_window", action='store_true',
                        help="Collect correlation values per window")
    parser.add_argument("--filter_windows_sent",
                        action='store_true', help="Count filtered windows")
    parser.add_argument("--stats_dir", type=str, default="./")
    #=================================================================================#

    args = parser.parse_args()

    args_dict = args.__dict__
    print("------------------------------------")
    print("Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    return args
