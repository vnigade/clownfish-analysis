"""
A simple visualizer using OpenCV.
"""
import argparse
import cv2 as cv
import numpy as np
import time


class VisualizerOpenCV:
    # Define video extension
    VIDEO_EXTENSION: str = "avi"
    # Define Action type
    Action = np.int64

    def __init__(self, opts: argparse.Namespace, video: str, fps: float = 30.0):
        self._video_file: str = opts.datasets_dir + "/" + opts.datasets + f"/videos/{video}.{VisualizerOpenCV.VIDEO_EXTENSION}"
        self._window_size: int = opts.window_size
        self._fps: float = fps
        self._window_id = "video"

    def show(self,
             local_predicted_actions: list[Action],
             remote_predicted_actions: list[Action],
             fusion_predicted_actions: list[Action],
             true_actions: list[int],
             txt_labels: dict[int, str]):

        cap = cv.VideoCapture(self._video_file)
        if not (cap.isOpened()):
            print("Cannot open ", self._video_file)
            return

        frame_id = 0
        left_margin = 10
        gap_margin = 75
        while True:
            try:
                ret, frame = cap.read()
                if ret is False:
                    break

                if frame_id >= (self._window_size // 2) < len(local_predicted_actions):
                    local_action = local_predicted_actions[frame_id]
                    remote_action = remote_predicted_actions[frame_id]
                    fusion_action = fusion_predicted_actions[frame_id]
                    true_action = true_actions[frame_id]

                    local_action_txt = txt_labels[local_action]
                    remote_action_txt = txt_labels[remote_action]
                    fusion_action_txt = txt_labels[fusion_action]
                    true_action_txt = txt_labels[true_action]

                    cv.putText(frame, f"Local: {local_action_txt}", (left_margin, gap_margin * 1), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
                    cv.putText(frame, f"Remote: {remote_action_txt}", (left_margin, gap_margin * 2), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)
                    cv.putText(frame, f"Clownfish: {fusion_action_txt}", (left_margin, gap_margin * 3), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)
                    cv.putText(frame, f"GT: {true_action_txt}", (left_margin, gap_margin * 4), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)
                else:
                    # The number of frames in the videos could be higher than the number of predictions
                    # because we do not take the very last background action into consideration in the
                    # ground truth actions itself. That's how the ground truth json file is generated.
                    # But the clownfish, local and remote predict all the frames. So, not an issue.
                    # break
                    pass

                cv.imshow(self._window_id, frame)
                cv.setWindowTitle(self._window_id, f"Clownfish (frame {frame_id} - fps: {self._fps:.1f})")
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_id += 1
                time.sleep(self._fps * 1e-3)

            except Exception as e:
                print(f"Exception {str(e)}")
                break

        print(f"Total frames: {frame_id}, {len(local_predicted_actions)}, {len(true_actions)}")
        cap.release()
        cv.destroyAllWindows()
