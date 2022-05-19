"""
A Clownfish visualizer using OpenCV.
"""
import cv2 as cv
import time

from .types import ActionList, ActionLabels, PredictionList


class VisualizerOpenCV:
    """
    An OpenCV-based visualizer.
    """

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, video: cv.VideoCapture, window_size: int, predictions: PredictionList, true_actions: ActionList, label_dict: ActionLabels, target_fps: int = 30):
        assert len(predictions) == len(true_actions)
        self._video: cv.VideoCapture = video
        self._window_size: int = window_size
        self._target_fps: int = target_fps
        self._predictions: PredictionList = predictions
        self._true_actions: ActionList = true_actions
        self._label_dict: ActionLabels = label_dict
        self._window_id: str = "video"

    def start(self) -> None:
        """
        Starts displaying of the video with predicted and ground truth action labels.

        :return: None.
        """

        local_predictions, remote_predictions, fusion_predictions = zip(*self._predictions)
        frame_id = 0
        left_margin = 10
        gap_margin = 75

        while True:
            try:
                ret, frame = self._video.read()
                if ret is False:
                    break

                if (self._window_size // 2) <= frame_id < len(local_predictions):
                    local_action = local_predictions[frame_id]
                    remote_action = remote_predictions[frame_id]
                    fusion_action = fusion_predictions[frame_id]
                    true_action = self._true_actions[frame_id]

                    local_action_txt = self._label_dict[local_action]
                    remote_action_txt = self._label_dict[remote_action]
                    fusion_action_txt = self._label_dict[fusion_action]
                    true_action_txt = self._label_dict[true_action]

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
                cv.setWindowTitle(self._window_id, f"Clownfish (frame {frame_id} - fps: {self._target_fps})")
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_id += 1
                time.sleep(float(self._target_fps) * 1e-3)

            except Exception as e:
                print(f"Exception {str(e)}")
                break

        print(f"Total frames: {frame_id}, {len(local_predictions)}, {len(self._true_actions)}")
        cv.destroyAllWindows()
