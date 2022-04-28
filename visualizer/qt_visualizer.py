"""
A Clownfish visualizer using Qt.
"""
import argparse
import cv2 as cv
import pathlib

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *

from typing import Optional


class MainWindow(QMainWindow):
    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)
        ui_file = pathlib.Path(__file__).with_name("qt_visualizer.ui")
        loadUi(ui_file, self)
        self.centralWidget.layout().setContentsMargins(9, 9, 9, 9)
        self.centralWidget.layout().setSpacing(6)

    def center(self):
        geometry = self.frameGeometry()
        cursor_position = QApplication.desktop().cursor().pos()
        screen = QApplication.desktop().screenNumber(cursor_position)
        center_point = QApplication.desktop().screenGeometry(screen).center()
        geometry.moveCenter(center_point)
        self.move(geometry.topLeft())


class VisualizerQt:
    # Define video extension
    VIDEO_EXTENSION: str = "avi"

    # Define coloring stylesheets for correct and wrong predictions
    CORRECT_CSS: str = "color: green;"
    WRONG_CSS: str = "color: red;"

    # Define Action and Prediction and list types
    Action = int
    ActionList = list[Action]
    Prediction = tuple[Action, Action, Action]
    PredictionList = list[Prediction]

    def __init__(self, opts: argparse.Namespace, video: str, fps: float = 30.0, *args):
        self._main_window = MainWindow(*args)
        self._main_window.show()
        self._main_window.playButton.clicked.connect(self.play_or_pause)
        self._main_window.restartButton.clicked.connect(self.restart)

        self._video_file: str = opts.datasets_dir + "/" + opts.datasets + f"/videos/{video}.{VisualizerQt.VIDEO_EXTENSION}"
        self._window_size: int = opts.window_size
        self._fps: float = fps
        self._window_id = "video"

        self._predictions: VisualizerQt.PredictionList = list()
        self._true_actions: VisualizerQt.ActionList = list()
        self._labels: dict[int, str] = dict()
        self._capture: Optional[cv.VideoCapture] = None
        self._frame_id: int = 0
        self._playing: bool = False

        self._timer: QTimer = QTimer()
        self._timer.timeout.connect(self.timeout)
        interval = int(1000.0 / self._fps)
        self._timer.start(interval)

    def init(self, predictions: PredictionList, true_actions: ActionList, labels: dict[int, str]) -> None:
        assert len(predictions) == len(true_actions)
        self._predictions = predictions
        self._true_actions = true_actions
        self._labels = labels

        self._capture = cv.VideoCapture(self._video_file)
        if not self._capture.isOpened():
            # todo: Make this a proper exception.
            print("Cannot open ", self._video_file)
        else:
            self._main_window.show()
            self._main_window.center()
            self.restart()

    def timeout(self):
        if self._playing:
            success, frame = self._capture.read()
            if success:
                # Update the video image
                height, width, channels = frame.shape
                assert channels == 3
                image = QImage(frame.data, width, height, 3 * width, QImage.Format_BGR888)
                pixmap = QPixmap().fromImage(image)
                self._main_window.imageLabel.setPixmap(pixmap)
                self._main_window.imageLabel.setFixedSize(width, height)
                self._main_window.imageLabel.setMaximumSize(width, height)

                # Update action labels
                # The number of frames in the videos could be higher than the number of predictions
                # because we do not take the very last background action into consideration in the
                # ground truth actions itself. That's how the ground truth json file is generated.
                # But the clownfish, local and remote predict all the frames. So, not an issue.
                local_predictions, remote_predictions, fusion_predictions = zip(*self._predictions)
                if self._frame_id >= (self._window_size // 2) < len(local_predictions):
                    local_action = local_predictions[self._frame_id]
                    remote_action = remote_predictions[self._frame_id]
                    fusion_action = fusion_predictions[self._frame_id]
                    true_action = self._true_actions[self._frame_id]

                    self._main_window.localLabel.setText(self._labels[local_action])
                    self._main_window.remoteLabel.setText(self._labels[remote_action])
                    self._main_window.fusionLabel.setText(self._labels[fusion_action])
                    self._main_window.trueLabel.setText(self._labels[true_action])

                    local_css = VisualizerQt.CORRECT_CSS if local_action == true_action else VisualizerQt.WRONG_CSS
                    remote_css = VisualizerQt.CORRECT_CSS if remote_action == true_action else VisualizerQt.WRONG_CSS
                    fusion_css = VisualizerQt.CORRECT_CSS if fusion_action == true_action else VisualizerQt.WRONG_CSS

                    self._main_window.localLabel.setStyleSheet(local_css)
                    self._main_window.remoteLabel.setStyleSheet(remote_css)
                    self._main_window.fusionLabel.setStyleSheet(fusion_css)

                # Update window title
                self._main_window.setWindowTitle(f"Clownfish (frame {self._frame_id} - fps: {self._fps:.1f})")

                # Update frame index
                self._frame_id += 1
            else:
                self._playing = False

    def play(self):
        self._playing = True
        self._main_window.playButton.setText("Pause")

    def pause(self):
        self._playing = False
        self._main_window.playButton.setText("Play")

    def restart(self):
        self._frame_id = 0
        if self._capture.isOpened():
            self._capture.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
        self.play()

    def play_or_pause(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()
