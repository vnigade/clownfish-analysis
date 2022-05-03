"""
A Clownfish visualizer using Qt.
"""
import argparse
import cv2 as cv
import pathlib

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.QtChart import QChart, QChartView, QScatterSeries

from typing import Optional


class Chart:
    """
    A chart holds series and charts to display the performance of a specific category of predictions (local, remote, fusion).
    """

    # Define axis specifics
    SERIES_Y_VALUE: float = 0.0
    X_LOOK_BACK: float = 50.0
    X_LOOK_FORWARD: float = 10.0

    # Define colors for correct and wrong predictions
    TRUE_COLOR: QColor = Qt.green
    FALSE_COLOR: str = Qt.red

    def __init__(self, title: str):
        self.true_series: QScatterSeries = QScatterSeries()
        self.false_series: QScatterSeries = QScatterSeries()
        self.chart: QChart = QChart()
        self.view: QChartView = QChartView(self.chart)

        self.chart.addSeries(self.true_series)
        self.chart.addSeries(self.false_series)
        self.chart.setTheme(QChart.ChartThemeDark)
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.chart.setBackgroundRoundness(0)
        self.chart.legend().hide()
        self.chart.layout().setContentsMargins(0, 0, 0, 0)

        self.true_series.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
        self.false_series.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
        self.true_series.setColor(Chart.TRUE_COLOR)
        self.false_series.setColor(Chart.FALSE_COLOR)

        self.chart.createDefaultAxes()
        for axis in self.chart.axes(Qt.Horizontal):
            axis.setVisible(False)
            axis.setGridLineVisible(False)
            axis.setRange(0.0, 1000.0)
        for axis in self.chart.axes(Qt.Vertical):
            axis.setVisible(False)
            axis.setGridLineVisible(False)
            axis.setRange(-0.5, 0.5)

    def append(self, action: int, true_action: int, frame_id: int) -> None:
        series = self.true_series if action == true_action else self.false_series
        series.append(frame_id, Chart.SERIES_Y_VALUE)
        for axis in self.chart.axes(Qt.Horizontal):
            axis.setRange(frame_id - Chart.X_LOOK_BACK, frame_id + Chart.X_LOOK_FORWARD)

    def clear(self) -> None:
        self.true_series.clear()
        self.false_series.clear()


class MainWindow(QMainWindow):
    """
    The Qt main window of the visualizer.
    """

    # Define coloring stylesheets for correct and wrong predictions
    TRUE_CSS: str = "color: green;"
    FALSE_CSS: str = "color: red;"

    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)
        ui_file = pathlib.Path(__file__).with_name("qt_visualizer.ui")
        loadUi(ui_file, self)

        # Setup central layout
        self.centralWidget.layout().setContentsMargins(9, 9, 9, 9)
        self.centralWidget.layout().setSpacing(6)

        # Setup chart series and views
        self.local_chart: Chart = Chart("Local")
        self.fusion_chart: Chart = Chart("Clownfish")
        self.remote_chart: Chart = Chart("Remote")

        # Insert charts into the layout
        local_charts_layout = QHBoxLayout()
        local_charts_layout.addWidget(self.local_chart.view)
        local_charts_layout.setContentsMargins(0, 0, 0, 0)
        self.localChartWidget.setLayout(local_charts_layout)

        fusion_charts_layout = QHBoxLayout()
        fusion_charts_layout.addWidget(self.fusion_chart.view)
        fusion_charts_layout.setContentsMargins(0, 0, 0, 0)
        self.fusionChartWidget.setLayout(fusion_charts_layout)

        remote_charts_layout = QHBoxLayout()
        remote_charts_layout.addWidget(self.remote_chart.view)
        remote_charts_layout.setContentsMargins(0, 0, 0, 0)
        self.remoteChartWidget.setLayout(remote_charts_layout)

    def center(self) -> None:
        geometry = self.frameGeometry()
        cursor_position = QApplication.desktop().cursor().pos()
        screen = QApplication.desktop().screenNumber(cursor_position)
        center_point = QApplication.desktop().screenGeometry(screen).center()
        geometry.moveCenter(center_point)
        self.move(geometry.topLeft())

    def resize_labels_to_required_size(self, labels: list[str]) -> None:
        # Get minimum size required for displaying the largest label
        # todo: This assumes that the font is the same for all action labels
        font = self.localActionLabel.font()
        font_metrics = QFontMetrics(font)
        sizes = [font_metrics.boundingRect(label).width() for label in labels]
        required_size = max(sizes)

        # Set the minimum size
        print(required_size)
        self.localActionLabel.setMinimumWidth(required_size)
        self.fusionActionLabel.setMinimumWidth(required_size)
        self.remoteActionLabel.setMinimumWidth(required_size)

    def set_frame_count(self, frame_count: int) -> None:
        self.frameScrollbar.setMaximum(frame_count - 1)

    def set_frame_index(self, frame_index: int) -> None:
        self.frameScrollbar.setSliderPosition(frame_index)

    def set_frame_image(self, frame: np.ndarray) -> None:
        height, width, channels = frame.shape
        assert channels == 3
        image = QImage(frame.data, width, height, 3 * width, QImage.Format_BGR888)
        label_width, label_height = self.imageLabel.width(), self.imageLabel.height()
        pixmap = QPixmap().fromImage(image)
        pixmap = pixmap.scaled(label_width - 2, label_height - 2, Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setMaximumSize(width, height)

    def set_true_action(self, action: str) -> None:
        self.trueLabel.setText(action)

    def set_local_prediction(self, action: str, percentage: float, correct: bool) -> None:
        self._update_labels(self.localActionLabel, self.localPercentageLabel, action, percentage, correct)

    def set_fusion_prediction(self, action: str, percentage: float, correct: bool) -> None:
        self._update_labels(self.fusionActionLabel, self.fusionPercentageLabel, action, percentage, correct)

    def set_remote_prediction(self, action: str, percentage: float, correct: bool) -> None:
        self._update_labels(self.remoteActionLabel, self.remotePercentageLabel, action, percentage, correct)

    @staticmethod
    def _update_labels(action_label: QLabel, percentage_label: QLabel, action: str, percentage: float, correct: bool) -> None:
        css = MainWindow.TRUE_CSS if correct else MainWindow.FALSE_CSS
        action_label.setText(action)
        percentage_label.setText(f"{percentage:.1f}%")
        action_label.setStyleSheet(css)
        percentage_label.setStyleSheet(css)


class VisualizerQt:
    """
    A Qt-based visualizer.
    """

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

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, opts: argparse.Namespace, video: str, fps: float = 30.0, centering: bool = False, *args):
        # Setup main window
        self._main_window = MainWindow(*args)
        self._main_window.playButton.clicked.connect(self._play_or_pause)
        self._main_window.restartButton.clicked.connect(self._restart)

        # Setup members
        self._video_file: str = opts.datasets_dir + "/" + opts.datasets + f"/videos/{video}.{VisualizerQt.VIDEO_EXTENSION}"
        self._window_size: int = opts.window_size
        self._fps: float = fps

        self._predictions: VisualizerQt.PredictionList = list()
        self._true_actions: VisualizerQt.ActionList = list()
        self._labels: dict[int, str] = dict()
        self._capture: Optional[cv.VideoCapture] = None
        self._playing: bool = False
        self._frame_id: int = 0
        self._evaluated_frames_count: int = 0
        self._correctness_counter: list[int] = [0, 0, 0]

        # Load video capture and extract meta information
        self._capture: cv.VideoCapture = cv.VideoCapture(self._video_file)
        self._frame_count: int = 0
        if self._capture.isOpened():
            self._frame_count = int(self._capture.get(cv.CAP_PROP_FRAME_COUNT))
            self._main_window.set_frame_count(self._frame_count)
        else:
            # todo: Make this a proper exception.
            print("Cannot open ", self._video_file)

        # Setup and start frame timer
        self._timer: QTimer = QTimer()
        self._timer.timeout.connect(self._timeout)
        interval = int(1000.0 / self._fps)
        self._timer.start(interval)

        # Show the main window
        self._main_window.show()
        if centering:
            self._main_window.center()

    def start(self, predictions: PredictionList, true_actions: ActionList, labels: dict[int, str]) -> None:
        """
        Starts the playing of the video.

        Call this method before entering the main loop of the Qt application.

        :param predictions: A tuple with lists of predictions for the local, remote, and fusion model.
        :param true_actions: A list with the ground truth actions.
        :param labels: A dictionary mapping action classes to text labels.
        :return: None.
        """

        assert len(predictions) == len(true_actions)
        self._predictions = predictions
        self._true_actions = true_actions
        self._labels = labels
        self._main_window.resize_labels_to_required_size(labels.values())

        if self._capture.isOpened():
            self._restart()
        else:
            # todo: Make this a proper exception.
            print("Could not start (video not properly loaded).")

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    def _timeout(self) -> None:
        if self._playing:
            success, frame = self._capture.read()
            if success:
                # Update the video image
                self._main_window.set_frame_image(frame)

                # Update action labels
                # The number of frames in the videos could be higher than the number of predictions
                # because we do not take the very last background action into consideration in the
                # ground truth actions itself. That's how the ground truth json file is generated.
                # But the clownfish, local and remote predict all the frames. So, not an issue.
                local_predictions, remote_predictions, fusion_predictions = zip(*self._predictions)
                if (self._window_size // 2) <= self._frame_id < len(local_predictions):
                    true_action = self._true_actions[self._frame_id]
                    local_action = local_predictions[self._frame_id]
                    fusion_action = fusion_predictions[self._frame_id]
                    remote_action = remote_predictions[self._frame_id]

                    self._evaluated_frames_count += 1
                    self._correctness_counter[0] += 1 if local_action == true_action else 0
                    self._correctness_counter[1] += 1 if fusion_action == true_action else 0
                    self._correctness_counter[2] += 1 if remote_action == true_action else 0
                    correctness_percentages = [100.0 * float(counter) / float(self._evaluated_frames_count) for counter in self._correctness_counter]

                    # Update prediction widgets
                    self._main_window.set_true_action(self._labels[true_action])
                    self._main_window.set_local_prediction(self._labels[local_action], correctness_percentages[0], local_action == true_action)
                    self._main_window.set_fusion_prediction(self._labels[fusion_action], correctness_percentages[1], fusion_action == true_action)
                    self._main_window.set_remote_prediction(self._labels[remote_action], correctness_percentages[2], remote_action == true_action)

                    self._main_window.local_chart.append(local_action, true_action, self._frame_id)
                    self._main_window.fusion_chart.append(fusion_action, true_action, self._frame_id)
                    self._main_window.remote_chart.append(remote_action, true_action, self._frame_id)

                # Update window title
                self._main_window.setWindowTitle(f"Clownfish (frame {self._frame_id} / {self._frame_count} - fps: {self._fps:.1f})")

                # Update frame index
                self._next_frame()
            else:
                # todo: add automatic restart?
                self._playing = False

    def _play(self) -> None:
        self._playing = True
        self._main_window.playButton.setText("Pause")

    def _pause(self) -> None:
        self._playing = False
        self._main_window.playButton.setText("Play")

    def _restart(self) -> None:
        self._frame_id = 0
        self._main_window.set_frame_index(self._frame_id)

        self._evaluated_frames_count = 0
        self._correctness_counter = [0, 0, 0]
        self._main_window.local_chart.clear()
        self._main_window.fusion_chart.clear()
        self._main_window.remote_chart.clear()

        if self._capture.isOpened():
            self._capture.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
        self._play()

    def _play_or_pause(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()

    def _next_frame(self) -> None:
        self._frame_id += 1
        self._main_window.set_frame_index(self._frame_id)
