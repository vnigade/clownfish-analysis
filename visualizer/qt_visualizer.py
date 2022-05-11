"""
A Clownfish visualizer using Qt.
"""
import cv2 as cv
import pathlib

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.QtChart import QChart, QChartView, QScatterSeries

from .types import ActionList, ActionLabels, PredictionList


class Chart:
    """
    A chart holds series and charts to display the performance of a specific category of predictions (local, remote, fusion).
    """

    # Define axis specifics
    SERIES_Y_VALUE: float = 0.0
    X_LOOK_BACK: float = 50.0
    X_LOOK_FORWARD: float = 10.0
    MAXIMUM_CHART_HEIGHT: float = 40.0

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
        self.chart.setMaximumHeight(Chart.MAXIMUM_CHART_HEIGHT)

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
    TRUE_CSS: str = "background-color: green;"
    FALSE_CSS: str = "background-color: red;"

    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)
        ui_file = pathlib.Path(__file__).with_name("qt_visualizer.ui")
        loadUi(ui_file, self)

        # Setup central layout
        self.centralWidget.layout().setContentsMargins(9, 9, 9, 9)
        self.centralWidget.layout().setSpacing(6)

        # Hide details on first show and setup details button
        self.detailsWidget.hide()
        self.detailsButton.clicked.connect(self._toggle_details)

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

    def resize_labels_to_required_size(self, labels: list[str]) -> None:
        # Get minimum size required for displaying the largest label
        # todo: This assumes that the font is the same for all action labels
        font = self.localActionLabel.font()
        font_metrics = QFontMetrics(font)
        sizes = [font_metrics.boundingRect(label).width() for label in labels]
        required_size = max(sizes)

        # Set the minimum size
        self.localActionLabel.setMinimumWidth(required_size)
        self.fusionActionLabel.setMinimumWidth(required_size)
        self.remoteActionLabel.setMinimumWidth(required_size)

    def set_frame_count(self, frame_count: int) -> None:
        self.frameScrollbar.setMaximum(frame_count - 1)

    def set_frame_index(self, frame_index: int) -> None:
        if not self.frameScrollbar.isSliderDown():
            self.frameScrollbar.setSliderPosition(frame_index)

    def set_frame_image(self, frame: np.ndarray) -> None:
        if self.isVisible():
            height, width, channels = frame.shape
            assert channels == 3
            image = QImage(frame.data, width, height, channels * width, QImage.Format_BGR888)
            label_width, label_height = self.imageLabel.width(), self.imageLabel.height()
            frame_width = self.imageLabel.frameWidth()
            pixmap = QPixmap().fromImage(image)
            pixmap = pixmap.scaled(label_width - 2 * frame_width, label_height - 2 * frame_width, Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setMinimumSize(1, 1)

    def set_true_action(self, action: str) -> None:
        self.trueLabel.setText(action)

    def set_main_prediction(self, action: str, correct: bool) -> None:
        self.predictionLabel.setText(action)
        self.predictionLabel.setStyleSheet(MainWindow.TRUE_CSS if correct else MainWindow.FALSE_CSS)

    def set_local_prediction(self, action: str, percentage: float, correct: bool) -> None:
        self._update_labels(self.localActionLabel, self.localPercentageLabel, action, percentage, correct)

    def set_fusion_prediction(self, action: str, percentage: float, correct: bool) -> None:
        self._update_labels(self.fusionActionLabel, self.fusionPercentageLabel, action, percentage, correct)

    def set_remote_prediction(self, action: str, percentage: float, correct: bool) -> None:
        self._update_labels(self.remoteActionLabel, self.remotePercentageLabel, action, percentage, correct)

    def clear_charts(self) -> None:
        self.local_chart.clear()
        self.fusion_chart.clear()
        self.remote_chart.clear()

    @staticmethod
    def _update_labels(action_label: QLabel, percentage_label: QLabel, action: str, percentage: float, correct: bool) -> None:
        action_label.setText(action)
        action_label.setStyleSheet(MainWindow.TRUE_CSS if correct else MainWindow.FALSE_CSS)
        percentage_label.setText(f"{percentage:.1f}%")

    def _toggle_details(self, visible: bool):
        self.detailsWidget.setVisible(visible)
        self.detailsButton.setText("Hide Details" if visible else "Show Details")


class VisualizerQt:
    """
    A Qt-based visualizer.
    """

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, video: cv.VideoCapture, window_size: int, predictions: PredictionList, true_actions: ActionList, label_dict: ActionLabels, target_fps: float = 30.0, *args):
        assert video.isOpened()
        self._video: cv.VideoCapture = video
        self._window_size: int = window_size
        self._predictions: PredictionList = predictions
        self._true_actions: ActionList = true_actions
        self._label_dict: ActionLabels = label_dict
        self._target_fps: float = target_fps

        # Precompute the prediction performance (in correct frames percentage) for each prediction model
        local_predictions, remote_predictions, fusion_predictions = zip(*predictions)
        self._local_correct_frame_counts: list[float] = self._compute_correct_frame_percentages(local_predictions, true_actions)
        self._remote_correct_frame_counts: list[float] = self._compute_correct_frame_percentages(remote_predictions, true_actions)
        self._fusion_correct_frame_counts: list[float] = self._compute_correct_frame_percentages(fusion_predictions, true_actions)

        # Extract video meta information
        self._frame_count: int = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        # Initialize playing state
        self._playing: bool = False
        self._frame_id: int = 0

        # Setup main window
        self._main_window = MainWindow(*args)
        self._main_window.playButton.clicked.connect(self._play_or_pause)
        self._main_window.restartButton.clicked.connect(self._restart)
        self._main_window.frameScrollbar.valueChanged.connect(self._jump_to_frame)
        self._main_window.set_frame_count(self._frame_count)
        self._main_window.resize_labels_to_required_size(list(label_dict.values()))

        # Setup and start frame timer
        self._timer: QTimer = QTimer()
        self._timer.timeout.connect(self._timeout)
        interval = int(1000.0 / self._target_fps)
        self._timer.start(interval)

    def start(self) -> None:
        """
        Starts the playing of the video.

        Call this method before entering the main loop of the Qt application.

        :return: None.
        """

        # Show the main window
        self._main_window.show()

        # Start playing the video
        assert self._video.isOpened()
        self._restart()

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    def _compute_correct_frame_percentages(self, predictions: ActionList, true_actions: ActionList) -> list[float]:
        offset = self._window_size // 2
        correct = np.array(predictions) == np.array(true_actions)
        correct = correct[offset:]
        counts = np.cumsum(correct.astype(np.int))
        percentages = np.zeros(len(predictions), dtype=np.float32)
        percentages[offset:] = 100.0 * counts.astype(np.float32) / (np.arange(len(counts), dtype=np.float32) + 1)
        return percentages.tolist()

    def _timeout(self) -> None:
        if self._playing:
            assert self._video.isOpened()
            success, frame = self._video.read()
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

                    # Update prediction widgets
                    self._main_window.set_true_action(self._label_dict[true_action])
                    self._main_window.set_main_prediction(self._label_dict[fusion_action], fusion_action == true_action)
                    self._main_window.set_local_prediction(self._label_dict[local_action], self._local_correct_frame_counts[self._frame_id], local_action == true_action)
                    self._main_window.set_fusion_prediction(self._label_dict[fusion_action], self._fusion_correct_frame_counts[self._frame_id], fusion_action == true_action)
                    self._main_window.set_remote_prediction(self._label_dict[remote_action], self._remote_correct_frame_counts[self._frame_id], remote_action == true_action)

                    self._main_window.local_chart.append(local_action, true_action, self._frame_id)
                    self._main_window.fusion_chart.append(fusion_action, true_action, self._frame_id)
                    self._main_window.remote_chart.append(remote_action, true_action, self._frame_id)

                # Update window title
                self._main_window.setWindowTitle(f"Clownfish (frame {self._frame_id} / {self._frame_count} - fps: {self._target_fps:.1f})")

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
        self._main_window.clear_charts()

        assert self._video.isOpened()
        self._video.set(cv.CAP_PROP_POS_FRAMES, 0)
        self._play()

    def _play_or_pause(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()

    def _next_frame(self) -> None:
        self._frame_id += 1
        self._main_window.set_frame_index(self._frame_id)

    def _jump_to_frame(self, frame: int) -> None:
        assert 0 <= frame <= self._frame_count
        self._frame_id = frame
        self._main_window.set_frame_index(self._frame_id)
        # todo: update the charts properly
        self._main_window.clear_charts()

        assert self._video.isOpened()
        self._video.set(cv.CAP_PROP_POS_FRAMES, frame)

        if not self._playing:
            self._play()
