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
from PyQt5.QtChart import QChart, QChartView, QScatterSeries

from typing import Optional


class Chart:
    """
    A chart holds series and charts to display the performance of a specific category of predictions (local, remote, fusion).
    """

    # Define axis specifics
    SERIES_Y_VALUE: float = 0.5
    X_LOOK_BACK: float = 50.0
    X_LOOK_FORWARD: float = 10.0

    # Define coloring stylesheets for correct and wrong predictions
    TRUE_COLOR: QColor = Qt.green
    FALSE_COLOR: str = Qt.red

    def __init__(self, title: str):
        self.true_series: QScatterSeries = QScatterSeries()
        self.false_series: QScatterSeries = QScatterSeries()
        self.chart: QChart = QChart()
        self.view: QChartView = QChartView(self.chart)

        self.chart.setTitle(title)
        self.chart.addSeries(self.true_series)
        self.chart.addSeries(self.false_series)
        self.chart.setTheme(QChart.ChartThemeDark)
        self.chart.legend().hide()

        self.true_series.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
        self.false_series.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
        self.true_series.setColor(Chart.TRUE_COLOR)
        self.false_series.setColor(Chart.FALSE_COLOR)

        self.chart.createDefaultAxes()
        for axis in self.chart.axes(Qt.Horizontal):
            axis.setGridLineVisible(False)
            axis.setRange(0.0, 1000.0)
        for axis in self.chart.axes(Qt.Vertical):
            axis.setGridLineVisible(False)
            axis.setRange(0.3, 0.7)

    def append(self, action: int, true_action: int, frame_id: int):
        series = self.true_series if action == true_action else self.false_series
        series.append(frame_id, Chart.SERIES_Y_VALUE)
        for axis in self.chart.axes(Qt.Horizontal):
            axis.setRange(frame_id - Chart.X_LOOK_BACK, frame_id + Chart.X_LOOK_FORWARD)


class MainWindow(QMainWindow):
    """
    The Qt main window of the visualizer.
    """

    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)
        ui_file = pathlib.Path(__file__).with_name("qt_visualizer.ui")
        loadUi(ui_file, self)

        # Setup central layout
        self.centralWidget.layout().setContentsMargins(9, 9, 9, 9)
        self.centralWidget.layout().setSpacing(6)

        # Setup image widget layout
        self.imageWidget.layout().setAlignment(Qt.AlignHCenter)

        # Setup chart series and views
        self.local_chart: Chart = Chart("Local")
        self.remote_chart: Chart = Chart("Remote")
        self.fusion_chart: Chart = Chart("Clownfish")

        charts_layout = QHBoxLayout()
        charts_layout.addWidget(self.local_chart.view)
        charts_layout.addWidget(self.remote_chart.view)
        charts_layout.addWidget(self.fusion_chart.view)
        self.chartsWidget.setLayout(charts_layout)

    def center(self):
        geometry = self.frameGeometry()
        cursor_position = QApplication.desktop().cursor().pos()
        screen = QApplication.desktop().screenNumber(cursor_position)
        center_point = QApplication.desktop().screenGeometry(screen).center()
        geometry.moveCenter(center_point)
        self.move(geometry.topLeft())


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

    def __init__(self, opts: argparse.Namespace, video: str, fps: float = 30.0, *args):
        self._main_window = MainWindow(*args)
        self._main_window.show()
        self._main_window.playButton.clicked.connect(self._play_or_pause)
        self._main_window.restartButton.clicked.connect(self._restart)

        self._video_file: str = opts.datasets_dir + "/" + opts.datasets + f"/videos/{video}.{VisualizerQt.VIDEO_EXTENSION}"
        self._window_size: int = opts.window_size
        self._fps: float = fps
        self._window_id = "video"

        self._predictions: VisualizerQt.PredictionList = list()
        self._true_actions: VisualizerQt.ActionList = list()
        self._labels: dict[int, str] = dict()
        self._capture: Optional[cv.VideoCapture] = None
        self._playing: bool = False
        self._frame_id: int = 0
        self._evaluated_frames_count: int = 0
        self._correctness_counter: list[int] = [0, 0, 0]

        self._timer: QTimer = QTimer()
        self._timer.timeout.connect(self._timeout)
        interval = int(1000.0 / self._fps)
        self._timer.start(interval)

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

        self._capture = cv.VideoCapture(self._video_file)
        if not self._capture.isOpened():
            # todo: Make this a proper exception.
            print("Cannot open ", self._video_file)
        else:
            self._main_window.show()
            self._main_window.center()
            self._restart()

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    def _timeout(self) -> None:
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

                    self._evaluated_frames_count += 1
                    self._correctness_counter[0] += 1 if local_action == true_action else 0
                    self._correctness_counter[1] += 1 if remote_action == true_action else 0
                    self._correctness_counter[2] += 1 if fusion_action == true_action else 0
                    correctness_percentages = [100.0 * float(counter) / float(self._evaluated_frames_count) for counter in self._correctness_counter]

                    local_label = self._labels[local_action]
                    remote_label = self._labels[remote_action]
                    fusion_label = self._labels[fusion_action]
                    true_label = self._labels[true_action]

                    local_css = VisualizerQt.CORRECT_CSS if local_action == true_action else VisualizerQt.WRONG_CSS
                    remote_css = VisualizerQt.CORRECT_CSS if remote_action == true_action else VisualizerQt.WRONG_CSS
                    fusion_css = VisualizerQt.CORRECT_CSS if fusion_action == true_action else VisualizerQt.WRONG_CSS

                    # Update widgets for version A
                    self._main_window.localLabel.setText(f"{local_label} ({correctness_percentages[0]:.1f}%)")
                    self._main_window.remoteLabel.setText(f"{remote_label} ({correctness_percentages[1]:.1f}%)")
                    self._main_window.fusionLabel.setText(f"{fusion_label} ({correctness_percentages[2]:.1f}%)")
                    self._main_window.trueLabel.setText(f"{true_label}")

                    self._main_window.localLabel.setStyleSheet(local_css)
                    self._main_window.remoteLabel.setStyleSheet(remote_css)
                    self._main_window.fusionLabel.setStyleSheet(fusion_css)

                    # Update widgets for version B
                    self._main_window.trueLabelB.setText(f"{true_label}")
                    self._main_window.local_chart.append(local_action, true_action, self._frame_id)
                    self._main_window.remote_chart.append(remote_action, true_action, self._frame_id)
                    self._main_window.fusion_chart.append(fusion_action, true_action, self._frame_id)

                # Update window title
                self._main_window.setWindowTitle(f"Clownfish (frame {self._frame_id} - fps: {self._fps:.1f})")

                # Update frame index
                self._frame_id += 1
            else:
                self._playing = False

    def _play(self) -> None:
        self._playing = True
        self._main_window.playButton.setText("Pause")

    def _pause(self) -> None:
        self._playing = False
        self._main_window.playButton.setText("Play")

    def _restart(self) -> None:
        self._frame_id = 0
        self._evaluated_frames_count = 0
        self._correctness_counter = [0, 0, 0]

        if self._capture.isOpened():
            self._capture.set(cv.CAP_PROP_POS_AVI_RATIO, 0)
        self._play()

    def _play_or_pause(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()
