"""
A Clownfish visualizer that displays a video stream with predicted and ground truth actions.
"""
import cv2 as cv
import sys

from PyQt5.QtWidgets import QApplication
from typing import Callable, Optional, Union

from .opencv_visualizer import VisualizerOpenCV
from .qt_visualizer import VisualizerQt
from .types import ActionList, ActionLabels, PredictionList


class Visualizer:
    """
    The Visualizer is the main window class for the frontend.

    Its main purpose is to provide a unified interface for different visualizer backends (Qt or OpenCV-based).
    """

    class VideoError(Exception):
        """
        Exception raised when the video capture could not be opened.
        """

        def __init__(self, filename: str):
            self.filename = filename

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, video_file: str, window_size: int, predictions: PredictionList, true_actions: ActionList, label_dict: ActionLabels, target_fps: int = 30, use_qt: bool = True, console_arguments: Optional[list[str]] = None):
        # Define and initialize members
        self._visualizer: Union[VisualizerOpenCV, VisualizerQt]
        self._qt_application: Optional[QApplication] = None
        self._default_exception_hook: Callable = sys.excepthook

        # Open video capture
        self._video: cv.VideoCapture = cv.VideoCapture(video_file)
        if not self._video.isOpened():
            raise Visualizer.VideoError(video_file)

        # Create visualizer backend
        if use_qt:
            # Qt applications present an issue that Python exceptions are swallowed.
            # The problem is fixed here by using a custom exception handler as follows:
            #   1. Back up the reference to the default system exception hook
            #   2. Set a custom exception hook that simply forwards to the system exception hook
            #   3. Restore the exception hook after the Qt application is finished
            sys.excepthook = self._custom_exception_hook
            self._application = QApplication(list() if not console_arguments else console_arguments)
            self._visualizer = VisualizerQt(self._video, window_size, predictions, true_actions, label_dict, target_fps)
        else:
            self._visualizer = VisualizerOpenCV(self._video, window_size, predictions, true_actions, label_dict, target_fps)

    def __del__(self):
        sys.excepthook = self._default_exception_hook

    def display(self) -> int:
        """
        Displays the video.

        :return: Returns the exit code of the visualizer (should be passed to sys.exit).
        """

        # Start showing the video with the visualizer backend
        return_code = 0
        if isinstance(self._visualizer, VisualizerQt):
            self._visualizer.start()
            return_code = self._application.exec_()
        else:
            self._visualizer.start()

        # Release the video capture
        self._video.release()

        return return_code

    # =================================================================================================================================================================================================
    # Private methods
    # =================================================================================================================================================================================================

    def _custom_exception_hook(self, exception_type, value, traceback):
        # Print the error and traceback
        print(exception_type, value, traceback)
        # Call the normal exception hook after
        self._default_exception_hook(exception_type, value, traceback)
        sys.exit(1)
