"""
A Clownfish visualizer that displays a video stream with predicted and ground truth actions.
"""
import argparse
import numpy as np
import sys

from PyQt5.QtWidgets import QApplication
from typing import Callable, Optional, Union

from .opencv_visualizer import VisualizerOpenCV
from .qt_visualizer import VisualizerQt


class Visualizer:
    """
    The Visualizer is the main window class for the frontend.

    Its main purpose is to provide a unified interface for different visualizer backends (Qt or OpenCV-based).
    """

    # Define Action and ActionList types
    Action = int
    ActionList = list[Action]

    # =================================================================================================================================================================================================
    # Public interface
    # =================================================================================================================================================================================================

    def __init__(self, opts: argparse.Namespace, video: str, fps: float = 30.0, use_qt: bool = True, console_arguments: Optional[list[str]] = None):
        self._visualizer: Union[VisualizerOpenCV, VisualizerQt]
        self._qt_application: Optional[QApplication] = None
        self._default_exception_hook: Callable = sys.excepthook

        if use_qt:
            # Qt applications present an issue that Python exceptions are swallowed.
            # The problem is fixed here by using a custom exception handler as follows:
            #   1. Back up the reference to the default system exception hook
            #   2. Set a custom exception hook that simply forwards to the system exception hook
            #   3. Restore the exception hook after the Qt application is finished
            sys.excepthook = self._custom_exception_hook
            self._application = QApplication(list() if not console_arguments else console_arguments)
            self._visualizer = VisualizerQt(opts, video, fps)
        else:
            self._visualizer = VisualizerOpenCV(opts, video, fps)

    def __del__(self):
        sys.excepthook = self._default_exception_hook

    def display(self, predictions: list[tuple[Action, Action, Action]], true_actions: ActionList, labels: dict[int, str]) -> int:
        """
        Displays a video with the annotations.

        :param predictions: A tuple with lists of predictions for the local, remote, and fusion model.
        :param true_actions: A list with the ground truth actions.
        :param labels: A dictionary mapping action classes to text labels.
        :return: Returns the exit code of the visualizer (should be passed to sys.exit).
        """

        assert len(predictions) == len(true_actions)

        return_code = 0
        if isinstance(self._visualizer, VisualizerQt):
            self._visualizer.start(predictions, true_actions, labels)
            return_code = self._application.exec_()
        else:
            self._visualizer.display(predictions, true_actions, labels)

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
