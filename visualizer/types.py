"""
Defines types for actions and predictions.
"""
from typing import Tuple, List, Dict

# Define types for actions and predictions
Action = int
Prediction = Tuple[Action, Action, Action]

# Define list types
ActionList = List[Action]
PredictionList = List[Prediction]

# Define action/label dictionary
ActionLabels = Dict[Action, str]
