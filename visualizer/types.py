"""
Defines types for actions and predictions.
"""

# Define types for actions and predictions
Action = int
Prediction = tuple[Action, Action, Action]

# Define list types
ActionList = list[Action]
PredictionList = list[Prediction]

# Define action/label dictionary
ActionLabels = dict[Action, str]
