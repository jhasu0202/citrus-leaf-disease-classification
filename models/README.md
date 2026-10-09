# Model Evaluation Results

Each model directory contains the evaluation artifacts available
from its experiment.

These may include:
- Classification report
- Confusion matrix
- Training history
- Test predictions
- Class mapping
- Model metadata

The large trained checkpoints are stored separately and are not
included in this repository by default.

ResNet34 and EfficientNet-B0 have reconstructed confusion matrices
based on previously recorded counts. Their original training-history
files were unavailable in the saved result folders.