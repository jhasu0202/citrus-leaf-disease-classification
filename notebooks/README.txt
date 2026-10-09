
CITRUS LEAF DISEASE CLASSIFICATION
==================================

Generated: 2026-10-09 12:20:59

PROJECT
-------
Image classification of five citrus leaf classes:
- Black spot
- Canker
- Greening
- Healthy
- Melanose

DATASET
-------
The supplied dataset contains 1,716 images according to the
recorded class counts:

     Class  Image Count  Percentage (%)
Black spot          238           13.87
    Canker          237           13.81
  Greening          220           12.82
   Healthy          705           41.08
  Melanose          316           18.41

No external dataset has been added to this package.

MODELS
------
1. ResNet34
2. EfficientNet-B0
3. Vision Transformer Base/16 (ViT-B/16)
4. MobileNetV3-Large
5. ConvNeXt-Tiny
6. Swin Transformer-Tiny
7. MobileViT-Small

RESULTS
-------
See model_comparison.csv and model_comparison.xlsx for the consolidated
performance table.

IMPORTANT COMPARISON LIMITATION
-------------------------------
ResNet34 and EfficientNet-B0 were trained in earlier experiments that
used different data splits from the later shared-split experiments.
Their numbers should not be interpreted as a strictly controlled,
head-to-head comparison with the other five models.

ViT-B/16, MobileNetV3-Large, ConvNeXt-Tiny, Swin-Tiny, and
MobileViT-Small reused the saved stratified split with seed 42.

TRAINING
--------
The recorded repository-based experiments generally used pretrained
backbones, frozen feature extractors, a replacement classification head,
weighted cross-entropy loss with label smoothing, and eight epochs.
See each model's original result files and model_metadata.json.

ARTIFACTS
---------
models/
    Each model's available checkpoint and evaluation artifacts.

model_comparison.csv
model_comparison.xlsx
dataset_class_distribution.csv
shared_data_split.csv (when available)
Citrus_Leaf_Disease_Classification_Report.pdf
Citrus_Leaf_Disease_Classification_Report.docx

MISSING HISTORICAL ARTIFACTS
----------------------------
The earlier ResNet34 and EfficientNet-B0 result folders did not contain
saved training histories or original confusion-matrix image files.
Their confusion matrices in this package were reconstructed from the
previously recorded count matrices. They are labelled reconstructed.

REPRODUCIBILITY AND INTERPRETATION
----------------------------------
The reported scores reflect the recorded experiments. The test scores
are not a guarantee of performance on new orchards, devices, lighting
conditions, or unseen field data.

For a rigorous final comparison, retrain every model with the same
train/validation/test split and evaluation pipeline. Consider repeated
seeds, per-class metrics, and external field validation.

CHECKPOINT LOADING
------------------
Model checkpoint loading depends on the original model definition and
the installed versions of PyTorch, torchvision, and/or timm.
Review the reference repository and the original training code before
loading a checkpoint.

END
