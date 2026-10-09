# Citrus Leaf Disease Classification

A deep learning project that classifies citrus leaf images into five
categories using seven pretrained computer vision architectures.

## Disease Classes

- Black spot
- Canker
- Greening
- Healthy
- Melanose

## Models Evaluated

1. ResNet34
2. EfficientNet-B0
3. Vision Transformer (ViT-B/16)
4. MobileNetV3-Large
5. ConvNeXt-Tiny
6. Swin Transformer-Tiny
7. MobileViT-Small

## Results

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| ResNet34 | 76.74% | 0.7500* | 0.7800* |
| EfficientNet-B0 | 88.37% | 0.8500* | 0.8800* |
| ViT-B/16 | 88.76% | 0.8554 | 0.8905 |
| MobileNetV3-Large | 77.52% | 0.7459 | 0.7849 |
| ConvNeXt-Tiny | 85.27% | 0.8269 | 0.8586 |
| Swin-Tiny | 85.27% | 0.8222 | 0.8566 |
| MobileViT-Small | 81.40% | 0.7731 | 0.8137 |

*ResNet34 and EfficientNet-B0 metrics are rounded historical values.

## Best Recorded Result

ViT-B/16 achieved the highest recorded test accuracy of 88.76%
among these experiments.

## Dataset

The recorded dataset contains 1,716 images.

| Class | Images |
|---|---:|
| Black spot | 238 |
| Canker | 237 |
| Greening | 220 |
| Healthy | 705 |
| Melanose | 316 |

The dataset is imbalanced, with Healthy being the largest class.

## Methodology

The experiments used pretrained image-classification architectures
with modified classification heads.

The recorded pipeline included image resizing and cropping, training
augmentation, ImageNet normalization, weighted cross-entropy loss
with label smoothing, and an adaptive learning-rate scheduler.

The summarized repository-based experiments used eight epochs.

## Important Experimental Limitation

ViT-B/16, MobileNetV3-Large, ConvNeXt-Tiny, Swin-Tiny, and
MobileViT-Small used the same saved stratified data split with seed 42.

ResNet34 and EfficientNet-B0 were trained in earlier experiments
using different splits. Their scores therefore do not form a
strictly controlled comparison with the other five models.

For a fair ranking, all seven architectures should be retrained
using the same train, validation, and test split.

## Repository Contents

- `notebooks/`: Notebook documentation and links.
- `reports/`: PDF and editable Word report.
- `results/`: Comparison tables and dataset information.
- `models/`: Model-specific evaluation artifacts.

## Model Checkpoints

The trained model checkpoints are not included in this repository
by default because they increase storage requirements.

See `models/README.md` for checkpoint storage and retrieval notes.

## Limitations

- The dataset is imbalanced.
- Some classes have lower recall than others.
- External field performance has not been established.
- Independent validation is required before real-world deployment.

## Disclaimer

This project is intended for academic experimentation and image
classification research. Predictions should not be treated as
professional agricultural diagnoses without independent validation.