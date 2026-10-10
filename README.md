# AI Image Detector

## About
AI Image Detector is a deep learning project that aims to classify images as real or AI-generated using a Convolutional Neural Network (CNN).

## Current Progress
- [x] Downloaded the Tiny-GenImage dataset
- [x] Built a PyTorch dataset loader
- [x] Verified image batches and tensor shapes
- [x] Implemented a baseline CNN
- [x] Trained the initial model
- [ ] Improve model performance
- [ ] Evaluate precision, recall, and F1-score
- [ ] Test on images from additional sources

## Technologies
- Python
- PyTorch
- Polars
- Pillow (PIL)

## Dataset
The project uses the Tiny-GenImage dataset. Dataset files are stored locally and are not included in this repository.

## Current Baseline
The initial CNN achieved 61.67% test accuracy on a 300-image test split from the first dataset shard. This is an early experimental result, not a measure of general performance.

## Project Structure
```text
AI-Image-Detector/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── test_dataloader.py
├── .gitignore
└── README.md
```

## Goal
Develop and evaluate a model that can distinguish real images from AI-generated images, including images from generators not seen during training.