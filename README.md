# Tick Photo Identification Using Deep Learning

A deep-learning pipeline for hierarchical identification of North American ixodid tick species from dorsal-view photographs, using transfer learning with InceptionV3.

## Overview

Rapid, accurate identification of ticks is essential for understanding the changing geographic distributions of these important disease vectors. Traditional dichotomous keys can be complex, require considerable training, and are open to human error. This repository provides the CNN-based classification system described in:

> **Potential of dorsal-view images in hierarchical automated tick identification**
> Ali Khalighifar, Amber Grant, Kellee Sundstrom, Kathryn Duncan, A. Townsend Peterson

The system classifies dorsal-view images of seven common North American hard tick species across life stages (larva, nymph, adult), species, and sex using a hierarchical framework:

| Species | Larva | Nymph | Adult |
| ------- | :---: | :---: | :---: |
| *Amblyomma americanum* | x | x | x |
| *Amblyomma maculatum* | x | x | x |
| *Dermacentor andersoni* | -- | x | x |
| *Dermacentor variabilis* | x | x | x |
| *Haemaphysalis longicornis* | x | x | x |
| *Ixodes scapularis* | x | x | x |
| *Rhipicephalus sanguineus* | x | x | x |

### Approach

- **Transfer learning** with [InceptionV3](https://arxiv.org/abs/1512.00567) pre-trained on ImageNet
- **Data augmentation** (random horizontal flips and rotations) to improve generalization
- **Fine-tuning** of upper convolutional layers after initial training with a reduced learning rate
- **Callbacks** for early stopping, learning-rate reduction, and model checkpointing

### Classification Tasks and Performance

The system was evaluated across five classification tasks:

| Task | Description | Classes | Accuracy |
| ---- | ----------- | :-----: | :------: |
| 1 | Life-stage classification (larva / nymph / adult) | 3 | 99.8% |
| 2 | Larval species classification | 6 | 88.3% |
| 3 | Nymph species classification | 7 | 88.6% |
| 4 | Adult species + sex classification | 13 | 94.6% |
| 5 | Hierarchical classification (life stage + species + sex) | -- | 92.6% |

## Repository Structure

```
tick-photo-ID/
├── train_tick_classifier.py   # Main training and prediction script
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── CITATION.cff         # Citation metadata
└── README.md            # This file
```

## Requirements

- Python 3.8 -- 3.10
- TensorFlow 2.8
- See [requirements.txt](requirements.txt) for the full dependency list

### Installation

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare the Directory Structure

Organize your tick images into the following layout. Each subfolder under `train/` and `test/` should be named after the tick species (or species--sex--stage class) it contains:

```
base_directory/
├── train/
│   ├── species_A/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── species_B/
│   │   └── ...
│   └── ...
├── test/
│   ├── species_A/
│   │   └── ...
│   ├── species_B/
│   │   └── ...
│   └── ...
└── checkpoints/
```

A 20% validation split is drawn automatically from the training set during model fitting.

### 2. Configure Paths

Open `train_tick_classifier.py` and set the following variables to match your directory layout:

```python
BASE_DIR = "Path to Base Directory"
TRAIN_DIR = "Path to Train Directory"
TEST_DIR = "Path to Test Directory"
NUM_CLASSES = 7  # Number of tick species or classes
CHECKPOINT_DIR = "Path to Checkpoint Directory"
```

### 3. Train the Model

```bash
python train_tick_classifier.py
```

Training proceeds in two phases:

1. **Initial training** (20 epochs) -- Only the classification head is trained while the InceptionV3 base remains frozen.
2. **Fine-tuning** (20 additional epochs) -- The upper layers of InceptionV3 (from layer 100 onward) are unfrozen and trained with a reduced learning rate. Early stopping and learning-rate reduction callbacks are active during this phase.

The best model checkpoint is saved to the specified checkpoint directory, and the final model is saved as `filename.model` (configurable via the `MODEL_FILE` variable).

### 4. Predict on New Images

Use the built-in `predict()` function to classify individual images:

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from train_tick_classifier import predict

img = image.load_img("path/to/image.png", target_size=(299, 299))
preds = predict(load_model("filename.model"), img)
```

To evaluate the full test set, iterate over all images in the test directory and apply the `predict()` function to each.

## Citation

If you use this code in your research, please cite:

> **Potential of dorsal-view images in hierarchical automated tick identification**
> Ali Khalighifar, Amber Grant, Kellee Sundstrom, Kathryn Duncan, A. Townsend Peterson
> *[Journal TBD]*
> DOI: [TBD]

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
