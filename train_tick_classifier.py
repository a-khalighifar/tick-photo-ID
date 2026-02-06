"""Tick species image classifier using transfer learning with InceptionV3.

This script implements a two-phase training pipeline for classifying
dorsal-view photographs of North American ixodid tick species:

  1. Initial training with a frozen InceptionV3 base (pre-trained on ImageNet).
  2. Fine-tuning of upper convolutional layers with a reduced learning rate.

Data augmentation (random horizontal flips and rotations) is applied during
training.  Early stopping, learning-rate reduction, and model checkpointing
callbacks are used during the fine-tuning phase.

Author: Ali Khalighifar
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# User-configurable paths
# ---------------------------------------------------------------------------
BASE_DIR = "Path to Base Directory"
TRAIN_DIR = "Path to Train Directory"
TEST_DIR = "Path to Test Directory"
CHECKPOINT_DIR = "Path to Checkpoint Directory"
MODEL_FILE = "filename.model"

NUM_CLASSES = "Number of classes/tick species"  # Set to an integer, e.g. 7

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.2
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 20
FINE_TUNE_AT = 100  # Fine-tune InceptionV3 layers from this index onward
FINE_TUNE_LR = 1e-3
LR_REDUCTION_FACTOR = 0.1
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE = 5


def build_data_generators(train_dir, image_size, batch_size, validation_split):
    """Create training and validation data generators with rescaling.

    A fraction of the training images (defined by *validation_split*) is
    reserved for validation.  The validation set is distinct from the test set.

    Parameters
    ----------
    train_dir : str
        Path to the training image directory (one subfolder per class).
    image_size : tuple of int
        Target (height, width) for resizing images.
    batch_size : int
        Number of images per batch.
    validation_split : float
        Fraction of training images reserved for validation.

    Returns
    -------
    train_dataset, validation_dataset
    """
    image_generator = ImageDataGenerator(
        rescale=1 / 255, validation_split=validation_split
    )

    train_dataset = image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=image_size,
        subset="training",
        class_mode="categorical",
    )

    validation_dataset = image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=image_size,
        subset="validation",
        class_mode="categorical",
    )

    return train_dataset, validation_dataset


def build_model(num_classes, image_size, dropout_rate):
    """Build an InceptionV3-based classifier with a frozen convolutional base.

    The model applies data augmentation (random horizontal flip and rotation),
    InceptionV3 preprocessing, global average pooling, dropout, and a softmax
    classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    image_size : tuple of int
        Spatial dimensions (height, width) of input images.
    dropout_rate : float
        Dropout probability applied before the output layer.

    Returns
    -------
    model : keras.Model
    base_model : keras.Model
        The InceptionV3 base (frozen), returned so it can be unfrozen later
        for fine-tuning.
    """
    # Data augmentation layers
    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # Load InceptionV3 without classification layers; freeze convolutional base
    base_model = keras.applications.inception_v3.InceptionV3(
        weights="imagenet",
        input_shape=(*image_size, 3),
        include_top=False,
    )
    base_model.trainable = False

    # Assemble the full model
    inputs = keras.Input(shape=(*image_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    predictions = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=predictions)

    return model, base_model


def train(model, train_dataset, validation_dataset, initial_epochs):
    """Train the model with the convolutional base frozen.

    Parameters
    ----------
    model : keras.Model
    train_dataset : DirectoryIterator
    validation_dataset : DirectoryIterator
    initial_epochs : int

    Returns
    -------
    history : keras.callbacks.History
    """
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_dataset, epochs=initial_epochs, validation_data=validation_dataset
    )
    return history


def fine_tune(model, base_model, train_dataset, validation_dataset, history,
              initial_epochs, fine_tune_epochs, fine_tune_at, fine_tune_lr,
              checkpoint_dir, model_file, lr_reduction_factor,
              early_stop_patience, lr_reduce_patience):
    """Fine-tune upper layers of the convolutional base.

    Layers before *fine_tune_at* remain frozen to reduce computational cost.
    Early stopping, model checkpointing, and learning-rate reduction callbacks
    are applied during this phase.

    Parameters
    ----------
    model : keras.Model
    base_model : keras.Model
        The InceptionV3 base whose upper layers will be unfrozen.
    train_dataset : DirectoryIterator
    validation_dataset : DirectoryIterator
    history : keras.callbacks.History
        History object from the initial training phase.
    initial_epochs : int
    fine_tune_epochs : int
    fine_tune_at : int
        Layer index from which to begin fine-tuning.
    fine_tune_lr : float
        Learning rate for fine-tuning.
    checkpoint_dir : str
        Path for saving model checkpoints.
    model_file : str
        Filename for saving the final model.
    lr_reduction_factor : float
    early_stop_patience : int
    lr_reduce_patience : int

    Returns
    -------
    history_fine : keras.callbacks.History
    """
    total_epochs = initial_epochs + fine_tune_epochs

    # Unfreeze the convolutional base for fine-tuning
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

    # Keep layers before fine_tune_at frozen
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(fine_tune_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=early_stop_patience,
            verbose=1, mode="max",
        ),
        ModelCheckpoint(
            filepath=checkpoint_dir, monitor="val_accuracy",
            mode="max", save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", mode="max",
            factor=lr_reduction_factor,
            patience=lr_reduce_patience, verbose=1,
        ),
    ]

    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=callbacks,
    )
    model.save(model_file)
    return history_fine


def predict(model, img):
    """Return class-probability predictions for a single image.

    Parameters
    ----------
    model : keras.Model
        A trained classifier.
    img : PIL.Image
        Input image (will be converted to an array and preprocessed).

    Returns
    -------
    numpy.ndarray
        Predicted class probabilities.
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


# -----------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Build data generators
    train_dataset, validation_dataset = build_data_generators(
        TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT,
    )
    # NUM_CLASSES can alternatively be derived from the dataset:
    # NUM_CLASSES = len(train_dataset.class_indices)

    # Build and train with frozen base
    model, base_model = build_model(NUM_CLASSES, IMAGE_SIZE, DROPOUT_RATE)
    history = train(model, train_dataset, validation_dataset, INITIAL_EPOCHS)

    # Fine-tune upper convolutional layers
    fine_tune(
        model, base_model, train_dataset, validation_dataset, history,
        INITIAL_EPOCHS, FINE_TUNE_EPOCHS, FINE_TUNE_AT, FINE_TUNE_LR,
        CHECKPOINT_DIR, MODEL_FILE,
        LR_REDUCTION_FACTOR, EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE,
    )

    # Example: predict on a single test image
    # img = image.load_img("path/to/test_image.png", target_size=IMAGE_SIZE)
    # preds = predict(load_model(MODEL_FILE), img)
