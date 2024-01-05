"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""
import datetime
import os
from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend
from sklearn.model_selection import train_test_split

# Path to the PetImages directory
DATASET_PATH = "PATH/TO/kagglecatsanddogs_5340/PetImages"

# Train / validation batch size
BATCH_SIZE = 32

# Number of train / validation epochs
EPOCHS = 10

# Learning rate
LEARNING_RATE = 0.001

# Set to True to enable tf.keras.callbacks.EarlyStopping
EARLY_STOPPING_ENABLED = False

# Hyperparameter fro tf.keras.callbacks.EarlyStopping
EARLY_STOPPING_DELTA = 0.0001


def load_images_and_labels(directory: str, label: int, images: List, labels: List) -> None:
    """Parses dataset

    Args:
        directory (str): dataset dir
        label (int): 0 for Cat, 1 for Dog
        images (List): parsed images
        labels (List): parsed labels
    """
    print(f"Parsing images from {directory} directory. Please wait")
    for filename in os.listdir(directory):
        try:
            if filename.endswith(".jpg"):
                # Open image and resize it
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                img_array = np.array(img)

                # Check shape
                assert img_array.shape == (224, 224, 3), f"Wrong image shape after resize: {img_array.shape}"

                # Add
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Error parsing image {filename}: {str(e)}")


def main() -> None:
    # Parse dataset
    images = []
    labels = []
    load_images_and_labels(os.path.join(DATASET_PATH, "Cat"), label=0, images=images, labels=labels)
    load_images_and_labels(os.path.join(DATASET_PATH, "Dog"), label=1, images=images, labels=labels)
    images = np.array(images)
    labels = np.array(labels)

    # Split dataset
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    print("Train Images Shape:", train_images.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("Test Images Shape:", test_images.shape)
    print("Test Labels Shape:", test_labels.shape)

    # Parse into TensorFlow Dataset type
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Prepare data
    def _preprocess_image(image, label):
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    train_data = train_data.map(_preprocess_image).batch(BATCH_SIZE)
    test_data = test_data.map(_preprocess_image).batch(BATCH_SIZE)

    # Build model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = tf.keras.Sequential(
        [base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(1, activation="sigmoid")]
    )
    callbacks = []

    # Initialize TensorBoard
    log_dir = os.path.join("tensorboard", "fit") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )
    callbacks.append(tensorboard_callback)

    # Initialize Early Stopping callback
    if EARLY_STOPPING_ENABLED:
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="accuracy", min_delta=EARLY_STOPPING_DELTA, patience=10
        )
        callbacks.append(earlystop_callback)

    # Compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Set learning rate
    backend.set_value(model.optimizer.learning_rate, LEARNING_RATE)

    # Train
    model.fit(train_data, epochs=EPOCHS, validation_data=test_data, callbacks=callbacks)

    # Save model to the model/ directory
    tf.saved_model.save(model, "model/")


if __name__ == "__main__":
    main()
