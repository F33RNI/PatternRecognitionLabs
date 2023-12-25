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
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Image to classify
IMAGE_PATH = "images/pinwheel.jpg"

# Original image will be resized to this size
IMAGE_SIZE = (224, 224)

# Number of top predictions to print
PREDICTIONS_TOP = 3


def classify(image_path: str):
    """
    Classifies image using ResNet50 model
    :param image_path: path to image file to classify
    :return:
    """
    image = Image.open(image_path)
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)

    # Convert the image to a TensorFlow tensor
    img_tensor = tf.convert_to_tensor(image)

    # Expand the dimensions of the tensor to add a batch dimension
    img_batch = tf.expand_dims(img_tensor, axis=0)

    # Preprocess the image
    img_preprocessed = preprocess_input(img_batch)

    # Load the ResNet50 model
    model = ResNet50(weights="imagenet")

    # Make a prediction using the model
    prediction_ = model.predict(img_preprocessed)

    # Decode the prediction
    predictions = decode_predictions(prediction_, top=PREDICTIONS_TOP)

    # Reduce the dimensions of the tensor
    return predictions[0]


if __name__ == "__main__":
    print("The top {} predicted classes are:".format(PREDICTIONS_TOP))
    for prediction in classify(IMAGE_PATH):
        print(prediction)
