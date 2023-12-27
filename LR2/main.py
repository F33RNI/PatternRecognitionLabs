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
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F

# Image to classify
IMAGE_PATH = "kagglecatsanddogs_5340/PetImages/Cat/2.jpg"


def main() -> None:
    # Load and process image
    image = Image.open(IMAGE_PATH)
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Load model
    model = ViTForImageClassification.from_pretrained("akahana/vit-base-cats-vs-dogs")

    # Predict
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)

    # Get the predicted class (cat or dog)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Map the class index to the corresponding label
    labels = ["cat", "dog"]
    predicted_label = labels[predicted_class]

    print("Predicted Label:", predicted_label)
    print("Probabilities:", probabilities)


if __name__ == "__main__":
    main()
