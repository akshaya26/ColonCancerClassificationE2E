"""
This will remove specular reflections from image
"""

import cv2
import numpy as np
import torch

print("Preprocess module loaded")

def inpaint(image_bytes):
    # Convert bytes → numpy
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes")

    # Convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)

    # Low saturation mask
    mask = (s < 15).astype(np.uint8) * 255

    # Dilate mask
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2
    )

    # Inpaint
    inpaint_image = cv2.inpaint(image, mask, 4, cv2.INPAINT_NS)

    return inpaint_image


def preprocess_image(image_bytes):
    # Step 1: inpaint
    image = inpaint(image_bytes)

    # Step 2: convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: resize
    image = cv2.resize(image, (512, 512))

    # Step 4: normalize
    image = image / 255.0
    image = (image - 0.2627) / 0.2455

    # Step 5: convert to tensor
    image = torch.tensor(image, dtype=torch.float32)

    # Step 6: add batch + channel dims → (1,1,512,512)
    image = image.unsqueeze(0).unsqueeze(0)

    return image