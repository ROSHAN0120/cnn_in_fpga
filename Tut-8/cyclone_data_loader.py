import cv2
import os
import numpy as np

def preprocess_images(directory):
    # Initialize an empty list to store flattened image matrices
    flattened_images = []

    # Common size for resizing
    common_size = (320, 320)

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image and its name ends with "_grayscale"
        if filename.lower().endswith("_grayscale.jpg") or filename.lower().endswith("_grayscale.jpg"):
            # Construct the full path to the image
            filepath = os.path.join(directory, filename)

            # Read the image using OpenCV
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale mode

            # Check if the image is successfully loaded
            if image is None:
                print(f"Unable to read image at path: {filepath}")
                continue

            # Resize the image to a common size (320x320)
            resized_image = cv2.resize(image, common_size)

            # Normalize the resized image
            normalized_image = resized_image / 255.0

            # Flatten the image to a 1D array
            flattened_image = normalized_image.flatten()

            # Append the flattened image to the list
            flattened_images.append(flattened_image)

    # Convert the list of flattened images to a numpy array
    flattened_images_array = np.array(flattened_images)

    return flattened_images_array



