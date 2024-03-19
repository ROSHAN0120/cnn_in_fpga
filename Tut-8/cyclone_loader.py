import cv2
import os
import numpy as np

def image_return():
    # Path to the folder containing the images
    folder_path = 'E:/Mini_Project/cyclone_cn/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED/'

    # List to store normalized image values
    training_input = []

    # Iterate over each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Read the color image
            color_image = cv2.imread(os.path.join(folder_path, filename))

            # Convert the color image to grayscale
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image to 28x28 pixels
            resized_image = cv2.resize(grayscale_image, (356, 356))

            # Normalize pixel values to the range [0, 1]
            normalized_image = resized_image / 255.0
    return normalized_image

#             # Flatten the image to a 1D() array
#     #flattened_image = np.array (normalized_image).reshape(-1,1)
#     #print(flattened_image.size)

#             # Append flattened image to the training input list

#     # for i in range(136):
#     #     if flattened_image.size != 12736:
#     #          raise ValueError("size doesnt match")
#     #     else:
#     #          training_input[i] = flattened_image.reshape(12736,1)
#     # Convert the list of normalized images to a numpy array
    
#     training_input = np.array(normalized_image)

#     # Define the corresponding training results (labels)
#     training_results = (25, 27, 28, 30, 30, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 38, 39, 40, 40, 40, 40, 41, 42, 42, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 52, 52, 53, 53, 53, 53, 54, 55, 55, 56, 57, 57, 57, 58, 58, 59, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 67, 67, 68, 69, 69, 70, 73, 74, 74, 74, 75, 77, 77, 81, 81, 82, 82, 83, 84, 84, 85, 85, 85, 86, 86, 86, 87, 94, 98, 99, 101, 102, 106, 111, 112, 115, 118, 119, 128)
#     # reshaped_array = np.array(training_results)#.reshape(-1, 1)

#     training_results = np.array(training_results).flatten()
#     # Convert training results to a numpy array
#     # training_results = np.array(training_results)
#     training_input = training_input.astype(np.float32)
#     training_results = training_results.astype(np.float32)
#     training_input = training_input.reshape(1,-1)
#     # training_input = training_input.T
#     tr_d = (training_input,training_results)
    
#     # training_inputs = [np.reshape(x, (126736, 1)) for x in tr_d[0]]
#     tr_results = [vectorized_result(y) for y in tr_d[1]]
   

#     # Check the shape of the training input array
   

#     # Combine training input and results into a tuple
#     training_data = zip(training_input, tr_results)
#     print(training_input.shape)
    

#     val_data = tr_d
#     # print("Shape of training input array:", training_input.shape)
#     # print(tr_d)
#     #validation_inputs = [np.reshape(x, (126736, 1)) for x in val_data[0]]
#     validation_data = zip(tr_d)
#     # print(validation_data)
#     return (training_data ,validation_data,tr_d)



# def vectorized_result(j):
#     """Return a 10-dimensional unit vector with a 1.0 in the jth
#     position and zeroes elsewhere.  This is used to convert a digit
#     (0...9) into a corresponding desired output from the neural
#     network."""
#     try:
#         j = int(j)
#     except ValueError:
#         raise ValueError("Input value 'j' must be an integer.")

#     if j < 1 or j > 128:
#         raise ValueError("Input value 'j' must be between 1 and 128.")

#     e = np.zeros((128, 1))
#     e[j-1] = 1.0
#     return e



# d = load_data()


