import cyclone_data_loader
import numpy as np

def vectorized_result(j):

    e = np.zeros((128, 1))
    e[j-1] = 1.0
    return e

def prepare_training_data():
    # Directory containing the images
    directory = "E:/Mini_Project/cyclone_cn/processed_images"

    # Preprocess images
    training_inputs = cyclone_data_loader.preprocess_images(directory)

    # Define training results
    training_results = (25, 27, 28, 30, 30, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 38, 39, 40, 40, 40, 40, 41, 42, 42, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 52, 52, 53, 53, 53, 53, 54, 55, 55, 56, 57, 57, 57, 58, 58, 59, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 67, 67, 68, 69, 69, 70, 73, 74, 74, 74, 75, 77, 77, 81, 81, 82, 82, 83, 84, 84, 85, 85, 85, 86, 86, 86, 87, 94, 98, 99, 101, 102, 106, 111, 112, 115, 118, 119, 128)
    training_results = np.array(training_results)

    # Combine inputs and results into training data
    tr_d = (training_inputs, training_results)

    # Reshape inputs and vectorize results
    training_inputs = [np.reshape(x, (102400, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    horizontal_result = tr_d[1].ravel()

    print(horizontal_result)
    # Zip inputs and results together
    train_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (102400, 1)) for x in tr_d[0]]
    validation_data = zip(validation_inputs, tr_d[1])

    return train_data , validation_data

