import numpy as np
import cyclone_loader
import mnist_loader

#data2,x,y = mnist_loader.load_data_wrapper()
data = cyclone_loader.load_data()
print(data[1])
#test_data_list = list(data2)
#print(test_data_list[0])

def split_data_load(data, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, shuffle=True, random_seed=None):
    """
    Split the dataset into training, testing, and validation sets.a

    Parameters:
        data (numpy.ndarray or list): The dataset to be split.
        train_ratio (float): The ratio of data to allocate for training. Default is 0.7.
        test_ratio (float): The ratio of data to allocate for testing. Default is 0.15.
        val_ratio (float): The ratio of data to allocate for validation. Default is 0.15.
        shuffle (bool): Whether to shuffle the dataset before splitting. Default is True.
        random_seed (int): Random seed for reproducibility. Default is None.

    Returns:
        tuple: A tuple containing three sets: (training_data, testing_data, validation_data).
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle the dataset if required
    if shuffle:
        np.random.shuffle(data)

    # Calculate the sizes of each set
    total_samples = len(data)
    train_size = int(train_ratio * total_samples)
    test_size = int(test_ratio * total_samples)
    val_size = total_samples - train_size - test_size

    # Split the dataset into training, testing, and validation sets
    training_data = zip(data[:train_size])
    testing_data = zip(data[train_size:train_size + test_size])
    validation_data = zip(data[train_size + test_size:])


    return training_data, testing_data, validation_data

# Example usage:
# Assume 'data' is your dataset
# training_data, testing_data, validation_data = split_data(data)
