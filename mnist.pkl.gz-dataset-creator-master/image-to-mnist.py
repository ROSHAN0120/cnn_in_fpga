from PIL import Image
from numpy import genfromtxt
import gzip,pickle
from glob import glob
import numpy as np
import pandas as pd

def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] / 255.0 for f in list(img.getdata())] 
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels, names = ["class"])
        return np.array(dataset), np.array(df["class"])
    else:
        return np.array(dataset)
    
Data1, y1 = dir_to_dataset("train/*.jpg","train.csv")
Data2, y2 = dir_to_dataset("valid/*.jpg","valid.csv")
Data3, y3 = dir_to_dataset("test/*.jpg","test.csv")

# Data and labels are read 

train_set_x = Data1[:97]
train_set_y = y1[:97]
val_set_x = Data2[:24]
val_set_y = y2[:24]
test_set_x = Data3[:15]
test_set_y = y3[:15]


# Divided dataset into 3 parts. I had 7717 images for training, 1653 images for validation and 1654 images for testing

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('cyclone_norm.pkl.gz','wb')
pickle.dump(dataset, f, protocol=2)
print(train_set)

f.close()
