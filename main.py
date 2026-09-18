from astropy.io import fits
import numpy as np
import pandas as pd
from pathlib import Path
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

filepath = "/home/bshra/Transient_Object_Classifier/TAO_transients/data/AGN/CSS071204:100029+071116.fits"
transients_root = Path("/home/bshra/Transient_Object_Classifier/TAO_transients/data")
non_transients_root = Path("/home/bshra/Transient_Object_Classifier/TAO_non-transients/data/NON")
cache_path = "arrays_cache.npz"


#function that loads transient objects
def load_object_images(filepath):
    with fits.open(filepath) as hdul:
        image_stack = []
        for hdu in hdul:
            if hdu.data is None:
                continue
            elif hdu.data.shape != (64,64):
                continue
            image_stack.append(hdu.data)
        images = np.array(image_stack)
        return images

result = load_object_images(filepath)
print(result.shape)

def build_dataset(transients_root):
    records = []
    for item in transients_root.iterdir():
        if item.is_dir():
            class_name = item.name
            for file in item.glob("*.fits"):
                record = {
                    "filepath": file,
                    "is_transient": 1,
                    "Class": class_name
                }
                records.append(record)
    return records

transients = build_dataset(transients_root)
print(len(transients))
class_counts = Counter()
for record in transients:
    class_name = record["Class"]
    class_counts[class_name] +=1

print(class_counts)
                           
def build_non_transient_dataset(non_transient_root):
    records = []
    for item in non_transient_root.glob("*.fits"):
        record = {
            "filepath": item,
            "is_transient": 0,
            "Class": None,
        }
        records.append(record)
    return records


non_transients = build_non_transient_dataset(non_transients_root)
manifest = transients + non_transients


transient_count = 0
non_transient_count = 0
for record in manifest:
    if record["is_transient"] == 1:
        transient_count +=1
    else:
        non_transient_count += 1
print(transient_count,non_transient_count)

labels = []
for record in manifest:
    labels.append(record["is_transient"])

manifest_train, manifest_temp = train_test_split(
    manifest,
    test_size=0.3,
    stratify=labels,
    random_state=42,
)

labels_temp = []
for record in manifest_temp:
    labels_temp.append(record["is_transient"])

manifest_val, manifest_test = train_test_split(
    manifest_temp,
    test_size=0.5,
    stratify= labels_temp,
    random_state=42
)

print(len(manifest_train), len(manifest_val), len(manifest_test))


def get_mean_image(filepath):
    images = load_object_images(filepath)
    mean_image = np.mean(images, axis=0)
    return mean_image

mean_img = get_mean_image(filepath)
print(mean_img.shape)

def build_image_label_arrays(manifest_split):
    images_list = []
    labels_list = []
    for i, record in enumerate(manifest_split):
        mean_img = get_mean_image(record["filepath"])
        images_list.append(mean_img)
        labels_list.append(record["is_transient"])
        if i % 500 == 0:
            print(f"Processed {i}/{len(manifest_split)}")
    images_list = np.array(images_list)
    labels_list = np.array(labels_list)
    return images_list, labels_list


if os.path.exists(cache_path):
    data = np.load(cache_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
else:
    X_train, y_train = build_image_label_arrays(manifest_train)
    X_val, y_val = build_image_label_arrays(manifest_val)
    X_test, y_test = build_image_label_arrays(manifest_test)
    np.savez(cache_path, X_train=X_train, y_train=y_train,
              X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)


print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

x_train = X_train.reshape(-1,64,64,1)
x_val = X_val.reshape(-1,64,64,1)
x_test = X_test.reshape(-1,64,64,1)

class TransientCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x

model = TransientCNN()
sample = torch.from_numpy(x_train[:4]).permute(0, 3, 1, 2).float()  # NHWC -> NCHW
out = model(sample)
print(out.shape)
    