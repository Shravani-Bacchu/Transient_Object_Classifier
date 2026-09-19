from astropy.io import fits
import numpy as np
import pandas as pd
from pathlib import Path
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
from sklearn.metrics import classification_report, confusion_matrix

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

train_mean = x_train.mean()
train_std = x_train.std()

x_train = (x_train - train_mean) /train_std
x_val = (x_val - train_mean) / train_std
x_test = (x_test - train_mean) / train_std

print(x_train.min(),x_train.max(), x_train.mean(), x_train.std())


class TransientCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=12544, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = TransientCNN()
sample = torch.from_numpy(x_train[:4]).permute(0, 3, 1, 2).float()
out = model(sample)
print(out.shape)
print(out) 
print(x_train.min(), x_train.max(), x_train.mean())

criterion = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(),lr=0.001)

X_train_t = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

X_val_t = torch.from_numpy(x_val).permute(0, 3, 1, 2).float()
y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)


train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)

xv, yv = next(iter(val_loader))
print(xv.shape, yv.shape)
epochs = 6

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        predictions = model(images)
        loss = criterion(predictions, labels)

        optimiser.zero_grad()    
        loss.backward()          
        optimiser.step()         

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}  train loss: {avg_train_loss:.2f}")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            predictions = model(images)
            loss = criterion(predictions, labels)
            val_loss += loss.item()

            predicted_classes = (predictions > 0.5).float()
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs}  train loss: {avg_train_loss:.2f} "f"val loss: {avg_val_loss:.2f}  val acc: {val_accuracy:.2f}")


X_test_t = torch.from_numpy(x_test).permute(0,3,1,2).float()
y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_classes = (predictions > 0.5).float()
        all_preds.append(predicted_classes)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))

manifest_train_t = []
for record in manifest_train:
    if record["is_transient"] == 1:
        manifest_train_t.append(record)

manifest_val_t = []
for record in manifest_val:
    if record["is_transient"] == 1:
        manifest_val_t.append(record)


manifest_test_t = []
for record in manifest_test:
    if record["is_transient"] == 1:
        manifest_test_t.append(record)

print(len(manifest_train_t), len(manifest_val_t), len(manifest_test_t))

class_names = ["AGN","BZ","CV","OTHER","SN"]
class_to_idx = {}
for i, name in enumerate(class_names):
    class_to_idx[name] = i

print(class_to_idx)

def build_class_arrays(manifest_split, class_to_idx):
    images_list = []
    labels_list = []
    for i, record in enumerate(manifest_split):
        mean_img = get_mean_image(record["filepath"])
        class_name = record["Class"]
        class_index = class_to_idx[class_name]
        images_list.append(mean_img)
        labels_list.append(class_index)
        if i % 500 == 0:
            print(f"Processed {i}/{len(manifest_split)}")
    images_list = np.array(images_list)
    labels_list = np.array(labels_list)
    return images_list, labels_list

X_test_small, y_test_small = build_class_arrays(manifest_train_t[:20], class_to_idx)
print(X_test_small.shape, y_test_small.shape)
print(y_test_small)

cache_path_stage2 = "arrays_cache_stage2.npz"

if os.path.exists(cache_path_stage2):
    data2 = np.load(cache_path_stage2)
    X_train2, y_train2 = data2["X_train"], data2["y_train"]
    X_val2, y_val2 = data2["X_val"], data2["y_val"]
    X_test2, y_test2 = data2["X_test"], data2["y_test"]
else:
    X_train2, y_train2 = build_class_arrays(manifest_train_t, class_to_idx)
    X_val2, y_val2 = build_class_arrays(manifest_val_t, class_to_idx)
    X_test2, y_test2 = build_class_arrays(manifest_test_t, class_to_idx)
    np.savez(cache_path_stage2, X_train=X_train2, y_train=y_train2,
              X_val=X_val2, y_val=y_val2, X_test=X_test2, y_test=y_test2)

print(X_train2.shape, y_train2.shape)
print(X_val2.shape, y_val2.shape)
print(X_test2.shape, y_test2.shape)

x_train2 = X_train2.reshape(-1, 64, 64, 1)
x_val2 = X_val2.reshape(-1, 64, 64, 1)
x_test2 = X_test2.reshape(-1, 64, 64, 1)

train_mean2 = x_train2.mean()
train_std2 = x_train2.std()

x_train2 = (x_train2 - train_mean2) / train_std2
x_val2 = (x_val2 - train_mean2) / train_std2
x_test2 = (x_test2 - train_mean2) / train_std2

print(x_train2.min(), x_train2.max(), x_train2.mean(), x_train2.std())