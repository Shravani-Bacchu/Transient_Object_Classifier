from astropy.io import fits
import numpy as np
from pathlib import Path
import os
from collections import Counter
from sklearn.model_selection import train_test_split

filepath = "/home/bshra/Transient_Object_Classifier/TAO_transients/data/AGN/CSS071204:100029+071116.fits"
transients_root = Path("/home/bshra/Transient_Object_Classifier/TAO_transients/data")
non_transients_root = Path("/home/bshra/Transient_Object_Classifier/TAO_non-transients/data/NON")

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
print(len(test))
class_counts = Counter()
for record in test:
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
print(len(non_transients))


manifest = transients + non_transients
print(len(manifest))

transient_count = sum(1 for record in manifest if record["is_transient"] == 1)
non_transient_count = sum(1 for record in manifest if record["is_transient"] == 0)
print(transient_count, non_transient_count)
