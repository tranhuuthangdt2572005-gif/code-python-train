import scipy.io as sio
import numpy as np

file_path = r"e:\lunain\research\SignFi\dataset_lab_276_dl.mat"

data = sio.loadmat(file_path)

labels = data["label_lab"].flatten()

print("Total samples:", len(labels))

unique, counts = np.unique(labels, return_counts=True)

print("\n===== LABEL DISTRIBUTION =====")

for label, count in zip(unique, counts):
    print(f"Label {label}: {count} samples")

print("\nNumber of classes:", len(unique))
print("Labels:", unique)