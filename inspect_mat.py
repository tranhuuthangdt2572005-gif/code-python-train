import scipy.io as sio

file_path = "dataset_lab_276_dl.mat"

data = sio.loadmat(file_path)

print("Keys:")
for key in data.keys():
    print(key)

print("\nVariables:")
for key, value in data.items():
    if not key.startswith("__"):
        print(
            key,
            "type =", type(value),
            "shape =", getattr(value, "shape", None),
            "dtype =", getattr(value, "dtype", None)
        )