import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Đổi thành đường dẫn file .mat của bạn
file_path = r"E:\lunain\research\SignFi\dataset_lab_276_dl.mat"

print("Loading dataset...")
data = sio.loadmat(file_path)

csid = data["csid_lab"]
labels = data["label_lab"].flatten()

print("CSI shape:", csid.shape)
print("Labels shape:", labels.shape)

# Lấy sample đầu tiên
sample_id = 0
sample = csid[:, :, :, sample_id]

print("\nSample shape:", sample.shape)
print("Label:", labels[sample_id])

# Complex CSI -> amplitude
amplitude = np.abs(sample)

# Complex CSI -> phase
phase = np.angle(sample)

print("Amplitude shape:", amplitude.shape)
print("Phase shape:", phase.shape)

# ------------------------------------------------
# Vẽ amplitude
# ------------------------------------------------

plt.figure(figsize=(12, 6))

plt.imshow(
    amplitude[:, :, 0].T,
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="Amplitude")
plt.xlabel("Time / CSI samples")
plt.ylabel("Subcarrier")
plt.title(f"CSI Amplitude - Sample {sample_id}, Label {labels[sample_id]}")

plt.tight_layout()
plt.show()

# ------------------------------------------------
# Vẽ phase
# ------------------------------------------------

plt.figure(figsize=(12, 6))

plt.imshow(
    phase[:, :, 0].T,
    aspect="auto",
    origin="lower"
)

plt.colorbar(label="Phase")
plt.xlabel("Time / CSI samples")
plt.ylabel("Subcarrier")
plt.title(f"CSI Phase - Sample {sample_id}, Label {labels[sample_id]}")

plt.tight_layout()
plt.show()