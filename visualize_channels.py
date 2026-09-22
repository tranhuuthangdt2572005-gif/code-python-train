import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

file_path = r"E:\lunain\research\SignFi\dataset_lab_276_dl.mat"

data = sio.loadmat(file_path)

csid = data["csid_lab"]
labels = data["label_lab"].flatten()

sample_id = 0
sample = csid[:, :, :, sample_id]

print("Sample:", sample_id)
print("Label:", labels[sample_id])
print("Sample shape:", sample.shape)

# Complex CSI -> amplitude
amplitude = np.abs(sample)

# Complex CSI -> phase
phase = np.angle(sample)

# ==============================
# AMPLITUDE - 3 CHANNELS
# ==============================

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for ch in range(3):
    im = axes[ch].imshow(
        amplitude[:, :, ch].T,
        aspect="auto",
        origin="lower"
    )

    axes[ch].set_title(f"Amplitude - Channel {ch + 1}")
    axes[ch].set_xlabel("Time / CSI samples")
    axes[ch].set_ylabel("Subcarrier")

    fig.colorbar(im, ax=axes[ch], label="Amplitude")

plt.suptitle(
    f"CSI Amplitude - Sample {sample_id}, Label {labels[sample_id]}",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ==============================
# PHASE - 3 CHANNELS
# ==============================

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for ch in range(3):
    im = axes[ch].imshow(
        phase[:, :, ch].T,
        aspect="auto",
        origin="lower"
    )

    axes[ch].set_title(f"Phase - Channel {ch + 1}")
    axes[ch].set_xlabel("Time / CSI samples")
    axes[ch].set_ylabel("Subcarrier")

    fig.colorbar(im, ax=axes[ch], label="Phase")

plt.suptitle(
    f"CSI Phase - Sample {sample_id}, Label {labels[sample_id]}",
    fontsize=14
)

plt.tight_layout()
plt.show()