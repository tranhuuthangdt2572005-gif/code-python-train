import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ==========================================
# 1. NGUỒN DỮ LIỆU & ĐỊNH DẠNG
# ==========================================
print("Loading dataset...")
mat_data = sio.loadmat('dataset_lab_276_dl.mat')

# Lấy dữ liệu CSI và Label
# Shape gốc: (200, 30, 3, 5520)
csi_data = mat_data['csid_lab'] 
labels = mat_data['label_lab']

# ==========================================
# 2. TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
print("Preprocessing data...")
# Chuyển đổi ma trận về chuẩn của Deep Learning: (Samples, Time, Subcarriers, Channels)
# Từ (200, 30, 3, 5520) -> (5520, 200, 30, 3)
csi_data = np.transpose(csi_data, (3, 0, 1, 2))

# Lấy Biên độ (Amplitude) từ số phức. 
# (Bỏ qua Phase tạm thời vì nhiễu nặng như đã thấy trong Figure_1_4.png)
X = np.abs(csi_data)

# Chuẩn hóa dữ liệu (Normalization) về khoảng [0, 1] hoặc Z-score
# Ở đây dùng Min-Max cơ bản trên toàn tập dữ liệu
X = (X - np.min(X)) / (np.max(X) - np.min(X))

# Xử lý nhãn: Nhãn gốc từ 1-276, chuyển về 0-275 để đưa vào One-Hot Encoding
y = labels.flatten() - 1 
num_classes = len(np.unique(y))
y_categorical = to_categorical(y, num_classes=num_classes)

# ==========================================
# 3. CHIA TẬP (SPLIT DATASET)
# ==========================================
# Chia 80% Train, 20% Test. Có stratify để đảm bảo mỗi class đều chia đều
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ==========================================
# 4. PHƯƠNG PHÁP MÁY HỌC (CNN 2D)
# ==========================================
# Coi ma trận (200, 30, 3) như một bức ảnh màu có chiều cao 200, rộng 30, 3 kênh màu (3 ăng ten)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 30, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 5. HUẤN LUYỆN & ĐÁNH GIÁ (DEMO KẾT QUẢ)
# ==========================================
print("Training model...")
history = model.fit(X_train, y_train, 
                    epochs=30, # Có thể tăng lên 50-100 nếu cần
                    batch_size=32, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# Đánh giá trên tập test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n=> Test Accuracy: {test_acc*100:.2f}%")

# Vẽ biểu đồ kết quả (Learning Curve)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()