import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# --- Pastikan file ditemukan ---
file_path = r'D:\Code\Github\Sign language detection\data_fixed.pickle'
print("Working directory:", os.getcwd())
print("File exists?", os.path.exists(file_path))

# --- Baca file ---
data_dict = pickle.load(open(file_path, 'rb'))
print(f"Total data: {len(data_dict['data'])}")

# --- Cek panjang setiap data ---
shapes = [len(d) for d in data_dict['data']]
unique_shapes = sorted(set(shapes))
print(f"Shape unik ditemukan: {unique_shapes}")

# --- Tentukan shape utama (yang paling sering muncul) ---
expected_shape = max(set(shapes), key=shapes.count)
print(f"Shape yang seharusnya: {expected_shape}")

# --- Tampilkan data rusak tanpa karakter unicode ---
print("\nData rusak (shape tidak sesuai):")
bad_indices = []
for i, d in enumerate(data_dict['data']):
    if len(d) != expected_shape:
        print(f"  Index {i} -> shape = {len(d)}")
        bad_indices.append(i)

if not bad_indices:
    print("Tidak ada data rusak ditemukan!")
else:
    print(f"Total data rusak: {len(bad_indices)}")

# --- Bersihkan data ---
clean_data = []
clean_labels = []
for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == expected_shape:
        clean_data.append(d)
        clean_labels.append(l)

print(f"\nData sebelum dibersihkan: {len(data_dict['data'])}")
print(f"Data setelah dibersihkan: {len(clean_data)}")

# --- Ubah ke numpy array ---
data = np.stack(clean_data)
labels = np.asarray(clean_labels)

# --- Split data ---
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# --- Train model ---
model = RandomForestClassifier()
model.fit(x_train, y_train)

# --- Evaluate ---
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"\n{score * 100:.2f}% of samples were classified correctly!")

# --- Simpan model ---
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("\nModel berhasil disimpan ke 'model.p'")
