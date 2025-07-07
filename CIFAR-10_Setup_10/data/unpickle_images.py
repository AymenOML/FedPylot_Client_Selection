import os
import pickle
import numpy as np
from PIL import Image

# === CONFIG ===
cifar_path = "./data/cifar-10-batches-py"
output_path = "./output_cifar10"
batches = [f"data_batch_{i}" for i in range(1, 6)]
test_batch = "test_batch"

# === Function to load batch file ===
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        filenames = [f.decode() for f in data_dict[b'filenames']]
        return images, labels, filenames

# === Function to save images ===
def save_images(images, labels, filenames, subset):
    for i in range(len(images)):
        img = images[i].reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        label = labels[i]
        filename = filenames[i]

        label_dir = os.path.join(output_path, subset, str(label))
        os.makedirs(label_dir, exist_ok=True)

        image_path = os.path.join(label_dir, filename + ".png")
        Image.fromarray(img).save(image_path)

# === Process training batches ===
for batch_name in batches:
    file_path = os.path.join(cifar_path, batch_name)
    images, labels, filenames = load_batch(file_path)
    save_images(images, labels, filenames, subset="train")

# === Process test batch ===
file_path = os.path.join(cifar_path, test_batch)
images, labels, filenames = load_batch(file_path)
save_images(images, labels, filenames, subset="test")

print(f"All CIFAR-10 images saved under {output_path}")
