import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import random

# === STEP 1: Load CIFAR-10 ===
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
num_classes = 10

# === STEP 2: Create non-IID client splits using Dirichlet distribution ===
def create_dirichlet_clients(dataset, num_clients=10, alpha=0.1, num_classes=10):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_idx = class_indices[cls]
        np.random.shuffle(cls_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, proportions)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    clients = [Subset(dataset, indices) for indices in client_indices]
    return clients

clients_data = create_dirichlet_clients(trainset, num_clients=10, alpha=0.1)

# === STEP 3: Compute label distributions
def get_label_distribution(dataset, num_classes=10):
    labels = [dataset[i][1] for i in range(len(dataset))]
    total = len(labels)
    counts = Counter(labels)
    distribution = np.array([counts.get(i, 0) / total for i in range(num_classes)])
    return distribution

# === STEP 4: KL divergence
def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    return np.sum(p * np.log(p / q))

# === STEP 5: Compute per-client divergence
def compute_all_kl_divergences(clients_data, global_data, num_classes=10):
    global_dist = get_label_distribution(global_data, num_classes)
    divergences = []
    for i, client_data in enumerate(clients_data):
        client_dist = get_label_distribution(client_data, num_classes)
        kl = kl_divergence(client_dist, global_dist)
        divergences.append((i, kl))
    return divergences

# === STEP 6: Run and plot
divergences = compute_all_kl_divergences(clients_data, trainset)

client_ids, kl_values = zip(*divergences)
plt.bar(client_ids, kl_values)
plt.xlabel("Client ID")
plt.ylabel("KL Divergence to Global")
plt.title("Data Skew per Client (CIFAR-10, non-IID α=0.1)")
plt.tight_layout()
plt.show()
