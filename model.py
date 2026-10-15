"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import gzip
import os
import tempfile
import urllib.request

import numpy as np
import torch


def load_fashion_mnist(n_train=10000, n_test=2000):
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    data = {}
    for key, fname in files.items():
        path = os.path.join(tempfile.gettempdir(), fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + fname, path)
        with gzip.open(path, "rb") as f:
            buf = f.read()
        if "images" in key:
            data[key] = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            data[key] = np.frombuffer(buf, dtype=np.uint8, offset=8)

    return {
        "X_train": torch.from_numpy(data["train_images"][:n_train].astype(np.float32) / 255.0),
        "y_train": torch.from_numpy(data["train_labels"][:n_train].astype(np.int64)),
        "X_test": torch.from_numpy(data["test_images"][:n_test].astype(np.float32) / 255.0),
        "y_test": torch.from_numpy(data["test_labels"][:n_test].astype(np.int64)),
    }

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.x = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return ((self.x[i] - self.mean) / self.std, self.y[i])

# Step 3 - make_loaders
def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    X_tr = data["X_train"][:-val_size]
    y_tr = data["y_train"][:-val_size]
    X_val = data["X_train"][-val_size:]
    y_val = data["y_train"][-val_size:]

    train_ds = FashionDataset(X_tr, y_tr)
    val_ds = FashionDataset(X_val, y_val)
    test_ds = FashionDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "sizes": (len(train_ds), len(val_ds), len(test_ds)),
    }

# Step 4 - MLP
class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

