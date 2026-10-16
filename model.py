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

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    losses = []
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

# Step 6 - evaluate
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += xb.size(0)
    return total_loss / n, correct / n

# Step 7 - fit
import copy 

def fit(model, loaders, epochs=5, lr=0.05, seed=42):
    torch.manual_seed(seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, loaders["val"], loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history

# Step 8 - lr_range_test
def lr_range_test(make_model, loader, lrs, n_batches=20, seed=42):
    results = {}
    for lr in lrs:
        torch.manual_seed(seed)
        model = make_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        model.train()
        losses = []
        for i, (xb, yb) in enumerate(loader):
            if i >= n_batches:
                break
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[lr] = sum(losses) / len(losses)
    return results

# Step 9 - random_search
import numpy as np
import torch


def random_search(loaders, n_trials=4, epochs=2, seed=42):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        hidden1 = int(rng.choice([100, 200, 300]))
        hidden2 = int(rng.choice([50, 100]))
        lr = float(rng.choice([0.01, 0.05, 0.1]))

        torch.manual_seed(seed)
        model = MLP(hidden1, hidden2)
        history = fit(model, loaders, epochs, lr, seed)

        trials.append({
            "hidden1": hidden1,
            "hidden2": hidden2,
            "lr": lr,
            "val_acc": max(history["val_acc"]),
        })

    best = max(trials, key=lambda t: t["val_acc"])
    return {"trials": trials, "best": best}

# Step 10 - test_accuracy
def test_accuracy(model, loaders):
    _, acc = evaluate(model, loaders["test"], nn.CrossEntropyLoss())
    return acc

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

