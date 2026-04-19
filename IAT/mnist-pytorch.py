import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch.nn.functional as F

# -----------------
# Device (CPU / GPU)
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------
# Hyperparameters
# -----------------
batch_size = 64
epochs = 5
learning_rate = 0.001

# -----------------
# Data (auto-download)
# -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -----------------
# Model
# -----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

model = Net().to(device)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def show_good_and_bad_predictions(model, loader, device, epoch, num_samples=5):
    model.eval()

    all_images = []
    all_labels = []
    all_preds = []
    all_confs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_confs.append(confs.cpu())

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = torch.cat(all_preds, dim=0)
    confs = torch.cat(all_confs, dim=0)

    correct_mask = preds == labels
    wrong_mask = preds != labels

    correct_idx = torch.where(correct_mask)[0]
    wrong_idx = torch.where(wrong_mask)[0]

    # "works very well" = correct with highest confidence
    if len(correct_idx) > 0:
        correct_confs = confs[correct_idx]
        top_correct_order = torch.argsort(correct_confs, descending=True)
        top_correct_idx = correct_idx[top_correct_order[:num_samples]]
    else:
        top_correct_idx = torch.tensor([], dtype=torch.long)

    # "fails" = wrong with highest confidence (most interesting mistakes)
    if len(wrong_idx) > 0:
        wrong_confs = confs[wrong_idx]
        top_wrong_order = torch.argsort(wrong_confs, descending=True)
        top_wrong_idx = wrong_idx[top_wrong_order[:num_samples]]
    else:
        top_wrong_idx = torch.tensor([], dtype=torch.long)

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    # If num_samples == 1, axes won't be 2D; this keeps indexing simple
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    # top row: correct predictions
    for i in range(num_samples):
        ax = axes[0, i]
        ax.axis("off")

        if i < len(top_correct_idx):
            idx = top_correct_idx[i].item()
            ax.imshow(images[idx].squeeze(), cmap="gray")
            ax.set_title(
                f"✓ P:{preds[idx].item()}\nT:{labels[idx].item()}\n{confs[idx].item():.2f}"
            )
        else:
            ax.set_title("No sample")

    # bottom row: wrong predictions
    for i in range(num_samples):
        ax = axes[1, i]
        ax.axis("off")

        if i < len(top_wrong_idx):
            idx = top_wrong_idx[i].item()
            ax.imshow(images[idx].squeeze(), cmap="gray")
            ax.set_title(
                f"✗ P:{preds[idx].item()}\nT:{labels[idx].item()}\n{confs[idx].item():.2f}"
            )
        else:
            ax.set_title("No error")

    fig.suptitle(f"Epoch {epoch}: top = most confident correct, bottom = most confident wrong")
    plt.tight_layout()
    plt.show()


import torch.nn.functional as F
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

def show_predictions(model, loader, device, num_samples=5):
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)

    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

    images = images.cpu()
    probs = probs.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    plt.figure(figsize=(10, 2))

    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")

        pred = preds[i].item()
        true = labels[i].item()
        conf = probs[i][pred].item()

        plt.title(f"P:{pred}\nT:{true}\n{conf:.2f}")

    plt.tight_layout()
    plt.show()

def launch_digit_canvas(model, device):
    model.eval()

    # large drawing canvas, later downsampled to 28x28
    canvas_size = 280
    brush_radius = 12
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    fig = plt.figure(figsize=(9, 5))
    ax_draw = fig.add_axes([0.05, 0.15, 0.45, 0.75])
    ax_bar = fig.add_axes([0.58, 0.2, 0.35, 0.65])
    ax_clear = fig.add_axes([0.18, 0.03, 0.18, 0.08])

    clear_button = Button(ax_clear, "Clear")

    img_artist = ax_draw.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    ax_draw.set_title("Draw a digit here")
    ax_draw.set_xticks([])
    ax_draw.set_yticks([])

    bars = ax_bar.bar(range(10), np.zeros(10))
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xlabel("Digit")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title("Model probabilities")

    is_drawing = False

    def preprocess_canvas():
        """
        Convert 280x280 canvas to 1x1x28x28 tensor roughly matching MNIST.
        """
        img = torch.tensor(canvas, dtype=torch.float32)

        # downsample to 28x28
        img = img.unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
        img = F.interpolate(img, size=(28, 28), mode="bilinear", align_corners=False)

        # normalize like MNIST input
        img = (img - 0.1307) / 0.3081
        return img.to(device)

    def predict():
        with torch.no_grad():
            x = preprocess_canvas()
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        for i, b in enumerate(bars):
            b.set_height(float(probs[i]))

        pred = int(np.argmax(probs))
        ax_bar.set_title(f"Model probabilities   |   prediction = {pred}")
        fig.canvas.draw_idle()

    def draw_at(event):
        if event.inaxes != ax_draw or event.xdata is None or event.ydata is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        yy, xx = np.ogrid[:canvas_size, :canvas_size]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= brush_radius ** 2

        # paint white on black
        canvas[mask] = 1.0

        # optional soft edge around brush
        outer_mask = (xx - x) ** 2 + (yy - y) ** 2 <= (brush_radius * 1.8) ** 2
        canvas[outer_mask] = np.maximum(canvas[outer_mask], 0.35)

        img_artist.set_data(canvas)
        predict()

    def on_press(event):
        nonlocal is_drawing
        if event.inaxes == ax_draw:
            is_drawing = True
            draw_at(event)

    def on_release(event):
        nonlocal is_drawing
        is_drawing = False

    def on_move(event):
        if is_drawing:
            draw_at(event)

    def on_clear(event):
        canvas.fill(0.0)
        img_artist.set_data(canvas)
        for b in bars:
            b.set_height(0.0)
        ax_bar.set_title("Model probabilities")
        fig.canvas.draw_idle()

    clear_button.on_clicked(on_clear)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()



# -----------------
# Loss + optimizer
# -----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# -----------------
# Training loop
# -----------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # forward
        output = model(data)
        loss = criterion(output, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # -----------------
    # Evaluation
    # -----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total

    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")

    show_good_and_bad_predictions(model, test_loader, device, epoch + 1, num_samples=5)
    # show_predictions(model, test_loader, device)


print("Training complete.")

launch_digit_canvas(model, device)
