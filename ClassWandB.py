from PIL import Image
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
import wandb  # <- wandb importieren

# 🔐 API Key Login (ersetze den String durch deinen echten API Key)
wandb.login(key="4c9968476bcc046f6d8b7204ae7a2ca803e0e0a9")  # ← HIER DEIN KEY

# 📋 wandb-Projekt initialisieren
wandb.init(
    project="image-classifier",
    config={
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "architecture": "CustomCNN"
    }
)
config = wandb.config

# Datenvorverarbeitung
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Grayscale
    image = image.resize((224, 224))             # Resize
    image = np.array(image, dtype=np.float32) / 255.0  # Normalisieren
    image = np.expand_dims(image, axis=0)        # Channel-Dimension
    return torch.tensor(image, dtype=torch.float32)

# Eigenes Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = idx
            class_folder = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_folder, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        return image_tensor, label

# Dataset laden
full_dataset = CustomImageDataset(root_dir="archive/Training")

# 80% Training / 20% Validierung
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Modell-Definition
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 12 * 12, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Modell & Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
classifier = ImageClassifier().to(device)

optimizer = Adam(classifier.parameters(), lr=config.learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training
train_losses = []
val_losses = []

for epoch in range(config.epochs):
    classifier.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    classifier.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = classifier(val_images)
            loss = loss_fn(val_outputs, val_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")

    # wandb log
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })

wandb.finish()

# Modell speichern
torch.save(classifier.state_dict(), 'model_state.pt')

# Modell laden
with open('model_state.pt', 'rb') as f:
    classifier.load_state_dict(load(f))

# Testdaten
test_dataset = CustomImageDataset(root_dir="archive/Testing")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class_names = test_dataset.classes

# Testauswertung
correct = 0
incorrect = 0
total = 0

classifier.eval()
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = classifier(images)
        predicted_index = torch.argmax(outputs, dim=1).item()
        true_index = labels.item()

        predicted_label = class_names[predicted_index]
        true_label = class_names[true_index]

        if predicted_label == true_label:
            correct += 1
        else:
            incorrect += 1
        total += 1

print('Correct predicted: ', correct)
print('Not correct predicted: ', incorrect)
print('Total Number of tested pictures:', total)
print(f"Accuracy: {correct / total * 100:.2f}%")
