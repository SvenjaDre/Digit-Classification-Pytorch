import yaml
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import wandb

# WandB Login
wandb.login(key="4c9968476bcc046f6d8b7204ae7a2ca803e0e0a9")

# Gerät wählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# Bildvorverarbeitung
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)

# Eigenes Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
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

# CNN Modell
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        config = wandb.config
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(128 * 14 * 14, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Trainings- und Testdatenverzeichnis
TRAIN_DIR = "archive/Training"
TEST_DIR = "archive/Testing"

# Trainingsfunktion für Sweep
def train():
    wandb.init(project="Classifier-3")
    config = wandb.config
    run_name = wandb.run.name  # z. B. "brave-dust-7"

    # Checkpoint-Verzeichnis vorbereiten
    checkpoint_dir = "Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset und Loader
    full_dataset = CustomImageDataset(root_dir=TRAIN_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Modell, Optimierer, Loss
    classifier = ImageClassifier().to(device)
    optimizer = Adam(classifier.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

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

        # Validierung
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = classifier(val_images)
                loss = loss_fn(val_outputs, val_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        # ✅ Checkpoint speichern alle 10 Epochen
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_epoch{epoch+1}.pt")
            torch.save(classifier.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            wandb.save(checkpoint_path)

    # Finales Modell speichern und auf Testdaten evaluieren
    final_model_path = "model_state.pt"
    torch.save(classifier.state_dict(), final_model_path)
    evaluate_on_test_data(final_model_path)

    wandb.finish()

# Sweep-Konfiguration laden
def load_sweep_config(path="sweep.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# Test-Funktion
def evaluate_on_test_data(model_path="model_state.pt"):
    print("\n🔍 Testing on separate test set...")

    test_dataset = CustomImageDataset(root_dir=TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_names = test_dataset.classes

    classifier = ImageClassifier().to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    correct = 0
    incorrect = 0
    total = 0

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

    accuracy = (correct / total) * 100

    print("✅ Correct predictions: ", correct)
    print("❌ Incorrect predictions: ", incorrect)
    print(" Total tested images:", total)
    print(f" Accuracy: {accuracy:.2f}%")

    wandb.log({
        "test_accuracy": accuracy,
        "test_correct": correct,
        "test_incorrect": incorrect,
        "test_total": total
    })

# Hauptfunktion
if __name__ == "__main__":
    sweep_config = load_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="Classifier-3")
    wandb.agent(sweep_id, function=train)
