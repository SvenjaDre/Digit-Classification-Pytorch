import yaml
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import wandb
import random
from sklearn.metrics import roc_auc_score, confusion_matrix

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


# Eigenes Dataset, Unterscheidung Glioma und Meningioma
#class CustomImageDataset(Dataset):
#    def __init__(self, root_dir):
#        self.image_paths = []
#        self.labels = []
#        self.classes = ['glioma', 'meningioma']  # Nur die beiden gewünschten Klassen
#
#        for class_name in ['glioma', 'meningioma']:  # Explizite Auswahl
#            class_folder = os.path.join(root_dir, class_name)
#            if not os.path.isdir(class_folder):
#                continue
#
#            label = 0 if class_name.lower() == 'glioma' else 1  # Glioma = 0, Meningioma = 1
#
#            for fname in os.listdir(class_folder):
#                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#                    self.image_paths.append(os.path.join(class_folder, fname))
#                    self.labels.append(label)
#
#    def __len__(self):
#        return len(self.image_paths)
#
#    def __getitem__(self, idx):
#        image_tensor = preprocess_image(self.image_paths[idx])
#        label = self.labels[idx]
#        return image_tensor, label, self.image_paths[idx]

#EIgenes Dataset, Unterscheidung no Tumor und Tumor
class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.classes = ['no_tumor', 'tumor']  # Neue Klassennamen
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            # Tumor = 1, No Tumor = 0
            label = 1 if class_name.lower() in ['glioma', 'meningioma'] else 0
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_tensor = preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        return image_tensor, label, self.image_paths[idx]


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
            nn.Linear(128 * 14 * 14, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Verzeichnisse
TRAIN_DIR = "archive/Training"
TEST_DIR = "archive/Testing"

# Trainingsfunktion für Sweep
def train():
    wandb.init(project="Messung-noTu-Tu")
    config = wandb.config
    #run_name = wandb.run.name
    run_name = f"trainsample_{config.train_samples}"
    wandb.run.name = run_name

    project_name = wandb.run.project

    # Checkpoint-Verzeichnisstruktur anlegen
    checkpoint_dir = os.path.join("Checkpoints", project_name, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Seed setzen für reproduzierbare Ergebnisse
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset laden
    full_dataset = CustomImageDataset(root_dir=TRAIN_DIR)
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    base_train_size = total_size - val_size

    # Fixe Aufteilung in Training und Validierung
    base_train_dataset, val_dataset = random_split(
        full_dataset,
        [base_train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Trainingsdaten auf gewünschte Anzahl reduzieren
    target_train_samples = min(config.train_samples, len(base_train_dataset))
    reduced_train_dataset, _ = random_split(
        base_train_dataset,
        [target_train_samples, len(base_train_dataset) - target_train_samples],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(reduced_train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    classifier = ImageClassifier().to(device)
    optimizer = Adam(classifier.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10  # Feste Patience für Early Stopping (kann hier angepasst werden)

    for epoch in range(config.epochs):
        classifier.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_incorrect = 0
        val_total = 0

        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_images, val_labels, _ in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                outputs = classifier(val_images)
                loss = loss_fn(outputs, val_labels)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions = torch.argmax(outputs, dim=1)

                val_correct += (predictions == val_labels).sum().item()
                val_incorrect += (predictions != val_labels).sum().item()
                val_total += val_labels.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_correct / val_total) * 100

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        auc_score = roc_auc_score(all_labels, all_probs)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Sens: {sensitivity:.2f} | "
            f"Spec: {specificity:.2f} | "
            f"AUC: {auc_score:.2f} | "
            f"✅ Correct: {val_correct} | "
            f"❌ Incorrect: {val_incorrect} | "
            f"📦 Total: {val_total}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_correct": val_correct,
            "val_incorrect": val_incorrect,
            "val_total": val_total,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc_score,
            "train_samples_used": target_train_samples,
            "val_probs_labels": wandb.Table(data=list(zip(all_probs, all_labels)), columns=["prob", "label"])
        })

        # Early Stopping Check
        if avg_val_loss < best_val_loss - 1e-4:  # kleine Toleranz für Verbesserung
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Bestes Modell speichern
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(classifier.state_dict(), best_model_path)
            print(f"🏅 Best model saved: {best_model_path}")
            wandb.save(best_model_path)
            wandb.log({"best_model_path": best_model_path})

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"🚨 Early stopping triggered after {epoch+1} epochs with no improvement in val_loss.")
            break

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03}.pt")
            torch.save(classifier.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            wandb.save(checkpoint_path)
            wandb.log({"checkpoint_path": checkpoint_path})

    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(classifier.state_dict(), final_model_path)
    print(f"✅ Final model saved: {final_model_path}")

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
        for images, labels, _ in test_loader:
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
    sweep_id = wandb.sweep(sweep_config, project="Messung-noTu-Tu")
    wandb.agent(sweep_id, function=train)

