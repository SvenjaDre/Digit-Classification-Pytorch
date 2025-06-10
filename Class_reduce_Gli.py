import yaml
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import wandb
import random
from sklearn.metrics import roc_auc_score, confusion_matrix

wandb.login(key="4c9968476bcc046f6d8b7204ae7a2ca803e0e0a9")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, class_names=['glioma', 'meningioma']):
        self.image_paths = image_paths
        self.labels = labels
        self.classes = class_names

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        return image_tensor, label, self.image_paths[idx]

    @staticmethod
    def from_directory(directory, class_names=['glioma', 'meningioma']):
        image_paths = []
        labels = []
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
            paths = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            image_paths.extend(paths)
            labels.extend([label] * len(paths))
        return CustomImageDataset(image_paths, labels, class_names)

def split_dataset_by_class(root_dir, train_percent):
    glioma_paths, meningioma_paths = [], []
    for class_name in ['glioma', 'meningioma']:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = 0 if class_name == 'glioma' else 1
        paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if class_name == 'glioma':
            glioma_paths = [(p, label) for p in paths]
        else:
            meningioma_paths = [(p, label) for p in paths]

    random.seed(42)
    random.shuffle(glioma_paths)
    random.shuffle(meningioma_paths)

    # Separat splitten (20 % Validierung)
    glioma_val_size = int(0.2 * len(glioma_paths))
    meningioma_val_size = int(0.2 * len(meningioma_paths))
    glioma_val = glioma_paths[:glioma_val_size]
    glioma_train_full = glioma_paths[glioma_val_size:]
    meningioma_val = meningioma_paths[:meningioma_val_size]
    meningioma_train = meningioma_paths[meningioma_val_size:]

    # Glioma Trainingsmenge reduzieren
    target_glioma_train_samples = int(train_percent * len(glioma_train_full))
    glioma_train_reduced = glioma_train_full[:target_glioma_train_samples]

    train_paths = glioma_train_reduced + meningioma_train
    train_labels = [label for _, label in train_paths]
    train_paths = [path for path, _ in train_paths]

    val_paths = glioma_val + meningioma_val
    val_labels = [label for _, label in val_paths]
    val_paths = [path for path, _ in val_paths]

    train_dataset = CustomImageDataset(train_paths, train_labels)
    val_dataset = CustomImageDataset(val_paths, val_labels)

    return train_dataset, val_dataset, len(glioma_train_reduced), len(meningioma_train)

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

TRAIN_DIR = "archive/Training"
TEST_DIR = "archive/Testing"

def train():
    wandb.init(project="Reduzierung-Gli")
    config = wandb.config

    train_dataset, val_dataset, glioma_train_count, meningioma_train_count = split_dataset_by_class(TRAIN_DIR, config.train_percent)
    total_train_samples = glioma_train_count + meningioma_train_count

    base_checkpoint_dir = os.path.join("Checkpoints", wandb.run.project, f"trainpercent_{config.train_percent}")
    run_id = 1
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"run_{run_id}")
    while os.path.exists(checkpoint_dir):
        run_id += 1
        checkpoint_dir = os.path.join(base_checkpoint_dir, f"run_{run_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.run.name = f"trainpercent_{config.train_percent}_run_{run_id}"
    wandb.run.tags = [f"percent_{config.train_percent}", f"run_{run_id}"]
    wandb.config.run_id = run_id

    print(f"📊 Training with {glioma_train_count} glioma and {meningioma_train_count} meningioma samples.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    classifier = ImageClassifier().to(device)
    optimizer = Adam(classifier.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

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
        val_total = 0
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_images, val_labels, _ in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = classifier(val_images)
                loss = loss_fn(outputs, val_labels)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions = torch.argmax(outputs, dim=1)

                val_correct += (predictions == val_labels).sum().item()
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
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
            f"Sens: {sensitivity:.2f} | Spec: {specificity:.2f} | AUC: {auc_score:.2f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc_score,
            "train_samples_used": total_train_samples,
            "glioma_train": glioma_train_count,
            "meningioma_train": meningioma_train_count,
            "val_probs_labels": wandb.Table(data=list(zip(all_probs, all_labels)), columns=["prob", "label"])
        })

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(classifier.state_dict(), best_model_path)
            wandb.save(best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping after {epoch+1} epochs.")
            break

    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(classifier.state_dict(), final_model_path)

    model_path = best_model_path if os.path.exists(best_model_path) else final_model_path
    evaluate_on_test_data(model_path)
    wandb.finish()

def evaluate_on_test_data(model_path):
    print("\n🔍 Testing on separate test set...")
    test_dataset = CustomImageDataset.from_directory(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_names = test_dataset.classes

    classifier = ImageClassifier().to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    all_probs, all_preds, all_labels = [], [], []
    correct, incorrect = 0, 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            incorrect += (preds != labels).sum().item()

    accuracy = 100 * correct / (correct + incorrect)
    auc_score = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"✅ Correct: {correct} | ❌ Incorrect: {incorrect} | 🎯 Accuracy: {accuracy:.2f}%")
    print(f"Sens: {sensitivity:.2f} | Spec: {specificity:.2f} | AUC: {auc_score:.2f}")

    wandb.log({
        "test_accuracy": accuracy,
        "test_auc": auc_score,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_correct": correct,
        "test_incorrect": incorrect,
        "test_total": correct + incorrect,
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        ),
        "test_probs_labels": wandb.Table(data=list(zip(all_probs, all_labels)), columns=["prob", "label"])
    })

def load_sweep_config(path="sweep_Gli_Men.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    sweep_config = load_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="Reduzierung-Gli")
    wandb.agent(sweep_id, function=train)
