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

class CustomImageDatasetFromLists(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.classes = ['no_tumor', 'tumor']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        return image_tensor, label, self.image_paths[idx]

    @classmethod
    def from_directory(cls, root_dir):
        image_paths, labels = [], []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = 1 if class_name.lower() in ['glioma', 'meningioma'] else 0
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(label)
        return cls(image_paths, labels)

def split_dataset_by_class(root_dir, train_percent):
    tumor_paths, no_tumor_paths = [], []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = 1 if class_name.lower() in ['glioma', 'meningioma'] else 0
        paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if label == 1:
            tumor_paths.extend([(p, 1) for p in paths])
        else:
            no_tumor_paths.extend([(p, 0) for p in paths])

    random.seed(42)
    random.shuffle(tumor_paths)
    random.shuffle(no_tumor_paths)

    tumor_val_size = int(0.2 * len(tumor_paths))
    no_tumor_val_size = int(0.2 * len(no_tumor_paths))

    tumor_val = tumor_paths[:tumor_val_size]
    tumor_train_full = tumor_paths[tumor_val_size:]
    no_tumor_val = no_tumor_paths[:no_tumor_val_size]
    no_tumor_train = no_tumor_paths[no_tumor_val_size:]

    target_tumor_train_samples = int(train_percent * len(tumor_train_full))
    tumor_train_reduced = tumor_train_full[:target_tumor_train_samples]

    train_combined = tumor_train_reduced + no_tumor_train
    val_combined = tumor_val + no_tumor_val
    random.shuffle(train_combined)
    random.shuffle(val_combined)

    train_paths = [p for p, _ in train_combined]
    train_labels = [l for _, l in train_combined]
    val_paths = [p for p, _ in val_combined]
    val_labels = [l for _, l in val_combined]

    train_dataset = CustomImageDatasetFromLists(train_paths, train_labels)
    val_dataset = CustomImageDatasetFromLists(val_paths, val_labels)

    return train_dataset, val_dataset, len(no_tumor_train), len(tumor_train_reduced)

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
    wandb.init(project="neu Reduzierung-Tu + Balance")
    config = wandb.config

    train_dataset, val_dataset, no_tumor_count, tumor_count = split_dataset_by_class(TRAIN_DIR, config.train_percent)
    total_train_samples = no_tumor_count + tumor_count

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

    print(f"📊 Training with {tumor_count} tumor and {no_tumor_count} no_tumor samples.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    classifier = ImageClassifier().to(device)
    optimizer = Adam(classifier.parameters(), lr=config.learning_rate)

    counts = torch.tensor([no_tumor_count, tumor_count], dtype=torch.float32, device=device)
    weights = 1.0 / counts
    class_weights = weights / weights.sum()
    print(f"Using class weights: {class_weights.cpu().tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

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
        all_probs, all_preds, all_labels = [], [], []

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
            "tumor_train": tumor_count,
            "no_tumor_train": no_tumor_count,
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

    if os.path.exists(best_model_path):
        evaluate_on_test_data(model_path=best_model_path)
    else:
        evaluate_on_test_data(model_path=final_model_path)

    wandb.finish()

def evaluate_on_test_data(model_path):
    print("\n🔍 Testing on separate test set...")
    test_dataset = CustomImageDatasetFromLists.from_directory(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classifier = ImageClassifier().to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    all_probs, all_preds, all_labels = [], [], []
    correct = 0
    incorrect = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted_index = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted_index.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted_index == labels).sum().item()
            incorrect += (predicted_index != labels).sum().item()

    total = correct + incorrect
    accuracy = (correct / total) * 100
    auc_score = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"✅ Correct: {correct} | ❌ Incorrect: {incorrect} | Total: {total}")
    print(f"🎯 Test Accuracy: {accuracy:.2f}% | Sens: {sensitivity:.2f} | Spec: {specificity:.2f} | AUC: {auc_score:.2f}")

    wandb.log({
        "test_accuracy": accuracy,
        "test_correct": correct,
        "test_incorrect": incorrect,
        "test_total": total,
        "test_auc": auc_score,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity
    })

def load_sweep_config(path="sweep_noTu_Tu.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    sweep_config = load_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="neu Reduzierung-Tu + Balance")
    wandb.agent(sweep_id, function=train)
