from PIL import Image
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Manual transform
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return torch.tensor(image, dtype=torch.float32)

# Custom dataset class (replaces torchvision.datasets.ImageFolder)
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

# Load training data
train_dataset = CustomImageDataset(root_dir="archive/Training")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the image classifier model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Create an instance of the image classifier model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
classifier = ImageClassifier().to(device)

# Define the optimizer and loss function
optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in range(100):
    classifier.train()
    for images, labels in train_loader:
        images, labels = images.to(device), torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(classifier.state_dict(), 'model_state.pt')

# Load the model
with open('model_state.pt', 'rb') as f:
    classifier.load_state_dict(load(f))

# Load testing data
test_dataset = CustomImageDataset(root_dir="archive/Testing")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class_names = test_dataset.classes

# Evaluate on test data
classifier.eval()
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = torch.tensor(labels).to(device)

        outputs = classifier(images)
        predicted_index = torch.argmax(outputs, dim=1).item()
        true_index = labels.item()

        predicted_label = class_names[predicted_index]
        true_label = class_names[true_index]

        print(f"[{idx+1}] Predicted: {predicted_label} | True: {true_label}")