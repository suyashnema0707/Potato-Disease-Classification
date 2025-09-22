# ==============================================================================
# Step 1: Setup, Imports, and Data Preparation (Common for all models)
# ==============================================================================
import os
import time
import copy
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from google.colab import files, drive

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mount Google Drive
drive.mount('/content/drive')

# Setup Kaggle API
print("Please upload your kaggle.json file")
files.upload()
!mkdir - p
~ /.kaggle
!cp
kaggle.json
~ /.kaggle /
!chmod
600
~ /.kaggle / kaggle.json

# ==============================================================================
# Step 2: Download, Extract, and Organize the NEW Dataset
# ==============================================================================
!kaggle
datasets
download - d
faysalmiah1721758 / potato - dataset - p / content /

print("Unzipping dataset...")
!unzip - qo / content / potato - dataset.zip - d / content /
print("Dataset unzipped.")

# --- FIX: CREATE A CLEAN DIRECTORY FOR THE DATASET ---
# ImageFolder requires the root directory to only contain class folders.
# We will create a new directory and move the class folders into it.
print("Organizing dataset...")
!mkdir / content / potato_data
!mv / content / Potato___Early_blight / content / potato_data /
!mv / content / Potato___healthy / content / potato_data /
!mv / content / Potato___Late_blight / content / potato_data /
print("Dataset organized.")

# --- DIAGNOSTIC STEP: Check the contents of the new directory ---
print("\n--- Content of the new clean data directory ---")
!ls / content / potato_data
print("-----------------------------------------------------\n")

# --- THIS IS THE CORRECTED DATA DIRECTORY PATH ---
# Point to the new, clean directory we just created.
data_dir = '/content/potato_data/'
print(f"Data directory: {data_dir}")

# ==============================================================================
# Step 3: Define Transformations and Load Data
# ==============================================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dataset = ImageFolder(data_dir)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply transforms to the datasets
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"Class names: {class_names}")
print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")
print("\n" + "=" * 50)
print("Data Preparation Complete.")
print("=" * 50 + "\n")


# ==============================================================================
# Step 4: Generic Training Loop and Model Definitions
# ==============================================================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc, history


results = {}


# --- Model 1: Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Model 2: Optimized CNN ---
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==============================================================================
# Step 5: Training and Evaluation of All Three Models
# ==============================================================================
# --- Train Model 1 ---
print("\n" + "=" * 50)
print("Training Model 1: Simple CNN from Scratch")
print("=" * 50 + "\n")
model_simple_cnn = SimpleCNN().to(device)
optimizer_simple = optim.Adam(model_simple_cnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler_simple = lr_scheduler.StepLR(optimizer_simple, step_size=7, gamma=0.1)
_, best_acc_simple, _ = train_model(model_simple_cnn, criterion, optimizer_simple, exp_lr_scheduler_simple,
                                    num_epochs=15)
results['Simple CNN'] = best_acc_simple.item()

# --- Train Model 2 ---
print("\n" + "=" * 50)
print("Training Model 2: Optimized CNN from Scratch")
print("=" * 50 + "\n")
model_optimized_cnn = OptimizedCNN().to(device)
optimizer_optimized = optim.Adam(model_optimized_cnn.parameters(), lr=0.001)
exp_lr_scheduler_optimized = lr_scheduler.StepLR(optimizer_optimized, step_size=7, gamma=0.1)
_, best_acc_optimized, _ = train_model(model_optimized_cnn, criterion, optimizer_optimized, exp_lr_scheduler_optimized,
                                       num_epochs=15)
results['Optimized CNN'] = best_acc_optimized.item()

# --- Train Model 3 ---
print("\n" + "=" * 50)
print("Training Model 3: Transfer Learning (ResNet18)")
print("=" * 50 + "\n")
model_tl = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
for param in model_tl.parameters():
    param.requires_grad = False
num_ftrs = model_tl.fc.in_features
model_tl.fc = nn.Linear(num_ftrs, num_classes)
model_tl = model_tl.to(device)
optimizer_tl = optim.Adam(model_tl.fc.parameters(), lr=0.001)
exp_lr_scheduler_tl = lr_scheduler.StepLR(optimizer_tl, step_size=7, gamma=0.1)
best_model_tl, best_acc_tl, _ = train_model(model_tl, criterion, optimizer_tl, exp_lr_scheduler_tl, num_epochs=10)
results['Transfer Learning'] = best_acc_tl.item()

# ==============================================================================
# Step 6: Final Comparison and Saving the Best Model
# ==============================================================================
print("\n" + "=" * 50)
print("         Comparison of Model Performance         ")
print("=" * 50 + "\n")

for model_name, acc in results.items():
    print(f"- {model_name}: Best Validation Accuracy = {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest performing model is: {best_model_name}")

if best_model_name == 'Transfer Learning':
    model_to_save = best_model_tl
elif best_model_name == 'Optimized CNN':
    model_to_save = model_optimized_cnn
else:
    model_to_save = model_simple_cnn

save_path = '/content/drive/My Drive/Colab_Models/best_potato_disease_model_new_dataset.pth'
model_dir = os.path.dirname(save_path)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(model_to_save.state_dict(), save_path)
print(f"Best model ('{best_model_name}') saved successfully to: {save_path}")
!ls - lh
'/content/drive/My Drive/Colab_Models/'

