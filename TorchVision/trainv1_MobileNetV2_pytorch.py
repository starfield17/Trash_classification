import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing parameters
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_CLASSES = 40

# Training target parameters
TARGET_TRAIN_ACC = 99.0  # Target training accuracy
TARGET_VAL_ACC = 95.0    # Target validation accuracy
MAX_EPOCHS = 200         # Maximum training epochs
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

class GarbageDataset(Dataset):
    def __init__(self, root_dir, txt_file, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        
        # Read file list
        with open(os.path.join(root_dir, txt_file), 'r') as f:
            self.data = []
            for line in f:
                img_path, label = line.strip().split()
                img_path = img_path.lstrip('./')
                self.data.append((img_path, int(label)))
        
        self.num_samples = len(self.data)
        print(f"Loaded {self.num_samples} samples")

    def preprocess_image(self, img_path):
        # Read and preprocess image
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.is_training:
            if torch.isnan(img).any():
                print(f"Warning: Image {img_path} contains NaN values")
                return torch.zeros(3, IMG_SIZE, IMG_SIZE)

            try:
                # Data augmentation
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, [2])  # Horizontal flip
                
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, [1])  # Vertical flip
                
                k = torch.randint(0, 4, (1,)).item()
                img = torch.rot90(img, k, [1, 2])  # Random rotation
                
                brightness_factor = 1.0 + torch.rand(1).item() * 0.4 - 0.2
                img = img * brightness_factor  # Brightness adjustment
                
                img = torch.clamp(img, 0.0, 1.0)  # Clamp values
                
            except Exception as e:
                print(f"Error during data augmentation for {img_path}: {str(e)}")
                return torch.zeros(3, IMG_SIZE, IMG_SIZE)

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = self.preprocess_image(img_path)
        return img, label

# Create MobileNetV2 model
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GarbageClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Freeze pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify final classification layer
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.last_channel, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    epoch = 0
    
    while epoch < MAX_EPOCHS:
        epoch += 1
        print(f'\nStarting Epoch {epoch}/{MAX_EPOCHS}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch}/{MAX_EPOCHS}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'Saving new best model, validation accuracy: {val_acc:.2f}%')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, 'best_model.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Check if targets are reached
        if train_acc >= TARGET_TRAIN_ACC and val_acc >= TARGET_VAL_ACC:
            print(f'\nTarget accuracy reached! Training accuracy: {train_acc:.2f}%, Validation accuracy: {val_acc:.2f}%')
            break
            
        # Early stopping check
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'\n{EARLY_STOPPING_PATIENCE} epochs without improvement, stopping training')
            break
            
        # Check for overfitting
        if train_acc > 98 and val_acc < 80:
            print('\nPossible overfitting detected, stopping training')
            break

def main():
    # Create datasets and data loaders
    train_dataset = GarbageDataset('garbage', 'train.txt', is_training=True)
    val_dataset = GarbageDataset('garbage', 'validate.txt', is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    model = GarbageClassifier(NUM_CLASSES).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader)
    
    # Export as TorchScript model
    model.eval()
    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('garbage_classifier.pt')

if __name__ == '__main__':
    main()
