import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理参数
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 40

class GarbageDataset(Dataset):
    def __init__(self, root_dir, txt_file, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        
        # 读取文件列表
        with open(os.path.join(root_dir, txt_file), 'r') as f:
            self.data = []
            for line in f:
                img_path, label = line.strip().split()
                img_path = img_path.lstrip('./')
                self.data.append((img_path, int(label)))
        
        self.num_samples = len(self.data)
        print(f"加载了 {self.num_samples} 个样本")

    def preprocess_image(self, img_path):
        # 读取和预处理图片
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            # 创建空白图片时直接使用float32类型
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

        # 将 NumPy 数组转换为 PyTorch 张量
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.is_training:
            # 检查 NaN 值
            if torch.isnan(img).any():
                print(f"警告：图片 {img_path} 包含 NaN 值")
                return torch.zeros(3, IMG_SIZE, IMG_SIZE)

            # 数据增强操作
            try:
                # 随机水平翻转
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, [2])
                
                # 随机垂直翻转
                if torch.rand(1) > 0.5:
                    img = torch.flip(img, [1])
                
                # 随机旋转
                k = torch.randint(0, 4, (1,)).item()
                img = torch.rot90(img, k, [1, 2])
                
                # 随机亮度调整
                brightness_factor = 1.0 + torch.rand(1).item() * 0.4 - 0.2
                img = img * brightness_factor
                
                # 确保值在合理范围内
                img = torch.clamp(img, 0.0, 1.0)
                
            except Exception as e:
                print(f"数据增强过程中出错 {img_path}: {str(e)}")
                return torch.zeros(3, IMG_SIZE, IMG_SIZE)

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = self.preprocess_image(img_path)
        return img, label

class GarbageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GarbageClassifier, self).__init__()
        # 可以选择 mobilenet_v3_large 或 mobilenet_v3_small
        self.model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        
        # 冻结预训练层
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 修改最后的分类层
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 1024),
            nn.Hardswish(),  # MobileNetV3使用Hardswish替代ReLU
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
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
        
        # 验证阶段
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # 创建数据集和数据加载器
    train_dataset = GarbageDataset('garbage', 'train.txt', is_training=True)
    val_dataset = GarbageDataset('garbage', 'validate.txt', is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 创建模型
    model = GarbageClassifier(NUM_CLASSES).to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader)
    
    # 导出为TorchScript模型
    model.eval()
    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('garbage_classifier.pt')

if __name__ == '__main__':
    main()
