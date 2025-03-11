import os

import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# 全局自定义参数
batch_size = 64
num_workers = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 定义loss和optimer
criterion = nn.CrossEntropyLoss()
optimizer = 'Adam'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 自定义数据集
class custom_data(Dataset):
    def __init__(self,data_root, paths, labels, transform=None):
        self.data_root = data_root
        self.paths = paths
        self.labels = labels
        self.transform = transform

        self.label_map = {"REAL": 1, "FAKE": 0}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.paths[idx])
        label_str = self.labels[idx]
        # 标签转换
        label = self.label_map[label_str]
        # 加载实际图像数据
        # image = Image.open(img_name).convert('RGB')
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label # 返回图像张量和标签

    def num_of_real_and_fake(self):
        real = 0
        fake = 0
        for label in self.labels:
            if label == "REAL":
                real += 1
            elif label == "FAKE":
                fake += 1
        return real, fake

# 封装Train类
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        elif optimizer == "SGD":
            self.optimizer = torch.optimizer.SGD(model.parameters(), lr=1e-4)
        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(self.train_loader, desc="Training", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 自动混合精度， 但效果不是很好
                # scaler = torch.amp.GradScaler(self.device)
                # with torch.amp.autocast(self.device):
                #     outputs = self.model(inputs)
                #     loss = self.criterion(outputs, labels)
                # self.optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()

                # 统计指标
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 实时更新进度条信息
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1e-5),
                    'acc': 100 * correct / total
                })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        return epoch_loss, epoch_acc

    def test_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100 * correct / total
        self.history['test_loss'].append(epoch_loss)
        self.history['test_acc'].append(epoch_acc)
        # 在test_epoch方法中添加
        if epoch_acc > max(self.history['test_acc']):
            # 后续保存到特定文件夹中
            torch.save(self.model.state_dict(), 'best_model.pth')
        return epoch_loss, epoch_acc

    def plot_history(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['test_loss'], label='Test Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['test_acc'], label='Test Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()


# 定义反归一化函数
def denormalize(tensor):
    tensor = tensor.clone().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)  # 确保值在[0,1]范围内

# 创建网格显示函数
def imshow_grid(loader, num_images=8):
    images, labels = next(iter(loader))
    images = denormalize(images[:num_images])

    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        ax = fig.add_subplot(2, num_images // 2, i + 1)
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title("REAL" if labels[i].item() == 1 else "FAKE")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(256), # 将输入图像大小调整为256x256像素
    transforms.RandomCrop(224), # 从256x256的图像中随机裁剪出224x224的图像区域
    transforms.RandomHorizontalFlip(), # 以一定的概率随机水平翻转图像
    transforms.ToTensor(), # 将PIL图像或NumPy ndarray转换为FloatTensor，并缩放到[0,1]
    transforms.Normalize(mean,std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 加载数据集
head_list = ['path', 'label']
my_data = pd.read_csv('../global_labels.csv', names=head_list)
# print(my_data)  # path labels
data_file = my_data.path.tolist()[1:] # 用于存储自己的数据路径
data_label = my_data.label.tolist()[1:]
# print(len(data_file))
# print(data_label)
custom_data = custom_data(
    data_root="../",
    paths=data_file,
    labels=data_label,
    transform=train_transform
)
# print(custom_data[1])
# print(len(custom_data))
print(f"REAL NUMS:{custom_data.num_of_real_and_fake()[0]}")
print(f"FAKE NUMS:{custom_data.num_of_real_and_fake()[1]}")
# 划分数据集
train_size = int(0.8 * len(custom_data))
test_size = len(custom_data) - train_size
train_dataset, test_dataset  = torch.utils.data.random_split(
    custom_data, [train_size, test_size]
)
test_dataset.dataset.transform = test_transform

# 创建Dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

# 测试
# 检查数据加载是否正常
sample_img, sample_label = next(iter(train_loader))
# print(f'Image shape: {sample_img.shape}')  # 应显示torch.Size([32, 3, 224, 224])
# print(f'Label shape: {sample_label.shape}')  # 应显示torch.Size([32])
# print(f'Label type: {sample_label.dtype}') # 应显示torch.int64

# 可视化样本
imshow_grid(train_loader)

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = SimpleCNN().cuda()
trainer = Trainer(model, train_loader,test_loader, criterion, optimizer, device)

num_epochs = 3
for epoch in range(num_epochs):
    train_loss, train_acc = trainer.train_epoch()
    test_loss, test_acc = trainer.test_epoch()
    print("\n")
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("-" * 50)



