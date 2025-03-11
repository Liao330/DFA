import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.config import EXP_DIR


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
            torch.save(self.model.state_dict(), f'{EXP_DIR}/best_model.pth')
        return epoch_loss, epoch_acc

    def plot_or_save_history(self, save_path):
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
        plt.savefig(save_path)
        # plt.show()