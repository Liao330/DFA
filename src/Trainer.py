import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from tqdm import tqdm
from src.config import EXP_DIR, MODEL_CLASS, WEIGHTS, CRITERION
from src.utils.visualize import plot_loss_curve, plot_acc_curve, plot_confusion_matrix, plot_f1_score, plot_roc


# 封装Train类
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        if criterion == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss(WEIGHTS)
        elif criterion == 'BCEWithLogitsLoss':
            self.criterion = torch.nn.BCEWithLogitsLoss(WEIGHTS)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        elif optimizer == "SGD":
            self.optimizer = torch.optimizer.SGD(model.parameters(), lr=1e-4)
        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
                        }

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(self.train_loader, desc="Training", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                # print(outputs.shape, outputs[:5])
                # print(labels.shape, labels[:5])
                if CRITERION == 'BCEWithLogitsLoss':
                    labels = labels.unsqueeze(1).float()
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
                }, refresh=False)

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
        all_true = []
        all_pred = []
        all_prob = []
        with torch.no_grad():
            with tqdm(self.test_loader, desc="Testing", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    all_true.append(labels.cpu().numpy())
                    all_pred.append(torch.argmax(outputs, dim=1).cpu().numpy())
                    all_prob.append(outputs.cpu().numpy())

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # 实时更新进度条信息
                    pbar.set_postfix({
                        'loss': running_loss / (pbar.n + 1e-5),
                        'acc': 100 * correct / total
                    }, refresh=False)
        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100 * correct / total
        self.history['test_loss'].append(epoch_loss)
        self.history['test_acc'].append(epoch_acc)

        all_true_flat = np.array(all_true).flatten()
        all_pred_flat = np.array(all_pred).flatten()
        # print(f"all_true[:5]:{all_true[:5]}")
        # print(f"all_pred[:5]:{all_pred[:5]}")
        # print(f"all_true_single[:5]:{all_true_flat[:5]}")
        # print(f"all_pred_single[:5]:{all_pred_flat[:5]}")

        cm = confusion_matrix(all_true_flat, all_pred_flat)
        print(cm)
        precision = precision_score(all_true_flat, all_pred_flat, average='macro')
        # print(f"Precision: {precision}")
        recall = recall_score(all_true_flat, all_pred_flat, average='macro')
        # print(f"Recall: {recall}")

        fpr, tpr, _ = roc_curve(all_true_flat, all_pred_flat)  # 二分类问题
        roc_auc = auc(fpr, tpr)
        test_f1 = f1_score(all_true_flat, all_pred_flat, average='macro')
        dic = {
            'loss': epoch_loss,
            'acc': epoch_acc,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'test_f1': test_f1,
            'cm': cm,
        }

        # 在test_epoch方法中添加
        if epoch_acc >= max(self.history['test_acc']):
            # 后续保存到特定文件夹中
            print(f"save the best model success")
            model_path = f'{EXP_DIR}/best_{MODEL_CLASS}_model_acc{epoch_acc}.pth'
            torch.save(self.model.state_dict(), model_path)
        else:
            model_path = 'sorry'
        return dic, model_path,

    def plot_or_save_history(self):
        # 绘制损失曲线
        plot_loss_curve(EXP_DIR, self.history['train_loss'], self.history['test_loss'])
        # 绘制准确率曲线
        plot_acc_curve(EXP_DIR, self.history['train_acc'], self.history['test_acc'])

