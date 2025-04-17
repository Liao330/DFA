import sys

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from tqdm import tqdm
from src.config import EXP_DIR, WEIGHTS, CRITERION, SEED, GPU_COUNT, LEARNING_RATE, is_DDP, is_DALI
from src.utils.visualize import plot_loss_curve, plot_acc_curve, plot_confusion_matrix, plot_f1_score, plot_roc


# 封装Train类
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, device_ids, rank):
        self.model = model.to(device)
        self.model_class = model._get_name()
        self.set_seed(SEED)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        # self.device = f'cuda:{device_ids[0]}' # 多gpu
        self.device_ids = device_ids
        self.rank = rank
        if criterion == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss(WEIGHTS)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-4)
        elif optimizer == "SGD":
            self.optimizer = torch.optimizer.SGD(model.parameters(), LEARNING_RATE, weight_decay=1e-4)
        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
                        }

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_raw_model(self):
        """获取原始模型（兼容 DDP 和非 DDP 模式）"""
        if isinstance(self.model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            return self.model.module
        return self.model

    def _unify_data_format(self, data_dict):
        if is_DALI:
            # DALI 返回的是一个列表，列表中每个元素是一个字典，包含张量
            dali_output = data_dict[0]  # DALIGenericIterator 返回的批次数据
            image = dali_output["image"]
            landmark = dali_output["landmark"] if "landmark" in dali_output else torch.zeros(
                image.shape[0], 81, 2, device=self.device
            )
            label = dali_output["label"]
            return {
                "image": image,
                "landmark": landmark,
                "label": label
            }
        else:
            return data_dict

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)
        test_points = [1.0]
        tested_points = set()
        test_metrics = {}
        model_path = ''
        scaler = torch.amp.GradScaler(self.device)

        # 在 DDP 模式下，只在 rank 0 显示进度条
        pbar = tqdm(self.train_loader, desc="Training", unit="batch", disable=self.device != 'cuda:0' and is_DDP)
        for batch_idx, data_dict in enumerate(pbar):
            # print(f"Rank {self.device}: Processing batch {batch_idx + 1}/{total_batches}")
            progress = (batch_idx + 1) / total_batches
            for point in test_points:
                if progress >= point and point not in tested_points:
                    print(f"\nRank {self.device}: Reached {point * 100}% progress, running test...")
                    test_metrics, model_path = self.test_epoch()
                    if self.device == 'cuda:0' and is_DDP:
                        print(f"Test at {point * 100}% progress: Loss: {test_metrics['loss']:.4f}, "
                              f"Acc: {test_metrics['acc']:.2f}%, "
                              f"Precision: {test_metrics['precision']:.4f}, "
                              f"Recall: {test_metrics['recall']:.4f}, "
                              f"F1: {test_metrics['test_f1']:.4f}, "
                              f"AUC: {test_metrics['roc_auc']:.4f}")
                    if model_path:
                        print(f"Model saved at: {model_path}")
                    tested_points.add(point)

            self.model.train()

            # 统一DALI和普通模式的数据结构
            data_dict = self._unify_data_format(data_dict)

            data_dict = {key: value.to(self.device) for key, value in data_dict.items()}
            img_inputs, lm_inputs, labels = data_dict["image"], data_dict["landmark"], data_dict["label"]

            if torch.isnan(img_inputs).any() or torch.isinf(img_inputs).any():
                print(f"Rank {self.device}: Warning: NaN or Inf detected in images")
            if torch.isnan(lm_inputs).any() or torch.isinf(lm_inputs).any():
                print(f"Rank {self.device}: Warning: NaN or Inf detected in landmarks")

            self.optimizer.zero_grad()
            with torch.amp.autocast(self.device):
                if self.model_class == 'DFACLIP':
                    # print(f"Rank {self.device}: Starting model forward pass...")
                    pred_dict = self.model(data_dict)
                    # print(f"Rank {self.device}: Forward pass completed.")
                    outputs = pred_dict['cls']
                    loss_dict = self._get_raw_model().get_losses(data_dict, pred_dict)
                    loss = loss_dict['overall']
                else:
                    outputs = self.model(img_inputs)
                    loss = self.criterion(outputs, labels)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Rank {self.device}: Warning: NaN or Inf detected in model outputs")
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Rank {self.device}: Warning: NaN or Inf detected in loss")
                break

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1e-5),
                'acc': 100 * correct / total
            })
            sys.stdout.flush()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        train_dict = {
            'epoch_loss': epoch_loss,
            'epoch_acc': epoch_acc
        }
        return train_dict, test_metrics, model_path

    def test_epoch(self):
        # 确保测试时只在 rank 0 保存模型和记录日志
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_true = []
        all_pred = []
        all_prob = []

        with torch.no_grad():
            if self.rank == 0:
                print(f"test num: {len(self.test_loader)}")
            pbar = tqdm(self.test_loader, desc="Testing", unit="batch", disable=self.device != 'cuda:0' and is_DDP)
            for data_dict in pbar:
                img_inputs, lm_inputs, labels = data_dict.values()
                data_dict = {key: value.to(self.device) for key, value in data_dict.items()}
                img_inputs, lm_inputs, labels = img_inputs.to(self.device), lm_inputs.to(self.device), labels.to(
                    self.device)

                if self.model_class == 'DFACLIP':
                    pred_dict = self.model(data_dict)
                    outputs = pred_dict['cls']
                    loss_dict = self._get_raw_model().get_losses(data_dict, pred_dict)
                    loss = loss_dict['overall']
                    if loss.isnan().any():
                        print(f"Rank {self.device}: NaN detected in loss")
                        print(pred_dict)
                else:
                    outputs = self.model(img_inputs)
                    loss = self.criterion(outputs, labels)

                all_true.append(labels.cpu())
                all_pred.append(torch.argmax(outputs.data, dim=1).cpu())
                all_prob.append(outputs.data.cpu())

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1e-5),
                    'acc': 100 * correct / total
                }, refresh=False)

        # 使用 torch.cat 拼接张量
        all_true = torch.cat(all_true, dim=0).cpu().numpy()
        all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
        all_prob = torch.cat(all_prob, dim=0).cpu().numpy()
        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100 * correct / total
        self.history['test_loss'].append(epoch_loss)
        self.history['test_acc'].append(epoch_acc)

        all_true_flat = np.array(all_true).flatten()
        all_pred_flat = np.array(all_pred).flatten()

        cm = confusion_matrix(all_true_flat, all_pred_flat)
        if self.device == 'cuda:0':
            print("Confusion Matrix:\n", cm)
        precision = precision_score(all_true_flat, all_pred_flat, average='macro')
        recall = recall_score(all_true_flat, all_pred_flat, average='macro')
        fpr, tpr, _ = roc_curve(all_true_flat, all_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        test_f1 = f1_score(all_true_flat, all_pred_flat, average='macro')

        # 下面代码有bug 暂时不在训练时计算v_auc
        # img_names = self.test_loader.dataset.data_dict['image']
        # if type(img_names[0]) is not list:
        #     # calculate video-level auc for the frame-level methods.
        #     v_auc, _ = self.get_video_metrics(img_names, all_pred_flat, all_true_flat)
        # else:
        #     # video-level methods
        #     v_auc = roc_auc
        v_auc = 1111111111111111

        dic = {
            'loss': epoch_loss,
            'acc': epoch_acc,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'test_f1': test_f1,
            'cm': cm,
            'video_auc': v_auc
        }

        if self.device == 'cuda:0':
            print("Keys in state_dict before saving:", list(self.model.state_dict().keys()))
        model_path = EXP_DIR + '/' + f'best_{self._get_raw_model()._get_name()}_model.pth'
        if (self.device == 'cuda:0' or is_DDP == False) and (
                len(self.history['test_acc']) == 0 or epoch_acc >= max(self.history['test_acc'])):
            print(f"save the best model success")
            torch.save(self._get_raw_model().state_dict(), model_path)
        return dic, model_path

    def get_video_metrics(self, image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred, pos_label=1)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer

    def plot_or_save_history(self):
        # 绘制损失曲线
        plot_loss_curve(EXP_DIR, self.history['train_loss'], self.history['test_loss'])
        # 绘制准确率曲线
        plot_acc_curve(EXP_DIR, self.history['train_acc'], self.history['test_acc'])

