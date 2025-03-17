import json

import numpy as np

from src.Trainer import Trainer
from src.config import *

def save_exp_config():
    # 保存配置文件内容到实验目录
    config_path = os.path.join(EXP_DIR, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"IMAGE_SIZE: {IMAGE_SIZE}\n")
        f.write(f"MEAN: {MEAN}\n")
        f.write(f"STD: {STD}\n")
        f.write(f"DEVICE: {DEVICE}\n")
        f.write(f"MODEL_CLASS: {MODEL_CLASS}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"CRITERION: {CRITERION}\n")
        f.write(f"OPTIMIZER_CLASS: {OPTIMIZER_CLASS}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"DATASET_PATH: {DATASET_PATH}\n")
        f.write(f"DATA_ROOT: {DATA_ROOT}\n")

def save_exp_plot(trainer:Trainer):
    trainer.plot_or_save_history()

def save_history_values(trainer:Trainer):
    # 保存实验结果
    result_path = os.path.join(EXP_DIR, "history.json")
    # 假设 trainer.history 是一个包含 ndarray 的字典
    history_dict = trainer.history.copy()  # 创建历史记录的副本

    # 将 ndarray 转换为列表
    for key, value in history_dict.items():
        if isinstance(value, np.ndarray):
            history_dict[key] = value.tolist()

    # 将历史记录转换为 JSON 格式的字符串
    history_str = json.dumps(history_dict, indent=4)

    # 写入文件
    with open(result_path, 'w') as f:
        f.write(history_str)

def save_train_epoch_results(epoch, train_loss, train_acc):
    # 保存实验结果
    result_path = os.path.join(EXP_DIR, "results.txt")
    with open(result_path, "a") as f:
        f.write(f"{'-'*22} EPOCH: {epoch + 1} {'-'*22}\n")
        f.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n")
        f.write("-" * 50 + '\n')

def save_test_epoch_results(epoch, test_loss, test_acc, precision, recall, roc_auc, test_f1, cm):
    # 保存实验结果
    result_path = os.path.join(EXP_DIR, "results.txt")
    with open(result_path, "a") as f:
        # f.write(f"{'-'*22} EPOCH: {epoch} {'-'*22}\n")
        f.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")
        f.write(f"Test Precision: {precision:.4f} | Test Recall: {recall:.4f}\n")
        f.write(f"Test Roc_Auc: {roc_auc:.4f} | Test F1: {test_f1:.4f}\n")
        f.write(f"Test CM: {cm}\n")
        f.write("-" * 50 + '\n')