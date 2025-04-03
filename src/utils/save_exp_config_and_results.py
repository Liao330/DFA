import json

import numpy as np
from datetime import datetime

from src.Trainer import Trainer
from src.config import *

import os
from datetime import datetime

def save_exp_config(model_class):
    """
    保存实验配置到文件，以美观的表格格式输出。

    保存路径：EXP_DIR/config.txt
    """
    # 确保 EXP_DIR 存在
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    # 保存路径
    config_path = os.path.join(EXP_DIR, "config.txt")

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 准备内容
    content = (
        f"{'='*50}\n"
        f"{model_class} Experiment Configuration | Saved at: {current_time}\n"
        f"{'='*50}\n\n"
    )

    # 基本配置
    content += (
        f"{'-'*20} Basic Configuration {'-'*20}\n"
        f"{'Parameter':<20} | {'Value':<30}\n"
        f"{'-'*20}-+-{'-'*30}\n"
        f"{'BATCH_SIZE':<20} | {BATCH_SIZE:<30}\n"
        f"{'NUM_WORKERS':<20} | {NUM_WORKERS:<30}\n"
        f"{'IMAGE_SIZE':<20} | {IMAGE_SIZE:<30}\n"
        # f"{'MEAN':<20} | {MEAN:<30}\n"
        # f"{'STD':<20} | {STD:<30}\n"
        f"{'DEVICE':<20} | {DEVICE:<30}\n"
        f"{'NUM_CLASS':<20} | {NUM_CLASS:<30}\n"
        f"{'SEED':<20} | {SEED:<30}\n"
        f"{'GPU_COUNT':<20} | {GPU_COUNT:<30}\n"
        f"{'USE_GPU_NUM':<20} | {USE_GPU_NUM:<30}\n"
        f"{'IS_DDP':<20} | {is_DDP}\n"
        f"\n"
    )

    # 模型配置
    content += (
        f"{'-'*20} Model Configuration {'-'*20}\n"
        f"{'Parameter':<20} | {'Value':<30}\n"
        f"{'-'*20}-+-{'-'*30}\n"
        # f"{'MODEL_CLASS':<20} | {MODEL_CLASS:<30}\n"
        f"{'NUM_EPOCHS':<20} | {NUM_EPOCHS:<30}\n"
        f"{'MODEL_DOWNLOAD_ROOT':<20} | {MODEL_DOWNLOAD_ROOT:<30}\n"
        f"\n"
    )

    # 训练配置
    content += (
        f"{'-'*20} Training Configuration {'-'*20}\n"
        f"{'Parameter':<20} | {'Value':<30}\n"
        f"{'-'*20}-+-{'-'*30}\n"
        # f"{'WEIGHTS':<20} | {WEIGHTS.tolist():<30}\n"
        f"{'CRITERION':<20} | {CRITERION:<30}\n"
        f"{'OPTIMIZER_CLASS':<20} | {OPTIMIZER_CLASS:<30}\n"
        f"{'LEARNING_RATE':<20} | {LEARNING_RATE:<30.6f}\n"
        f"\n"
    )

    # 数据集配置
    content += (
        f"{'-'*20} Dataset Configuration {'-'*20}\n"
        f"{'Parameter':<20} | {'Value':<30}\n"
        f"{'-'*20}-+-{'-'*30}\n"
        f"{'DATASET_PATH':<20} | {DATASET_PATH:<30}\n"
        f"{'DATA_ROOT':<20} | {DATA_ROOT if DATA_ROOT else 'N/A':<30}\n"
        f"{'INFER_DATASET':<20} | {TEST_DATASET:<30}\n"
        f"\n"
    )

    # 日志和实验配置
    content += (
        f"{'-'*20} Logging and Experiment Configuration {'-'*20}\n"
        f"{'Parameter':<20} | {'Value':<30}\n"
        f"{'-'*20}-+-{'-'*30}\n"
        f"{'LOG_DIR':<20} | {LOG_DIR:<30}\n"
        f"{'EXP_DIR':<20} | {EXP_DIR:<30}\n"
        f"{'Timestamp':<20} | {timestamp:<30}\n"
        f"{'-'*50}\n"
    )

    # 写入文件
    with open(config_path, "w") as f:
        f.write(content)

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

# def save_train_epoch_results(epoch, train_loss, train_acc):
#     # 保存实验结果
#     result_path = os.path.join(EXP_DIR, "results.txt")
#     with open(result_path, "a") as f:
#         f.write(f"{'-'*22} EPOCH: {epoch + 1} {'-'*22}\n")
#         f.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n")
#         f.write("-" * 50 + '\n')


def save_epoch_results(epoch, train_loss, train_acc, test_loss, test_acc, precision, recall, roc_auc, test_f1, cm, video_auc):
    """
    保存测试阶段的实验结果到文件，以美观的表格格式输出。

    Args:
        epoch (int): 当前 epoch 编号
        test_loss (float): 测试损失
        test_acc (float): 测试准确率
        precision (float): 测试精确率
        recall (float): 测试召回率
        roc_auc (float): 测试 ROC-AUC 分数
        test_f1 (float): 测试 F1 分数
        cm (np.ndarray): 混淆矩阵
    """
    # 确保 EXP_DIR 存在
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    # 保存路径
    result_path = os.path.join(EXP_DIR, "results.txt")

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 准备表格内容
    header = (
        f"{'='*50}\n"
        f"Epoch {epoch} Exp Results | Time: {current_time}\n"
        f"{'='*50}\n"
    )

    # 指标表格
    metrics = (
        f"{'Metric':<15} | {'Value':<10}\n"
        f"{'-'*15}-+-{'-'*10}\n"
        f"{'Train Loss':<15} | {train_loss:<10.4f}\n"
        f"{'Train Acc':<15} | {train_acc:<10.4f}\n"
        f"{'Test Loss':<15} | {test_loss:<10.4f}\n"
        f"{'Test Acc':<15} | {test_acc:<10.4f}\n"
        f"{'Precision':<15} | {precision:<10.4f}\n"
        f"{'Recall':<15} | {recall:<10.4f}\n"
        f"{'ROC-AUC':<15} | {roc_auc:<10.4f}\n"
        f"{'F1 Score':<15} | {test_f1:<10.4f}\n"
        f"{'Video-AUC':<15} | {video_auc:<10.4f}\n"
    )

    # 混淆矩阵
    cm_str = "Confusion Matrix:\n" + str(cm).replace('\n', '\n    ') + "\n"

    # 组合所有内容
    content = header + metrics + cm_str + f"{'='*50}\n\n"

    # 写入文件
    with open(result_path, "a") as f:
        f.write(content)


def save_infer_results(infer_acc, precision, recall, roc_auc, infer_f1, cm, video_auc):
    # 确保 EXP_DIR 存在
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    # 保存路径
    result_path = os.path.join(EXP_DIR, "results.txt")

    # 指标表格
    metrics = (
        f"{'Infer Metric':<15} | {'Value':<10}\n"
        f"{'-' * 15}-+-{'-' * 10}\n"
        f"{'Infer Acc':<15} | {infer_acc:<10.4f}\n"
        f"{'Precision':<15} | {precision:<10.4f}\n"
        f"{'Recall':<15} | {recall:<10.4f}\n"
        f"{'ROC-AUC':<15} | {roc_auc:<10.4f}\n"
        f"{'F1 Score':<15} | {infer_f1:<10.4f}\n"
        f"{'Video-AUC':<15} | {video_auc:<10.4f}\n"
    )

    # 混淆矩阵
    cm_str = "Confusion Matrix:\n" + str(cm).replace('\n', '\n    ') + "\n"

    # 组合所有内容
    content = metrics + cm_str + f"{'=' * 50}\n\n"

    # 写入文件
    with open(result_path, "a") as f:
        f.write(content)