from datetime import datetime

from ..Trainer import Trainer
from ..config import *

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
        f.write(f"CRITERION: {CRITERION.__class__.__name__}\n")
        f.write(f"OPTIMIZER_CLASS: {OPTIMIZER_CLASS}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"DATASET_PATH: {DATASET_PATH}\n")
        f.write(f"DATA_ROOT: {DATA_ROOT}\n")

def save_exp_results(trainer:Trainer, best_test_acc, best_test_loss):
    # 保存训练历史图
    plot_path = os.path.join(EXP_DIR, "training_history.png")
    trainer.plot_or_save_history(save_path=plot_path)

    # 保存实验结果
    result_path = os.path.join(EXP_DIR, "results.txt")
    with open(result_path, "w") as f:
        f.write(f"Best Test Loss: {best_test_loss:.4f}\n")
        f.write(f"Best Test Accuracy: {best_test_acc:.2f}%\n")