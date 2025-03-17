import os
from datetime import datetime
import torch

# 基本配置
BATCH_SIZE = 64
NUM_WORKERS = 0
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 2

# 模型配置
MODEL_CLASS = 'SimpleCNN' # 与src/models中的文件名保持一致
NUM_EPOCHS = 2

# 训练配置
WEIGHTS = torch.tensor([1.0, 5.0]).to(DEVICE)
CRITERION = 'CrossEntropyLoss'
OPTIMIZER_CLASS = 'Adam'
LEARNING_RATE = 1e-4

# 数据集配置 相对于项目根目录
DATASET_PATH = 'global_labels.csv'
DATA_ROOT = ''
INFER_DATASET = 'new_labels.csv'

# tensorboard日志配置
now = datetime.now()
idx = f"{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}"
LOG_DIR = f'src/logs/{MODEL_CLASS}/experiment{idx}'

# 实验记录配置
timestamp = f"{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}"
EXP_DIR = f"experiments/exp_{MODEL_CLASS}_{timestamp}"
os.makedirs(EXP_DIR, exist_ok=True)