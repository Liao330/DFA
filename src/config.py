import os
from datetime import datetime
import torch

# 判断操作系统类型
if os.name == "nt":  # Windows
    NUM_WORKERS = 0
    MODEL_DOWNLOAD_ROOT = r"E:\github_code\Unnamed1\weights"
elif os.name == "posix":  # Linux 或 macOS
    NUM_WORKERS = 8
    MODEL_DOWNLOAD_ROOT = r"/8lab/lljjff/Unnamed/weights"

GPU_COUNT = torch.cuda.device_count()
# print(f"Detected GPUs: {GPU_COUNT}")

# 基本配置
BATCH_SIZE = 32
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASS = 2
SEED = 706
USE_GPU_NUM = min(1, GPU_COUNT)

if USE_GPU_NUM > 1:
    is_DDP = True
else:
    is_DDP = False

# 模型配置
MODEL_CLASS = 'Unknow' # 与src/models中的文件名保持一致
NUM_EPOCHS = 1

# 训练配置
WEIGHTS = torch.tensor([1.0, 5.6]).to(DEVICE)
CRITERION = 'CrossEntropyLoss'
OPTIMIZER_CLASS = 'Adam'
LEARNING_RATE = 0.0002

# 数据集配置 相对于项目根目录
DATASET_PATH = 'new_global_labels.csv'
DATA_ROOT = ''
TEST_DATASET = 'DFDC_labels.csv'
# TEST_DATASET = 'DFDCP_labels.csv'

# tensorboard日志配置
now = datetime.now()
idx = f"{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}"
LOG_DIR = f'src/logs/{MODEL_CLASS}/experiment{idx}'

# 实验记录配置
timestamp = f"{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}"
EXP_DIR = f"experiments/exp_{MODEL_CLASS}_{timestamp}"
os.makedirs(EXP_DIR, exist_ok=True)