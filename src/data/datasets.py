import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class BaseDataset(Dataset):
    """基础数据集类"""

    def __init__(self, data_root, csv_path, transform=None):
        """
        Args:
            data_root (str): 数据根目录
            csv_path (str): 标签文件路径
            transform (callable): 数据预处理方法
        """
        self.data_root = data_root
        self.transform = transform
        self._load_metadata(csv_path)
        self.label_map = {"REAL": 1, "FAKE": 0}

    def _load_metadata(self, csv_path):
        """加载数据路径和标签"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在于 {csv_path}")

        df = pd.read_csv(csv_path)
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()

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
        return image, label  # 返回图像张量和标签

    def num_of_real_and_fake(self):
        real = 0
        fake = 0
        for label in self.labels:
            if label == "REAL":
                real += 1
            elif label == "FAKE":
                fake += 1
        return real, fake