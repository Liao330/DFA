from torch.utils.data import DataLoader, random_split
from src.data.datasets import BaseDataset
from src.data.transforms import build_transforms
from src.config import *
import os

def get_dataloader():
    print("当前工作目录:", os.getcwd())

    train_transform, test_transform = build_transforms(MEAN, STD)

    custom_data = BaseDataset(
        data_root=DATA_ROOT,
        csv_path=DATASET_PATH,
        transform=train_transform
    )

    # 划分数据集
    train_size = int(0.8 * len(custom_data))
    test_size = len(custom_data) - train_size
    train_dataset, test_dataset = random_split(
        custom_data, [train_size, test_size]
    )
    test_dataset.dataset.transform = test_transform

    print(f"REAL NUMS:{custom_data.num_of_real_and_fake()[0]}")
    print(f"FAKE NUMS:{custom_data.num_of_real_and_fake()[1]}")

    # 创建Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, test_loader