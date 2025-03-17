from torch.utils.data import DataLoader, random_split, ConcatDataset
from src.data.datasets import BaseDataset
from src.data.transforms import build_transforms
from src.config import *
import os

def get_infer_data(infer_dataset):
    # 划分数据集
    infer_size = int(0.8 * len(infer_dataset))
    test_size = len(infer_dataset) - infer_size
    infer_data, test_data = random_split(
        infer_dataset, [infer_size, test_size]
    )
    return infer_data, test_data

def get_dataloader():
    print("当前工作目录:", os.getcwd())

    train_transform, test_transform = build_transforms(MEAN, STD)

    custom_data = BaseDataset(
        data_root=DATA_ROOT,
        csv_path=DATASET_PATH,
        transform=train_transform
    )
    infer_data = BaseDataset(
        data_root=DATA_ROOT,
        csv_path=INFER_DATASET,
        transform=train_transform
    )
    DFDC_train_data, DFDC_test_data = get_infer_data(infer_data)
    # 划分数据集
    train_size = int(0.8 * len(custom_data))
    test_size = len(custom_data) - train_size
    # 随机划分数据集
    train_dataset, test_dataset = random_split(
        custom_data, [train_size, test_size]
    )
    # 顺序划分数据集
    # train_dataset = torch.utils.data.Subset(custom_data, range(train_size))
    # test_dataset = torch.utils.data.Subset(custom_data, range(train_size, len(custom_data)))
    test_dataset.dataset.transform = test_transform
    # 合并 DFDC数据集
    test_dataset = ConcatDataset([test_dataset, DFDC_test_data])
    train_dataset = ConcatDataset([train_dataset, DFDC_train_data])

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
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, test_loader

# a, b = get_dataloader()
# print(len(a), len(b))