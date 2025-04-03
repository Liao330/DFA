from torch.utils.data import DataLoader, random_split, ConcatDataset
from src.data.datasets import BaseDataset, TrainDataset
from src.data.transforms import build_transforms
from src.config import *

from torch.utils.data.distributed import DistributedSampler
import os



def get_infer_data(infer_dataset):
    # 划分数据集
    infer_size = int(0.8 * len(infer_dataset))
    test_size = len(infer_dataset) - infer_size
    infer_data, test_data = random_split(
        infer_dataset, [infer_size, test_size]
    )
    return infer_data, test_data

def get_dataloader(rank, world_size):

    if rank == 0:
        print("当前工作目录:", os.getcwd())

    train_transform, test_transform = build_transforms(MEAN, STD)

    custom_data = TrainDataset(
        data_root=DATA_ROOT,
        csv_path=DATASET_PATH,
        # transform=train_transform,
    )
    # infer_data = BaseDataset(
    #     data_root=DATA_ROOT,
    #     csv_path=INFER_DATASET,
    #     # transform=train_transform,
    # )
    # DFDC_train_data, DFDC_test_data = get_infer_data(infer_data)
    # 划分数据集
    # train_size = int(0.8 * len(custom_data))
    # test_size = len(custom_data) - train_size
    # # 随机划分数据集
    # train_dataset, test_dataset = random_split(
    #     custom_data, [train_size, test_size]
    # )
    total_size = len(custom_data)

    # 计算训练集、验证集和测试集的大小
    train_size = int(0.7 * total_size)  # 训练集占70%
    val_size = int(0.1 * total_size)  # 验证集占10%
    test_size = total_size - train_size - val_size  # 测试集占剩余的20%

    # 随机划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        custom_data, [train_size, val_size, test_size]
    )
    if rank == 0:
        print(f"REAL NUMS:{custom_data.num_of_real_and_fake()[0] }")
        print(f"FAKE NUMS:{custom_data.num_of_real_and_fake()[1] }")

    # 根据是否使用 DDP 设置 train_loader
    if is_DDP and rank is not None and world_size is not None:
        # DDP 模式：使用 DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # DDP 模式下由 sampler 控制 shuffle
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            sampler=train_sampler,
            collate_fn=BaseDataset.collate_fn
        )
    else:
        # 非 DDP 模式：普通 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            collate_fn=BaseDataset.collate_fn
        )

    # test_loader 不需要 DistributedSampler，但需要确保每个进程的测试数据一致
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 测试时不打乱数据
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=BaseDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证时不打乱数据
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=BaseDataset.collate_fn
    )
    return train_loader, test_loader, val_loader

# a, b = get_dataloader()
# print(len(a), len(b))