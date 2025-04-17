import argparse
import os
import sys
import torch
# 获取 src/main.py 的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（/usr/lljjff/Unnamed1）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import logging

from src.Trainer import Trainer
from src.infer import infer
from src.utils.save_exp_config_and_results import save_history_values, save_epoch_results, save_infer_results
from src.utils.load_model import load_model
from src.utils.save_exp_config_and_results import save_exp_config, save_exp_plot
from src.utils.visualize import imshow_grid, print_test_epoch_result
from src.data.loaders import get_dataloader
from src.utils.logger import ExperimentLogger
from src.config import *

def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--model_class', type=str, help='The class name of the model to train')
    args = parser.parse_args()
    return args

def main():
    """主训练函数，支持 DDP 和 DataParallel 模式"""
    # 从环境变量获取 rank 和 world_size
    global model_path
    args = parse_args()

    model_class = args.model_class

    if is_DDP:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = f'cuda:{rank}'
    else:
        rank = 0
        world_size = 1
        device = DEVICE

    setup_logging(rank)

    try:
        # 获取dataloader
        train_loader, test_loader, val_loader = get_dataloader(rank, world_size)
        if rank == 0:
            logging.info(f"Total batches in train_loader: {len(train_loader)}")

        # 获取GPU数量
        GPU_COUNT = torch.cuda.device_count()
        if rank == 0:
            logging.info(f"Total GPUs available: {GPU_COUNT}")
            logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        # 定义设备列表
        device_ids = list(range(min(USE_GPU_NUM, GPU_COUNT)))
        if rank == 0:
            logging.info(f"The device_ids is: {device_ids}")

        # 定义模型
        model = load_model(model_class, device)

        if is_DDP:
            # DDP 模式
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            dist.barrier()
        else:
            # DataParallel 模式
            if len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.to(device)

        trainer = Trainer(model, train_loader, test_loader, CRITERION, OPTIMIZER_CLASS, device, device_ids, rank)

        # 创建tensorboard Logger
        Logger = ExperimentLogger(LOG_DIR)

        if rank == 0:
            logging.info(f"use the model {model_class}")
            save_exp_config(model_class)
        print("=== begin to train ===")
        for epoch in range(NUM_EPOCHS):
            if is_DDP:
                # DDP 模式下，每个 epoch 需要重新设置 sampler
                train_loader.sampler.set_epoch(epoch)

            if rank == 0:
                logging.info(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
            train_dict, test_dict, model_path = trainer.train_epoch()
            train_loss, train_acc = train_dict['epoch_loss'], train_dict['epoch_acc']

            Logger.log_metrics({'train_loss': train_loss}, epoch)
            Logger.log_metrics({'train_acc': train_acc}, epoch)

            test_loss, test_acc, precision, recall, roc_auc, test_f1, cm, video_auc = test_dict.values()

            Logger.log_metrics({'test_loss': test_loss}, epoch)
            Logger.log_metrics({'test_acc': test_acc}, epoch)

            if rank == 0:
                logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                save_epoch_results(epoch, train_loss, train_acc, test_loss, test_acc, precision, recall, roc_auc, test_f1, cm, video_auc)

        if rank == 0: # no val
            # print(f"the save model path is :{model_path}")
            # infer_dic = infer(rank,model_class, val_loader, model_path)
            # infer_acc, precision, recall, roc_auc, infer_f1, cm, video_auc = infer_dic.values()
            # save_infer_results(infer_acc, precision, recall, roc_auc, infer_f1, cm, video_auc)
            pass

        if rank == 0:
            save_exp_plot(trainer)

        Logger.close()
    finally:
        if is_DDP:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
