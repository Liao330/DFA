import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.config import *
from src.data.datasets import BaseDataset
from src.data.loaders import get_infer_data
from src.utils.load_model import load_model

def infer(model_path):
    infer_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    infer_dataset = BaseDataset(
        data_root=DATA_ROOT,
        csv_path=INFER_DATASET,
        transform=infer_transform
    )
    infer_data, _ = get_infer_data(infer_dataset)
    infer_loader = DataLoader(
        infer_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = load_model(MODEL_CLASS, DEVICE)
    path_to_model = model_path
    model.load_state_dict(torch.load(path_to_model))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(infer_loader, desc="Infering", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 实时更新进度条信息
                pbar.set_postfix({
                    'acc': 100 * correct / total
                }, refresh=False)
    accuracy = 100 * correct / total
    print(accuracy)
    return accuracy