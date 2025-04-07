import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import *
from src.data.datasets import TestDataset, BaseDataset
from src.utils.load_model import load_model

parser = argparse.ArgumentParser(description="Train a model")
# parser.add_argument('--model_class', type=str, help='The class name of the model to train')
parser.add_argument('--test_dataset', type=str, help='The dataset of the model to test')
# --test_dataset
args = parser.parse_args()


test_dataset = f'{args.test_dataset}_labels.csv'
print(f'===> Load {test_dataset} start!')

test_data = TestDataset(
    data_root=DATA_ROOT,
    csv_path=test_dataset,
    # csv_path=TEST_DATASET,
    # transform=train_transform,
)

real, fake = test_data.num_of_real_and_fake()
print(f"Real:{real}")
print(f"Fake:{fake}")

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    collate_fn=BaseDataset.collate_fn
)
print(f'===> Load {test_dataset} done!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试函数
def evaluate_model(model, test_loader):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch")
        for data_dict in pbar:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            pred_dict = model(data_dict, inference=True)
            probs = torch.softmax(pred_dict['cls'], dim=1)[:, 1].cpu().numpy()
            labels = data_dict['label'].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            model.prob.append(probs)
            model.label.append(labels)
            model.correct += (preds == labels).sum()
            model.total += len(labels)
            total += labels.size(0)
            _, predicted = torch.max(pred_dict['cls'].data, 1)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({
                'acc': 100 * correct / total
            }, refresh=False)
        metrics = model.get_test_metrics()
    return metrics

# 测试所有变体
models = {
    "Full Model": load_model('DFACLIP', DEVICE),
    "No Global": load_model('DFACLIP_NoGlobal', DEVICE),
    "No FG": load_model('DFACLIP_NoFG', DEVICE),
    "No IFC": load_model('DFACLIP_NoIFC', DEVICE)
}

# 加载预训练权重（假设路径一致）
pretrained_path = "weights/best_DFACLIP_model.pth"
for name, model in models.items():
    print(f"===> start eval the model: {model._get_name()}")
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)
    metrics = evaluate_model(model, test_loader)
    print(f"{name}: {metrics}")

