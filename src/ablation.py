import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import torch
from datetime import datetime
import os

from src.config import *
from src.data.datasets import TestDataset, BaseDataset
from src.utils.load_model import load_model

parser = argparse.ArgumentParser(description="Evaluate models on multiple datasets")
parser.add_argument('--test_datasets', type=str, default="Celeb-DF-v1,Celeb-DF-v2,DFDC,DFDCP",
                    help='Comma-separated list of datasets to test (e.g., Celeb-DF-v1,Celeb-DF-v2)')
args = parser.parse_args()

dataset_names = args.test_datasets.split(',')
print(f"Testing datasets: {dataset_names}")

def get_video_metrics(image, pred, label):
    result_dict = {}
    new_label = []
    new_pred = []
    for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
        s = item[0]
        if '\\' in s:
            parts = s.split('\\')
        else:
            parts = s.split('/')
        a = parts[-2]
        b = parts[-1]
        if a not in result_dict:
            result_dict[a] = []
        result_dict[a].append(item)
    image_arr = list(result_dict.values())
    for video in image_arr:
        pred_sum = 0
        label_sum = 0
        leng = 0
        for frame in video:
            pred_sum += float(frame[1])
            label_sum += int(frame[2])
            leng += 1
        new_pred.append(pred_sum / leng)
        new_label.append(int(label_sum / leng))
    fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
    v_auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return v_auc, v_eer

def get_test_metrics(all_probs, all_labels, correct, total):
    y_pred = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    ap = metrics.average_precision_score(y_true, y_pred)
    acc = correct / total
    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch", ascii=True)
        for data_dict in pbar:
            data_dict = {k: v.to(DEVICE) for k, v in data_dict.items()}
            pred_dict = model(data_dict, inference=True)
            probs = torch.softmax(pred_dict['cls'], dim=1)[:, 1].cpu().numpy()
            labels = data_dict['label'].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.append(probs)
            all_labels.append(labels)
            total += labels.size
            _, predicted = torch.max(pred_dict['cls'].data, 1)
            predicted = predicted.cpu()
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'acc': 100 * correct / total}, refresh=False)

        if not all_probs or not all_labels:
            print("Warning: No predictions or labels collected!")
            return {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'video_eer': 0}

        test_metrics = get_test_metrics(all_probs, all_labels, correct, total)
        img_names = test_loader.dataset.data_dict['image']
        if type(img_names[0]) is not list:
            v_auc, v_eer = get_video_metrics(img_names, np.concatenate(all_probs), np.concatenate(all_labels))
        else:
            v_auc = test_metrics['auc']
            v_eer = 0
        test_metrics['video_auc'] = v_auc
        test_metrics['video_eer'] = v_eer
    return test_metrics

def load_dataset(dataset_name):
    csv_path = f'{dataset_name}_labels.csv'
    print(f'===> Load {csv_path} start!')
    test_data = TestDataset(data_root=DATA_ROOT, csv_path=csv_path)
    real, fake = test_data.num_of_real_and_fake()
    print(f"Real: {real}")
    print(f"Fake: {fake}")
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, drop_last=False, collate_fn=BaseDataset.collate_fn)
    print(f'===> Load {csv_path} done!')
    return test_loader

def save_metrics_to_file(filename, dataset_name, model_name, metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'a') as f:
        f.write(f"\n=== {timestamp} ===\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n" if isinstance(value, (int, float)) else f"  {key}: {value}\n")
        f.write("\n")

models = {
    "Only Global": load_model('DFACLIP_OnlyGlobal', DEVICE),
    "Only FG": load_model('DFACLIP_OnlyFG', DEVICE),
    "No Global": load_model('DFACLIP_NoGlobal', DEVICE),
    "No FG": load_model('DFACLIP_NoFG', DEVICE),
    "No IFC": load_model('DFACLIP_NoIFC', DEVICE),
    "Full Model": load_model('DFACLIP', DEVICE),
}

pretrained_path = "weights/best_DFACLIP_model.pth"
state_dict = torch.load(pretrained_path, map_location=DEVICE)

output_file = "ablation_information.txt"
if os.path.exists(output_file):
    with open(output_file, 'a') as f:
        f.write("\n" + "=" * 50 + "\n")

for dataset_name in dataset_names:
    print(f"\n=== Evaluating on {dataset_name} ===\n")
    test_loader = load_dataset(dataset_name)

    for name, model in models.items():
        print(f"===> Start evaluating the model: {model._get_name()} on {dataset_name}")
        model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 以兼容部分缺失的权重
        model.to(DEVICE)
        test_metrics = evaluate_model(model, test_loader)
        print(f"{name} on {dataset_name}: {test_metrics}")
        save_metrics_to_file(output_file, dataset_name, name, test_metrics)

print("\n=== All evaluations completed ===")