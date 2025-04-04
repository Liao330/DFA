import argparse
import random
from sklearn import metrics

import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, f1_score, \
    average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
from src.config import *
from src.data.datasets import BaseDataset, TestDataset
from src.utils.load_model import load_model
from tabulate import tabulate

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument('--model_class', type=str, help='The class name of the model to train')
parser.add_argument('--test_dataset', type=str, help='The dataset of the model to test')
# --test_dataset
args = parser.parse_args()


def print_pretty_results(dic, class_names):
    # 基础指标表格
    main_table = [
        ["Accuracy", f"{dic['acc']:.2f}%"],
        ["Precision", f"{dic['precision']:.4f}"],
        ["Recall", f"{dic['recall']:.4f}"],
        ["Err", f"{dic['err']:.4f}"],
        ["F1 Score", f"{dic['test_f1']:.4f}"],
        ["ROC AUC", f"{dic['roc_auc']:.4f}"],
        ["Video AUC", f"{dic['video_auc']:.4f}"]
    ]

    # 类别详细指标
    cm = dic['cm']
    precision_per_class = precision_score(dic['true_labels'], dic['pred_labels'], average=None)
    recall_per_class = recall_score(dic['true_labels'], dic['pred_labels'], average=None)

    class_table = []
    for i, name in enumerate(class_names):
        class_table.append([
            name,
            f"{precision_per_class[i]:.4f}",
            f"{recall_per_class[i]:.4f}",
            f"{cm[i, i]}/{cm[i].sum()}"
        ])

    # 打印输出
    print("\n\033[1;36m" + "=" * 50 + "\033[0m")
    print("\033[1;34mModel Evaluation Results\033[0m")
    print("\033[1;36m" + "=" * 50 + "\033[0m")

    print("\n\033[1;33mMain Metrics:\033[0m")
    print(tabulate(main_table, headers=["Metric", "Value"],
                   tablefmt="pretty", stralign="center"))

    print("\n\033[1;33mClass-wise Performance:\033[0m")
    print(tabulate(class_table,
                   headers=["Class", "Precision", "Recall", "TP/Total"],
                   tablefmt="pretty",
                   stralign="center"))

    print("\n\033[1;33mConfusion Matrix Summary:\033[0m")
    print(f"Total Samples: {sum(sum(cm))}")
    print(f"True Positives: {sum(cm.diagonal())}")
    print(f"False Positives: {sum(cm.sum(axis=0) - cm.diagonal())}")
    print(f"False Negatives: {sum(cm.sum(axis=1) - cm.diagonal())}")

def init_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(SEED)

def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
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


def test_epoch(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_true = []
    all_pred = []
    all_prob = []


    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch")
        for data_dict in pbar:
            img_inputs, lm_inputs, labels = data_dict.values()
            data_dict = {key: value.to(DEVICE) for key, value in data_dict.items()}
            img_inputs, lm_inputs, labels = img_inputs.to(DEVICE), lm_inputs.to(DEVICE), labels.to(
                DEVICE)

            if model._get_name() == 'DFACLIP':
                pred_dict = model(data_dict)
                outputs = pred_dict['cls']
            else:
                outputs = model(img_inputs)

            all_true.append(labels.cpu())
            all_pred.append(torch.argmax(outputs.data, dim=1).cpu())
            # 对每个batch的输出执行softmax
            probabilities = torch.softmax(outputs, dim=1)  # 添加维度说明
            all_prob.append(probabilities.cpu().numpy())  # 保存概率值

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'acc': 100 * correct / total
            }, refresh=False)

    # 使用 torch.cat 拼接张量
    all_true = torch.cat(all_true, dim=0).cpu().numpy()
    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_prob = np.concatenate(all_prob, axis=0)
    epoch_acc = 100 * correct / total

    all_true_flat = np.array(all_true).flatten()
    all_pred_flat = np.array(all_pred).flatten()


    cm = confusion_matrix(all_true_flat, all_pred_flat)
    precision = average_precision_score(all_true_flat, all_pred_flat)
    recall = recall_score(all_true_flat, all_pred_flat, pos_label=1)
    fpr, tpr, _ = roc_curve(all_true_flat, all_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    test_f1 = f1_score(all_true_flat, all_pred_flat, pos_label=1)

    img_names = test_loader.dataset.data_dict['image']
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, all_pred_flat, all_true_flat)
    else:
        # video-level methods
        v_auc = roc_auc

    dic = {
        'acc': epoch_acc,
        'precision': precision,
        'recall': recall,
        'err': eer,
        'roc_auc': roc_auc,
        'test_f1': test_f1,
        'cm': cm,
        'pred_labels': all_pred_flat,  # 添加预测标签字段
        'true_labels': all_true_flat,
        'probabilities': all_prob,
        'video_auc': v_auc
    }
    return dic


def visualize_results(model_class, test_dataset, dic, class_names=None, save_path=f'{EXP_DIR}/test_results.png'):
    """
    可视化测试结果
    参数：
        dic: 测试结果字典
        class_names: 类别名称列表（用于混淆矩阵）
        save_path: 结果保存路径
    """
    plt.figure(figsize=(18, 12))
    sns.set_style("whitegrid")
    sns.set_palette("pastel")
    plt.rcParams.update({'font.size': 12})

    # 创建网格布局
    gs = GridSpec(3, 3, figure=plt.gcf())

    # 指标汇总表格
    ax0 = plt.subplot(gs[0, 0])
    metrics = [
        ['Accuracy', f"{dic['acc']:.2f}%"],
        ['Precision', f"{dic['precision']:.4f}"],
        ['Recall', f"{dic['recall']:.4f}"],
        ['Err', f"{dic['err']:.4f}"],
        ['F1 Score', f"{dic['test_f1']:.4f}"],
        ['ROC AUC', f"{dic['roc_auc']:.4f}"],
        ['Video AUC', f"{dic['video_auc']:.4f}"]
    ]
    table = ax0.table(cellText=metrics,
                      colLabels=['Metric', 'Value'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    ax0.axis('off')
    ax0.set_title(f'{model_class} on {test_dataset} Performance Summary', fontsize=16, pad=20)

    # 混淆矩阵
    ax1 = plt.subplot(gs[0, 1:])
    cm = dic['cm']
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax1)
    ax1.set_xlabel('Predicted Label', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=14)
    ax1.set_title('Confusion Matrix', fontsize=16)

    # ROC曲线（假设二分类，多分类需要调整）
    ax2 = plt.subplot(gs[1, :])
    fpr, tpr, _ = roc_curve(dic['true_labels'], dic['probabilities'][:, 1], pos_label=1)  # 需要真实标签和概率
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right")

    # 指标分布雷达图
    ax3 = plt.subplot(gs[2, :], polar=True)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    values = [
        dic['acc'] / 100,  # 转换为0-1范围
        dic['precision'],
        dic['recall'],
        dic['test_f1'],
        dic['roc_auc']
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values += values[:1]
    ax3.plot(angles, values, color='skyblue', linewidth=2, linestyle='solid')
    ax3.fill(angles, values, color='skyblue', alpha=0.4)
    plt.xticks(angles[:-1], categories, size=14)
    ax3.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=12)
    plt.ylim(0, 1)
    plt.title('Metrics Radar Chart', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():

    init_seed()

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

    model_class = args.model_class
    model = load_model(model_class, DEVICE)
    model_weights_path = f"weights/best_{model_class}_model.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE), strict=False)

    print(f'===> Load {model_weights_path} done!')

    dic = test_epoch(model, test_loader)
    # 添加可视化调用
    class_names = ['Real', 'Fake']  # 根据实际类别修改
    visualize_results(model_class, test_dataset, dic, class_names=class_names)

    print_pretty_results(dic, class_names)

if __name__ == '__main__':
    main()


