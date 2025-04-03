import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

# 定义反归一化函数
def denormalize(tensor):
    tensor = tensor.clone().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)  # 确保值在[0,1]范围内

# 创建网格显示函数
def imshow_grid(loader, num_images=8):
    images, labels = next(iter(loader))
    images = denormalize(images[:num_images])
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        ax = fig.add_subplot(2, num_images // 2, i + 1)
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title("REAL" if labels[i].item() == 1 else "FAKE")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_loss_curve(EXP_DIR, train_loss, test_loss):
    loss_plot_path = os.path.join(EXP_DIR, "loss_history.png")
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(loss_plot_path)  # 保存损失曲线图
    plt.close()  # 关闭当前图形，以免影响下一个图形的显示

def plot_acc_curve(EXP_DIR, train_acc, test_acc):
    acc_plot_path = os.path.join(EXP_DIR, "acc_history.png")
    plt.figure(figsize=(6, 4))
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(acc_plot_path)  # 保存准确率曲线图
    plt.close()  # 关闭当前图形

def plot_confusion_matrix(EXP_DIR, cm):
    # 确保 y_true 和 y_pred 是单标签格式
    # y_true_single = np.argmax(y_true, axis=1)
    # y_pred_single = np.argmax(y_pred, axis=1)

    cm_plot_path = os.path.join(EXP_DIR, "confusion_matrix.png")
    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # cm = confusion_matrix(y_true_single, y_pred_single)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    # plt.savefig(cm_plot_path)
    plt.close()

def plot_f1_score(EXP_DIR, test_f1):
    # 确保 y_true 和 y_pred 是单标签格式
    # y_true_single = np.argmax(y_true, axis=1)
    # y_pred_single = np.argmax(y_pred, axis=1)
    f1_plot_path = os.path.join(EXP_DIR, "f1_score.png")
    # 绘制F1曲线
    # train_f1 = [f1_score(y_true, y_pred, average='macro')] * len(train_loss)
    # train_f1 = [f1_score(y_true_single, y_pred_single, average='macro')] * len(train_loss)
    # test_f1 = [f1_score(y_pred_single, y_pred_single, average='macro')] * len(test_loss)
    # test_f1 = [f1_score(y_true, y_pred, average='macro')] * len(test_loss)
    plt.figure(figsize=(6, 4))
    # plt.plot(train_f1, label='Train F1')
    plt.plot(test_f1, label='Test F1')
    plt.title('F1 Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(f1_plot_path)
    plt.close()

def plot_roc(EXP_DIR, fpr, tpr):
    # 确保 y_true 和 y_pred 是单标签格式
    # y_true_single = np.argmax(y_true, axis=1)

    roc_plot_path = os.path.join(EXP_DIR, "roc.png")
    # 绘制ROC曲线
    # fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # 二分类问题
    # fpr, tpr, _ = roc_curve(y_true_single, y_pred_proba[:, 1])  # 二分类问题
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    # plt.savefig(roc_plot_path)
    plt.close()

def print_test_epoch_result(test_loss, test_acc, precision, recall, roc_auc, test_f1, cm):
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print(f"Test Precision: {precision:.4f} | Test Recall: {recall:.2f}%")
    print(f"Test Roc_Auc: {roc_auc:.4f} | Test F1: {test_f1:.2f}%")
    print(f"Test CM: {cm}")
    print("-" * 50)

def single_res_fmap_heatmap(image, logits, fmap):
    # """
    #
    #     # 加载图像
    # # image_path = r'E:\github_code\Unnamed1\dataset\processed\FaceForensics++\manipulated_sequences\Face2Face\c23\frames\103_082\203.png'
    # image_path = r'E:\github_code\Unnamed1\dataset\processed\FaceForensics++\original_sequences\youtube\c23\frames\001\029.png'
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    # # Convert NumPy array to PIL Image
    # image_pil = Image.fromarray(image)
    #
    # # 转换为张量并归一化
    # to_tensor = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # image_tensor = to_tensor(image_pil).unsqueeze(0).cuda()
    #
    # image_tensor.requires_grad_(True)  # 启用梯度计算
    #
    # """

    print("Logits:", logits)
    print("Feature map shape:", fmap.shape)

    # 获取目标类别（预测类别）
    target_class = logits.argmax(dim=1).item()
    print(f"Predicted class: {target_class}")

    # 计算梯度
    fmap.retain_grad()  # 保留特征图的梯度
    logits[0, target_class].backward()  # 对目标类别反向传播

    # 获取梯度和权重
    gradients = fmap.grad  # [1, 2048, 7, 7]
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, 2048, 1, 1]，对空间维度求平均

    # 计算 Grad-CAM
    gradcam = (weights * fmap).sum(dim=1).relu()  # [1, 7, 7]，加权求和并应用 ReLU
    gradcam = gradcam[0].cpu().detach().numpy()  # [7, 7]

    # 上采样到原始图像大小
    gradcam_tensor = torch.from_numpy(gradcam).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 7]
    gradcam_up = F.interpolate(gradcam_tensor, size=(224, 224), mode='bilinear',
                               align_corners=False).squeeze().numpy()  # [224, 224]

    # 归一化
    if gradcam_up.max() > gradcam_up.min():
        gradcam_norm = (gradcam_up - gradcam_up.min()) / (gradcam_up.max() - gradcam_up.min() + 1e-8)
    else:
        gradcam_norm = np.zeros_like(gradcam_up)

    # 转换为热图
    heatmap = cv2.applyColorMap((gradcam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)  # [224, 224, 3]

    # 叠加到原始图像
    image_resized = cv2.resize(image, (224, 224))  # [224, 224, 3]
    overlay = cv2.addWeighted(image_resized, 0.5, heatmap, 0.5, 0)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_resized)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gradcam_norm, cmap='viridis')
    plt.title('Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    plt.show()

def single_clip_fmap_heatmap(image, logits, fmap):
    print("Logits:", logits)
    print("Feature map shape:", fmap.shape)

    # 获取目标类别（预测类别）
    target_class = logits.argmax(dim=1).item()
    print(f"Predicted class: {target_class}")

    # 计算梯度
    fmap_grad = fmap.clone().detach().requires_grad_(True)  # 创建一个新的副本并设置 requires_grad=True
    logits[0, target_class].backward(retain_graph=True)  # 对目标类别反向传播

    # 获取梯度和权重
    gradients = fmap_grad.grad  # [1, 385, 1024]
    if gradients is None:
        raise ValueError("Gradients are None, please check the backward pass.")
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, 385, 1, 1]，对空间维度求平均

    # 计算 Grad-CAM
    gradcam = (weights * fmap_grad).sum(dim=1).relu()  # [1, 1, 1024]，加权求和并应用 ReLU
    gradcam = gradcam[0].cpu().detach().numpy()  # [1, 1024]

    # 上采样到原始图像大小
    gradcam_tensor = torch.from_numpy(gradcam).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1024]
    gradcam_up = F.interpolate(gradcam_tensor, size=(224, 224), mode='bilinear',
                               align_corners=False).squeeze().numpy()  # [224, 224]

    # 归一化
    if gradcam_up.max() > gradcam_up.min():
        gradcam_norm = (gradcam_up - gradcam_up.min()) / (gradcam_up.max() - gradcam_up.min() + 1e-8)
    else:
        gradcam_norm = np.zeros_like(gradcam_up)

    # 转换为热图
    heatmap = cv2.applyColorMap((gradcam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)  # [224, 224, 3]

    # 叠加到原始图像
    image_resized = cv2.resize(image, (224, 224))  # [224, 224, 3]
    overlay = cv2.addWeighted(image_resized, 0.5, heatmap, 0.5, 0)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gradcam_norm, cmap='viridis')
    plt.title('Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.axis('off')
    plt.show()

# # 测试数据
# y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
# y_pred = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
# y_pred_proba = np.array([
#     [0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.3, 0.7],
#     [0.6, 0.4], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.7, 0.3],
#     [0.2, 0.8], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]
# ])
# # 模拟训练和测试损失
# train_loss = np.random.rand(10) * 0.5  # 假设训练了10个epoch
# test_loss = np.random.rand(10) * 0.5
# print(train_loss.shape,train_loss)
# # 调用绘图函数
# plot_confusion_matrix('None', y_true, y_pred)
# plot_f1_score('None', y_true, y_pred, train_loss, test_loss)
# plot_roc('None', y_true, y_pred_proba)