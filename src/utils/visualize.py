import torch
import matplotlib.pyplot as plt

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

def plot_loss_curve(train_loss, test_loss, loss_plot_path):
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

def plot_acc_curve(train_acc, test_acc, acc_plot_path):
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