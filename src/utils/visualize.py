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