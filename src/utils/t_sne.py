import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from src.utils.load_model import load_model
from src.data.datasets import TestDataset
from torch.utils.data import DataLoader, Subset
from src.config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, DEVICE



# 加载数据集
# dataset_name = "FaceForensics++"  # 可切换为 "DFDC"
dataset_name = "DFDC"  # 可切换为 "DFDC"
csv_path = f"{dataset_name}_labels.csv"
test_data = TestDataset(data_root=DATA_ROOT, csv_path=csv_path)
print(f"Loaded dataset {dataset_name}")

# 获取真实和虚假样本的索引
labels = np.array(test_data.data_dict['label'])
real_indices = np.where(labels == 0)[0]  # 假设 0 表示真实
fake_indices = np.where(labels == 1)[0]  # 假设 1 表示虚假

# 随机抽取 500 个真实和 500 个虚假样本
np.random.seed(706)
real_sample_size = min(500, len(real_indices))
fake_sample_size = min(500, len(fake_indices))
real_sampled_indices = np.random.choice(real_indices, size=real_sample_size, replace=False)
fake_sampled_indices = np.random.choice(fake_indices, size=fake_sample_size, replace=False)
sampled_indices = np.concatenate([real_sampled_indices, fake_sampled_indices])

# 创建子集
sampled_dataset = Subset(test_data, sampled_indices)
test_loader = DataLoader(sampled_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True, drop_last=False, collate_fn=TestDataset.collate_fn)
print(f"Sampled {real_sample_size} real and {fake_sample_size} fake images from {dataset_name}")

# 加载两个模型并提取特征
model_classes = ['Xception', 'DFACLIP']
features_dict = {}
labels_dict = {}

for model_class in model_classes:
    model = load_model(model_class, DEVICE)
    model_weights_path = f"weights/best_{model_class}_model.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE), strict=False)
    model.eval()
    print(f"Loaded model from {model_weights_path}")

    # 提取特征
    all_features = []
    all_labels = []
    with torch.no_grad():
        for data_dict in tqdm(test_loader, desc=f"Extracting features for {model_class}"):
            data_dict = {key: value.to(DEVICE) for key, value in data_dict.items()}
            if model_class == "DFACLIP":
                pred_dict = model(data_dict, inference=True)
                features = pred_dict['features'].cpu().numpy()  # [B, 1024]
            elif model_class == "Xception":
                images = data_dict['image']
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                pred_dict = model(images, inference=True)
                features = pred_dict['features'].cpu().numpy()  # [B, 2048]
            else:
                raise ValueError(f"Unsupported model class: {model_class}")
            labels = data_dict['label'].cpu().numpy()
            all_features.append(features)
            all_labels.append(labels)

    # 存储特征和标签
    features_dict[model_class] = np.concatenate(all_features, axis=0)
    labels_dict[model_class] = np.concatenate(all_labels, axis=0)
    print(f"{model_class} - Extracted features shape: {features_dict[model_class].shape}, labels shape: {labels_dict[model_class].shape}")

# T-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
tsne_features_dict = {}
for model_class in model_classes:
    tsne_features_dict[model_class] = tsne.fit_transform(features_dict[model_class])

# 并排可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 颜色映射（匹配上传图片的颜色）
palette = {"Real": "pink", "Fake": "green"}

# 左图：Xception (CLIP)
label_names_xception = np.array(["Real" if label == 0 else "Fake" for label in labels_dict['Xception']])
sns.scatterplot(x=tsne_features_dict['Xception'][:, 0], y=tsne_features_dict['Xception'][:, 1],
                hue=label_names_xception, palette=palette, s=50, alpha=0.6, ax=ax1)
ax1.set_title("Xception", fontsize=16)
ax1.set_xlabel("")
ax1.set_ylabel("DFDC", fontsize=12)
ax1.grid(True)
ax1.legend(title="Label", loc="lower right")

# 右图：DFACLIP (OURS)
label_names_dfaclip = np.array(["Real" if label == 0 else "Fake" for label in labels_dict['DFACLIP']])
sns.scatterplot(x=tsne_features_dict['DFACLIP'][:, 0], y=tsne_features_dict['DFACLIP'][:, 1],
                hue=label_names_dfaclip, palette=palette, s=50, alpha=0.6, ax=ax2)
ax2.set_title("OURS", fontsize=16)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.grid(True)
ax2.legend(title="Label", loc="lower right")

# 调整布局
plt.tight_layout()
plt.savefig(f"tsne_comparison_{dataset_name}_sampled.png", dpi=300, bbox_inches="tight")
plt.show()

