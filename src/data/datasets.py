import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import albumentations as A
from torchvision import transforms as T


class BaseDataset(Dataset):
    """基础数据集类，支持图像和关键点数据"""

    def __init__(self, data_root, csv_path, config=None, mode='train'):
        self.data_root = data_root
        self.config = config if config is not None else {
            'resolution': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'data_aug': {
                'flip_prob': 0.5,
                'rotate_limit': 15,
                'rotate_prob': 0.5,
                'blur_limit': 3,
                'blur_prob': 0.1,
            }
        }
        self.mode = mode
        self.label_map = {"REAL": 1, "FAKE": 0}
        self._load_metadata(csv_path)

    def _load_metadata(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在于 {csv_path}")
        df = pd.read_csv(csv_path)
        self.images = df['img'].tolist()
        if 'landmark' in df.columns:
            self.landmarks = df['landmark'].tolist()
        else:
            self.landmarks = None
        self.labels = df['label'].tolist()

    def init_data_aug_method(self):
        raise NotImplementedError("子类必须实现 init_data_aug_method 方法")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raise NotImplementedError("子类必须实现 __getitem__ 方法")

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch], dim=0)
        landmarks = torch.stack([item['landmark'] for item in batch], dim=0)
        labels = torch.stack([item['label'] for item in batch], dim=0)
        return {'image': images, 'landmark': landmarks, 'label': labels}

    def num_of_real_and_fake(self):
        real = sum(1 for label in self.labels if label == "REAL")
        fake = sum(1 for label in self.labels if label == "FAKE")
        return real, fake


class TrainDataset(BaseDataset):
    """训练数据集类，基于代码 B"""

    def __init__(self, data_root, csv_path, config=None):
        super().__init__(data_root, csv_path, config, mode='train')
        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        aug_transform = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug'].get('flip_prob', 0.5)),
            A.Rotate(limit=self.config['data_aug'].get('rotate_limit', 15),
                     p=self.config['data_aug'].get('rotate_prob', 0.5)),
            A.GaussianBlur(blur_limit=self.config['data_aug'].get('blur_limit', 3),
                           p=self.config['data_aug'].get('blur_prob', 0.1)),
            A.Resize(height=self.config.get('resolution', 224), width=self.config.get('resolution', 224)),
        ], keypoint_params=A.KeypointParams(format='xy'))

        def transform_fn(img, landmark):
            img_np = np.array(img)
            kwargs = {'image': img_np}
            if landmark is not None and landmark.shape[0] > 0:
                kwargs['keypoints'] = landmark.numpy() if isinstance(landmark, torch.Tensor) else landmark
            else:
                kwargs['keypoints'] = torch.zeros((81, 2), dtype=torch.float32).numpy()

            transformed = aug_transform(**kwargs)
            img_trans = transformed['image']
            landmark_trans = transformed.get('keypoints')

            img_trans = T.ToTensor()(img_trans)
            img_trans = T.Normalize(mean=self.config.get('mean', [0.485, 0.456, 0.406]),
                                    std=self.config.get('std', [0.229, 0.224, 0.225]))(img_trans)

            if landmark_trans is None or len(landmark_trans) == 0:
                landmark_trans = torch.zeros((81, 2), dtype=torch.float32)
            else:
                landmark_trans = torch.from_numpy(np.array(landmark_trans, dtype=np.float32))
                if landmark_trans.shape[0] != 81:
                    if landmark_trans.shape[0] < 81:
                        last_point = landmark_trans[-1:, :]
                        num_missing = 81 - landmark_trans.shape[0]
                        padding = last_point.repeat(num_missing, 1)
                        landmark_trans = torch.cat([landmark_trans, padding], dim=0)
                    else:
                        landmark_trans = landmark_trans[:81, :]
            return img_trans, landmark_trans

        return transform_fn

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.images[idx])

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"错误加载图像 {img_name}: {e}")
            return self.__getitem__(0)

        if self.landmarks is not None:
            landmarks_path = os.path.join(self.data_root, self.landmarks[idx])
            if os.path.exists(landmarks_path):
                try:
                    landmark = np.load(landmarks_path)
                    landmark = torch.from_numpy(landmark.astype(np.float32))
                    if landmark.shape != (81, 2):
                        landmark = torch.zeros((81, 2), dtype=torch.float32)
                except Exception as e:
                    print(f"错误加载关键点 {landmarks_path}: {e}")
                    landmark = torch.zeros((81, 2), dtype=torch.float32)
            else:
                landmark = torch.zeros((81, 2), dtype=torch.float32)
        else:
            landmark = torch.zeros((81, 2), dtype=torch.float32)

        label_str = self.labels[idx]
        label = self.label_map[label_str]

        image, landmark = self.transform(image, landmark)

        if landmark.shape != (81, 2):
            landmark = torch.zeros((81, 2), dtype=torch.float32)

        return {
            'image': image,
            'landmark': landmark,
            'label': torch.tensor(label, dtype=torch.long)
        }


class TestDataset(BaseDataset):
    """测试数据集类，基于代码 A"""

    def __init__(self, data_root, csv_path, config=None):
        super().__init__(data_root, csv_path, config, mode='test')
        self.transform = self.init_data_aug_method()
        self.data_dict = {
            'image': [],
            'landmark': [],
            'label': []
        }
        self._load_additional_metadata()

    def _load_additional_metadata(self):
        for img_name, label_str in zip(self.images, self.labels):
            self.data_dict['image'].append(os.path.join(self.data_root, img_name))
            self.data_dict['label'].append(self.label_map[label_str])
            if self.landmarks is not None:
                lm_path = os.path.join(self.data_root, self.landmarks[self.images.index(img_name)])
                self.data_dict['landmark'].append(lm_path)
            else:
                self.data_dict['landmark'].append(None)

    def init_data_aug_method(self):
        def default_transform(img, landmark):
            transform = T.Compose([
                T.Resize((self.config.get('resolution', 224), self.config.get('resolution', 224))),
                T.ToTensor(),
                T.Normalize(mean=self.config.get('mean', [0.485, 0.456, 0.406]),
                            std=self.config.get('std', [0.229, 0.224, 0.225]))
            ])
            return transform(img), landmark

        return default_transform

    def _get_fallback_sample(self, idx):
        return {
            'image': torch.zeros(3, self.config['resolution'], self.config['resolution']),
            'landmark': torch.zeros((81, 2)),
            'label': torch.tensor(0)
        }

    def _fix_landmark_shape(self, landmark):
        if landmark.shape[0] < 81:
            last_point = landmark[-1:, :]
            padding = last_point.repeat(81 - landmark.shape[0], 1)
            return torch.cat([landmark, padding], dim=0)
        return landmark[:81, :]

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_root, self.images[idx])

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"错误加载图像 {img_name}: {e}")
            return self._get_fallback_sample(idx)

        if self.landmarks is not None:
            landmarks_path = os.path.join(self.data_root, self.landmarks[idx])
            if os.path.exists(landmarks_path):
                try:
                    landmark = np.load(landmarks_path)
                    landmark = torch.from_numpy(landmark.astype(np.float32))
                    if landmark.shape != (81, 2):
                        landmark = torch.zeros((81, 2), dtype=torch.float32)
                except Exception as e:
                    print(f"错误加载关键点 {landmarks_path}: {e}")
                    landmark = torch.zeros((81, 2), dtype=torch.float32)
            else:
                landmark = torch.zeros((81, 2), dtype=torch.float32)
        else:
            landmark = torch.zeros((81, 2), dtype=torch.float32)

        label_str = self.labels[idx]
        label = self.label_map[label_str]

        image, landmark = self.transform(image, landmark)

        if landmark.shape != (81, 2):
            landmark = self._fix_landmark_shape(landmark)

        return {
            'image': image,
            'landmark': landmark,
            'label': torch.tensor(label, dtype=torch.long)
        }


# 示例使用
if __name__ == "__main__":
    data_root = ""
    csv_path = r"E:\github_code\Unnamed1\DFDC_labels.csv"

    # 训练数据集
    train_dataset = TrainDataset(data_root, csv_path)
    print(f"训练数据集大小: {len(train_dataset)}")
    real, fake = train_dataset.num_of_real_and_fake()
    print(f"训练数据 - REAL: {real}, FAKE: {fake}")

    # 测试数据集
    test_dataset = TestDataset(data_root, csv_path)
    print(f"测试数据集大小: {len(test_dataset)}")
    real, fake = test_dataset.num_of_real_and_fake()
    print(f"测试数据 - REAL: {real}, FAKE: {fake}")