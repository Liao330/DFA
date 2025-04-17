import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
import nvidia.dali as dali
import cloudpickle
import os
import numpy as np
import pandas as pd
import torch

from src.config import BATCH_SIZE, DATA_ROOT, DATASET_PATH


class LandmarkProcessor:
    @staticmethod
    def load_landmark(landmark_paths):
        rank = os.environ.get('RANK', 'unknown')
        try:
            if isinstance(landmark_paths, (list, np.ndarray)):  # 批处理模式（暂时不会触发）
                result = []
                for p in landmark_paths:
                    path = p.decode('utf-8')
                    if not os.path.exists(path):
                        print(f"Rank {rank}: Landmark file not found: {path}")
                        result.append(np.zeros((8, 2), dtype=np.float32))
                    else:
                        data = np.load(path).astype(np.float32)
                        if data.shape != (8, 2):
                            print(f"Rank {rank}: Invalid shape {data.shape} for {path}")
                            result.append(np.zeros((8, 2), dtype=np.float32))
                        else:
                            result.append(data)
                result = np.array(result)
                print(f"Rank {rank}: Landmark batch, type={type(result)}, shape={result.shape}")
                return result
            else:  # 单样本模式
                path = landmark_paths.decode('utf-8')
                if not os.path.exists(path):
                    print(f"Rank {rank}: Landmark file not found: {path}")
                    return np.zeros((8, 2), dtype=np.float32)
                result = np.load(path).astype(np.float32)
                if result.shape != (8, 2):
                    print(f"Rank {rank}: Invalid shape: {result.shape}, path: {path}")
                    result = np.zeros((8, 2), dtype=np.float32)
                print(f"Rank {rank}: Landmark single, type={type(result)}, shape={result.shape}")
                return result
        except Exception as e:
            print(f"Rank {rank}: Error loading landmark {landmark_paths}: {str(e)}")
            return np.zeros((8, 2), dtype=np.float32)


def load_csv_data(csv_path, shard_id, num_shards, shuffle=False):
    df = pd.read_csv(csv_path, header=0)
    print(f"CSV columns: {df.columns.tolist()}")
    required_columns = ["img", "landmark", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file missing required column: {col}")

    # 分片
    total_size = len(df)
    shard_size = total_size // num_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size if shard_id < num_shards - 1 else total_size
    df = df.iloc[start_idx:end_idx]

    # 可选随机打乱
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    image_paths = df["img"].tolist()
    label_map = {'REAL': 0, 'FAKE': 1}
    labels = np.array([label_map[label] for label in df["label"].tolist()], dtype=np.int64)
    landmark_paths = df["landmark"].tolist()

    print(f"Shard {shard_id}/{num_shards}: image_paths length={len(image_paths)}, sample={image_paths[:5]}")
    print(f"Shard {shard_id}/{num_shards}: labels length={len(labels)}, sample={labels[:5].tolist()}")
    print(f"Shard {shard_id}/{num_shards}: landmark_paths length={len(landmark_paths)}, sample={landmark_paths[:5]}")
    return image_paths, labels, landmark_paths


class SampleSource:
    def __init__(self, data, batch_size, shuffle=False, is_batch=False, is_landmark=False):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_batch = is_batch
        self.is_landmark = is_landmark  # 新增标志，用于区分 landmark 数据
        self.total_samples = len(data)
        self.indices = np.arange(self.total_samples)
        self.rng = np.random.RandomState(seed=42)
        if shuffle:
            self.rng.shuffle(self.indices)

    def __call__(self, sample_info):
        rank = os.environ.get('RANK', 'unknown')
        try:
            if isinstance(sample_info, dali.types.SampleInfo):
                batch_idx = sample_info.iteration
                sample_idx = sample_info.idx_in_batch if not self.is_batch else None
            elif isinstance(sample_info, int):
                batch_idx = sample_info
                sample_idx = None
            else:
                raise TypeError(f"Rank {rank}: Unsupported sample info type: {type(sample_info)}")

            batch_start = batch_idx * self.batch_size
            if batch_start >= self.total_samples:
                raise StopIteration

            batch_indices = self.indices[batch_start:batch_start + self.batch_size]

            if not self.is_batch and sample_idx is not None:  # batch=False，逐样本返回
                sample_idx = batch_indices[sample_idx]
                sample = self.data[sample_idx]
                if self.is_landmark:  # 处理 landmark 数据
                    path = sample
                    if not os.path.exists(path):
                        print(f"Rank {rank}: Landmark file not found: {path}")
                        result = np.zeros((8, 2), dtype=np.float32)
                    else:
                        result = np.load(path).astype(np.float32)
                        if result.shape != (8, 2):
                            print(f"Rank {rank}: Invalid shape {result.shape} for {path}")
                            result = np.zeros((8, 2), dtype=np.float32)
                    print(f"Rank {rank}: Landmark single, type={type(result)}, shape={result.shape}")
                    return result
                elif isinstance(sample, str):  # 处理图像路径
                    result = np.array([sample.encode('utf-8')], dtype=object)
                    print(f"Rank {rank}: batch=False, type={type(result)}, value={result}")
                    return result
                elif isinstance(sample, (int, np.integer)):  # 处理标签
                    result = np.array([sample], dtype=np.int64)
                    print(f"Rank {rank}: batch=False, type={type(result)}, value={result}")
                    return result
                else:
                    raise ValueError(f"Rank {rank}: Unsupported data type: {type(sample)}")

            # batch=True，批次返回（暂时未使用）
            batch_data = self.data[batch_indices]
            if isinstance(self.data[0], str):
                result = np.array([d.encode('utf-8') for d in batch_data], dtype=object)
                print(f"Rank {rank}: batch=True, type={type(result)}, value={result[:2]}... (len={len(result)})")
                return result
            elif isinstance(self.data[0], (int, np.integer)):
                result = np.array(batch_data, dtype=np.int64)
                print(f"Rank {rank}: batch=True, type={type(result)}, value={result[:2]}... (len={len(result)})")
                return result
            else:
                raise ValueError(f"Rank {rank}: Unsupported data type: {type(self.data[0])}")
        except Exception as e:
            print(f"Rank {rank}: SampleSource error: {str(e)}")
            raise

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['rng']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = np.random.RandomState(seed=42)

@pipeline_def(batch_size=BATCH_SIZE, num_threads=4, device_id=0, py_start_method="spawn")
def create_dali_pipeline(mode, data_dir, csv_data, shard_id, num_shards, is_training=True):
    image_paths, labels, landmark_paths = load_csv_data(
        csv_data, shard_id, num_shards, shuffle=is_training
    )

    image_source = SampleSource(image_paths, BATCH_SIZE, is_training, is_batch=False, is_landmark=False)
    label_source = SampleSource(labels, BATCH_SIZE, is_training, is_batch=False, is_landmark=False)
    landmark_source = SampleSource(landmark_paths, BATCH_SIZE, is_training, is_batch=False, is_landmark=True)

    images = fn.external_source(
        source=image_source,
        batch=False,
        parallel=False,
    )
    labels = fn.external_source(
        source=label_source,
        batch=False,
        parallel=False,
    )
    landmarks = fn.external_source(
        source=landmark_source,
        batch=False,
        parallel=False,
    )

    # 图像解码
    images = fn.decoders.image(images, device="mixed")

    # 数据增强
    if is_training:
        images = fn.random_resized_crop(images, size=(224, 224), random_area=[0.08, 1.0])
        images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    else:
        images = fn.resize(images, size=(224, 224))

    # 归一化处理
    images = fn.crop_mirror_normalize(
        images,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        dtype=types.FLOAT
    )

    # 标签处理，确保批次维度
    labels = fn.cast(labels, dtype=types.INT64)
    labels = fn.reshape(labels, shape=[-1])

    # 确保 landmarks 的形状正确
    landmarks = fn.reshape(landmarks, shape=(-1, 8, 2))

    outputs = [images.gpu(), labels.gpu()]
    if mode == 'train':
        outputs.insert(1, landmarks.gpu())

    return tuple(outputs)