#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 定义一个函数来运行 Python 训练脚本
run_training() {
    python -m src.main --model_class=$1
}

# 运行不同模型的训练
run_training "Xception"
run_training "ResNext"
run_training "EfficientViT"
run_training "SimpleCNN"
run_training "ConvNext"