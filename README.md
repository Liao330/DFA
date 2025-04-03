# Unnamed

### 安装

#### pip 



#### conda



### 运行

#### 单机多卡运行

设置显卡使用序号（参考服务器的显卡数量进行分配）

export CUDA_VISIBLE_DEVICES=4,5,6,7

DDP模式运行：

torchrun --nproc_per_node=4 src/main.py

#### 单机单卡

export CUDA_VISIBLE_DEVICES=0

python -m src.main



### todo

按数据集名进行训练、测试