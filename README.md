# Unnamed

### 安装

#### conda

```
conda create -n DFACLIP python=3.10

conda activate DFACLIP

cd DFACLIP

pip install -r  requirements.txt
( -i https://pypi.mirrors.ustc.edu.cn/simple 如有需要)

conda deactivate
```



### 运行

#### 单机多卡运行

即use_gpu_num > 1

设置显卡使用序号（参考服务器的显卡数量进行分配）

```
export CUDA_VISIBLE_DEVICES=4,5,6,7
DDP模式运行：
torchrun --nproc_per_node=4 src/main.py --model_class=DFACLIP
```

#### 单机单卡

即use_gpu_num <= 1

```
export CUDA_VISIBLE_DEVICES=0
python -m src.main --model_class=DFACLIP
```

#### 测试某一数据集

```
python -m src.test --model_class=DFACLIP --test_dataset=your—dataset
如: python -m src.test --model_class=DFACLIP --test_dataset=DFDC
```

#### 消融实验

```
python -m src.ablation  --test_dataset=your—dataset(默认是全部)
如: python -m src.ablation --test_dataset=DFDC
```



### todo

- 按不同数据集进行训练、测试（已完成）
- 用于消融实验的继承类设计（已完成）
- 使用DALI框架（失败）
- DFD数据集处理
- 