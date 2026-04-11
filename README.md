# Deepfake Forensics Adapter (DFA)

This repository is the official implementation of the paper:

**[Deepfake Forensics Adapter: A Dual-Stream Network for Generalizable Deepfake Detection](https://arxiv.org/abs/2603.01450)**

> **Paper**: [https://arxiv.org/abs/2603.01450](https://arxiv.org/abs/2603.01450)

> **Authors**: Jianfeng Liao, Yichen Wei, Raymond Chan Ching Bon, Shulan Wang, Kam-Pui Chow, Kwok-Yan Lam

> **Venue**: Accepted at **ICDF2C 2025**.


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

### 代码目录概览

```
DFA-project/
├── data/                 # 数据目录
│   ├── raw/             # 原始数据（不提交到 Git）
│   └── processed/       # 处理后的数据（不提交到 Git）
│
├── experiments/         # 实验记录（每次实验的子目录，不提交）
│   └── exp_model_name_20230801/    # 实验日期或 ID
│
├── src/                 # 核心代码
│   ├── data/           # 数据加载与预处理
│   ├── utils/          # 工具函数
│   ├── models/         # 模型类
│   ├── ablation.py     # 消融实验
│   ├── main.py         # 训练脚本
│   ├── test.py         # 测试脚本
│   ├── Trainer.py      # 封装训练代码
│   ├── config.py       # 配置信息
│   └── fullcode.py     # 完整训练代码
│
├── extra_data.py        # 生成数据集的csv文件
├── requirements.txt     # Python 依赖
├── environment.yml      # Conda 环境配置
└── .gitignore           # 排除大文件/临时文件
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

##### 训练DFACLIP

即use_gpu_num <= 1

```
export CUDA_VISIBLE_DEVICES=0
python -m src.main --model_class=DFACLIP
```

##### 训练各种模型

```
sh train_together.sh
```

#### 测试某一数据集

```
python -m src.test --model_class=DFACLIP --test_dataset=your—dataset
如: python -m src.test --model_class=DFACLIP --test_dataset=DFDC
注：后续将test代码合并到ablation中，请对应修改参数
```

#### 消融实验

```
python -m src.ablation  --test_dataset=your—dataset(默认是全部)
如: python -m src.ablation --test_dataset=DFDC
```

#### 各模型权重下载：

```
通过网盘分享的文件：weights
链接: https://pan.baidu.com/s/1l_qkNl3hCWhHTSyH-5EaDg?pwd=y8su 提取码: y8su 
--来自百度网盘超级会员v5的分享
```



## 📖 Citation

If you find this work useful for your research, please cite the original paper:
```bibtex
@article{liao2026deepfake,
  title     = {Deepfake Forensics Adapter: A Dual-Stream Network for Generalizable Deepfake Detection},
  author    = {Jianfeng Liao and Yichen Wei and Raymond Chan Ching Bon and Shulan Wang and Kam-Pui Chow and Kwok-Yan Lam},
  journal   = {Proceedings of the International Conference on Digital Forensics and Cyber Crime (ICDF2C)}, 
  year      = {2025},
  note      = {To appear},
  archivePrefix = {arXiv},
  eprint    = {2603.01450},
  primaryClass = {cs.CV},
  doi       = {10.48550/arXiv.2603.01450},
  url       = {https://arxiv.org/abs/2603.01450}
}
```
