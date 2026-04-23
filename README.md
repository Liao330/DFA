# Deepfake Forensics Adapter (DFA)

This repository is the official implementation of the paper:

**[Deepfake Forensics Adapter: A Dual-Stream Network for Generalizable Deepfake Detection](https://arxiv.org/abs/2603.01450)**

> **Paper**: [https://arxiv.org/abs/2603.01450](https://arxiv.org/abs/2603.01450)
>
> **Authors**: Jianfeng Liao, Yichen Wei, Raymond Chan Ching Bon, Shulan Wang, Kam-Pui Chow, Kwok-Yan Lam
>
> **Venue**: Accepted at **ICDF2C 2025**.

---

## Installation

### Conda

```bash
conda create -n DFACLIP python=3.10

conda activate DFACLIP

cd DFACLIP

pip install -r requirements.txt
# (append `-i https://pypi.mirrors.ustc.edu.cn/simple` if you need a faster mirror)

conda deactivate
```

---

## Code Structure

```
DFA-project/
├── data/                 # Data directory
│   ├── raw/              # Raw data (NOT committed to Git)
│   └── processed/        # Processed data (NOT committed to Git)
│
├── experiments/          # Experiment logs (one sub-directory per run, NOT committed)
│   └── exp_model_name_20230801/    # Experiment date or ID
│
├── src/                  # Core source code
│   ├── data/             # Data loading & preprocessing
│   ├── utils/            # Utility functions
│   ├── models/           # Model definitions
│   ├── ablation.py       # Ablation study
│   ├── main.py           # Training entry point
│   ├── test.py           # Testing script
│   ├── Trainer.py        # Training loop wrapper
│   ├── config.py         # Configuration
│   └── fullcode.py       # Full training pipeline
│
├── extra_data.py         # Generate dataset CSV files
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment file
└── .gitignore            # Ignore large / temporary files
```

---

## How to Run

### Multi-GPU on a Single Machine

When `use_gpu_num > 1`, set the visible GPU indices according to your server configuration:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
# Launch in DDP mode:
torchrun --nproc_per_node=4 src/main.py --model_class=DFACLIP
```

### Single-GPU on a Single Machine

#### Training DFACLIP

When `use_gpu_num <= 1`:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m src.main --model_class=DFACLIP
```

#### Training Other Models

```bash
sh train_together.sh
```

### Testing on a Specific Dataset

```bash
python -m src.test --model_class=DFACLIP --test_dataset=your_dataset
# e.g.
python -m src.test --model_class=DFACLIP --test_dataset=DFDC
# Note: the testing script will be merged into `ablation.py` later;
#       please adjust the arguments accordingly when that happens.
```

### Ablation Study

```bash
python -m src.ablation --test_dataset=your_dataset   # defaults to all datasets
# e.g.
python -m src.ablation --test_dataset=DFDC
```

---

## Pre-trained Weights

Pre-trained weights are shared via Baidu Netdisk:

```
Link: https://pan.baidu.com/s/1l_qkNl3hCWhHTSyH-5EaDg?pwd=y8su
Access code: y8su
(Shared from a Baidu Netdisk Super VIP v5 account)
```

### ⭐ Support This Project

If you find this repository helpful for your research or work, **please consider giving us a star on GitHub!** 🌟
Your support is the biggest motivation for us to keep maintaining and improving this project. It also helps more researchers discover this work. Thank you so much! 🙏

---

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
