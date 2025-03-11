from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, metrics: dict, step: int):
        """记录标量指标"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def save_config(self, config: dict):
        """保存配置文件"""
        config_path = self.log_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

    def close(self):
        self.writer.close()