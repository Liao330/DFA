from src.Trainer import Trainer
from src.models.SimpleCNN import SimpleCNN
from src.utils.load_model import load_model
from src.utils.save_exp_config_and_results import save_exp_config, save_exp_results
from src.utils.visualize import imshow_grid
from src.data.loaders import get_dataloader
from src.utils.logger import ExperimentLogger
from config import *

# 获取dataloader
train_loader, test_loader = get_dataloader()

# 可视化样本测试（可选）
# imshow_grid(train_loader)

# 定义模型
model = load_model(MODEL_CLASS, DEVICE)
trainer = Trainer(model, train_loader,test_loader, CRITERION, OPTIMIZER_CLASS, DEVICE)

# 创建tensorboard Logger
Logger = ExperimentLogger(LOG_DIR)

# 初始化best acc和loss
best_test_acc = 0
best_test_loss = 100

for epoch in range(NUM_EPOCHS):
    print(f"第{epoch + 1} 轮开始：")
    train_loss, train_acc = trainer.train_epoch()
    Logger.log_metrics(Logger.todict(train_loss), epoch)
    Logger.log_metrics(Logger.todict(train_acc), epoch)

    test_loss, test_acc = trainer.test_epoch()
    Logger.log_metrics(Logger.todict(test_loss), epoch)
    Logger.log_metrics(Logger.todict(test_acc), epoch)

    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("-" * 50)

    # 更新最佳测试准确率和损失值
    best_test_acc = max(best_test_acc, test_acc)
    best_test_loss = min(best_test_loss, test_loss)

# 保存本次实验的配置信息和实验结果
save_exp_config()
save_exp_results(trainer, best_test_acc, best_test_loss)

# 关闭Logger
Logger.close()
