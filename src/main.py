from src.Trainer  import Trainer
from src.infer import infer
from src.utils.save_exp_config_and_results import save_history_values, save_test_epoch_results, save_train_epoch_results
from src.utils.load_model import load_model
from src.utils.save_exp_config_and_results import save_exp_config, save_exp_plot
from src.utils.visualize import imshow_grid, print_test_epoch_result
from src.data.loaders import get_dataloader
from src.utils.logger import ExperimentLogger
from src.config import *

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
save_model_path = ''

print(f"use the model {MODEL_CLASS}")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    train_loss, train_acc = trainer.train_epoch()
    Logger.log_metrics({'train_loss':train_loss}, epoch)
    Logger.log_metrics({'train_loss':train_loss}, epoch)

    dic, save_model_path = trainer.test_epoch()

    test_loss, test_acc, precision, recall, roc_auc, test_f1, cm= dic.values()

    Logger.log_metrics({'test_loss':test_loss}, epoch)
    Logger.log_metrics({'test_acc':test_acc}, epoch)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    print_test_epoch_result(test_loss, test_acc, precision, recall, roc_auc, test_f1, cm)

    # 保存每轮的实验结果
    save_train_epoch_results(epoch, train_loss, train_acc)
    save_test_epoch_results(epoch, test_loss, test_acc, precision, recall, roc_auc, test_f1, cm)


infer_acc = infer(save_model_path)
# 保存本次实验的配置信息
save_exp_config()
save_exp_plot(trainer)
# save_history_values(trainer)

# 关闭Logger
Logger.close()
