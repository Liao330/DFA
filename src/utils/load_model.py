import sys
import os
import importlib

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
def load_model(model_class, device):
    # module = importlib.import_module(f"src.models.{model_class}")
    if model_class[:7] == "DFACLIP":
        # 特殊处理 DFACLIP 的路径
        module = importlib.import_module(f"src.models.DFACLIP.{model_class}")
    else:
        # 其他模型的路径
        module = importlib.import_module(f"src.models.{model_class}")
    model = getattr(module, model_class)()
    model.to(device)
    return model

# model = load_model('ResNext', 'cuda')
# print(model)