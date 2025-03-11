import importlib

def load_model(model_class, device):
    module = importlib.import_module(f"src.models.{model_class}")
    model = getattr(module, model_class)()
    model.to(device)
    return model

model = load_model('ResNext', 'cuda')
print(model)