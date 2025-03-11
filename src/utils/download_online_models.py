from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
from src.utils.proxy import ProxyManager

proxy = ProxyManager()

def get_clip_visual(proxy=proxy, model_name="openai/clip-vit-base-patch16"):
    proxy.set_proxy()
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    proxy.unset_proxy()
    return processor, model.vision_model


def get_vit_model(proxy=proxy, model_name="google/vit-base-patch16-224-in21k"):
    proxy.set_proxy()
    # processor = AutoProcessor.from_pretrained(model_name)
    configuration = ViTConfig(
        image_size=224,
    )
    model = ViTModel.from_pretrained(model_name, config=configuration)
    proxy.unset_proxy()
    return None, model