import torchvision.transforms as transforms

def build_transforms(mean, std):
    train_transform = transforms.Compose([
        transforms.Resize(256),  # 将输入图像大小调整为256x256像素
        transforms.RandomCrop(224),  # 从256x256的图像中随机裁剪出224x224的图像区域
        transforms.RandomHorizontalFlip(),  # 以一定的概率随机水平翻转图像
        transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor，并缩放到[0,1]
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return  train_transform, test_transform
