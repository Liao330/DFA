import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, f1_score
from tqdm import tqdm

from src.config import *
from src.utils.load_model import load_model

def infer(rank, model_class, val_loader, model_path):


    model = load_model(model_class, DEVICE)
    path_to_model = model_path
    model.load_state_dict(torch.load(path_to_model))

    model.eval()
    correct = 0
    total = 0
    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        print('===> Infer Start!')
        pbar = tqdm(val_loader, desc="Infering", unit="batch")
        for data_dict in pbar:
            img_inputs, lm_inputs, labels = data_dict.values()
            data_dict = {key: value.to(DEVICE) for key, value in data_dict.items()}
            img_inputs, lm_inputs, labels = img_inputs.to(DEVICE), lm_inputs.to(DEVICE), labels.to(
                DEVICE)

            if model_class == 'DFACLIP':
                pred_dict = model(data_dict)
                outputs = pred_dict['cls']
            else:
                outputs = model(img_inputs)

            all_true.append(labels.cpu())
            all_pred.append(torch.argmax(outputs.data, dim=1).cpu())
            all_prob.append(outputs.data.cpu())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'acc': 100 * correct / total
            }, refresh=False)

        # 使用 torch.cat 拼接张量
    all_true = torch.cat(all_true, dim=0).cpu().numpy()
    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_prob = torch.cat(all_prob, dim=0).cpu().numpy()
    epoch_acc = 100 * correct / total

    all_true_flat = np.array(all_true).flatten()
    all_pred_flat = np.array(all_pred).flatten()

    cm = confusion_matrix(all_true_flat, all_pred_flat)
    if rank == 0:
        print("Confusion Matrix:\n", cm)
    precision = precision_score(all_true_flat, all_pred_flat, average='macro')
    recall = recall_score(all_true_flat, all_pred_flat, average='macro')
    fpr, tpr, _ = roc_curve(all_true_flat, all_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    test_f1 = f1_score(all_true_flat, all_pred_flat, average='macro')
    video_auc = 1111111111111111111

    dic = {
        'acc': epoch_acc,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'test_f1': test_f1,
        'cm': cm,
        'video_auc': video_auc,
    }
    print('===> Infer Done!')
    return dic