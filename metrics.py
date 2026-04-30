import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def top_k_accuracy(outputs, targets, k=1):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred    = outputs.topk(k, dim=1, largest=True, sorted=True)
        pred       = pred.t()
        correct    = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k  = correct[:k].reshape(-1).float().sum(0)
        return (correct_k * 100.0 / batch_size).item()

def top1_accuracy(outputs, targets):
    return top_k_accuracy(outputs, targets, k=1)

def top5_accuracy(outputs, targets):
    return top_k_accuracy(outputs, targets, k=5)

def per_class_accuracy(all_preds, all_targets, class_names):
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    per_class   = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = (all_targets == cls_idx)
        if mask.sum() == 0:
            per_class[cls_name] = None
        else:
            correct = (all_preds[mask] == cls_idx).sum()
            per_class[cls_name] = float(correct) / float(mask.sum()) * 100.0
    return per_class

def compute_confusion_matrix(all_preds, all_targets, num_classes):
    return confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

def compute_precision_recall_f1(all_preds, all_targets, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds,
        labels=list(range(len(class_names))),
        average=None, zero_division=0
    )
    return {cls_name: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
            for i, cls_name in enumerate(class_names)}

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    model.to(device)
    all_preds, all_targets, all_outputs = [], [], []
    print('Evaluating on ' + str(len(dataloader.dataset)) + ' images...')
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images  = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
            if batch_idx % 50 == 0:
                print('  Batch ' + str(batch_idx) + '/' + str(len(dataloader)))
    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.tensor(all_targets)
    top1 = top_k_accuracy(all_outputs_tensor, all_targets_tensor, k=1)
    top5 = top_k_accuracy(all_outputs_tensor, all_targets_tensor, k=5)
    pca  = per_class_accuracy(all_preds, all_targets, class_names)
    cm   = compute_confusion_matrix(all_preds, all_targets, len(class_names))
    prf  = compute_precision_recall_f1(all_preds, all_targets, class_names)
    print('Top-1: ' + str(round(top1,2)) + '%  |  Top-5: ' + str(round(top5,2)) + '%')
    return {'top1_acc': top1, 'top5_acc': top5, 'per_class_acc': pca,
            'confusion_matrix': cm, 'precision_recall_f1': prf,
            'all_preds': np.array(all_preds), 'all_targets': np.array(all_targets)}
