import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np


def cal_false_alarm(normal_preds, threshold=0.5):
    """FAR sobre vídeos 100% normais do teste -- mesma convenção do PEL4VAD
    (test.py:cal_false_alarm), pra ficar comparável entre os dois."""
    if not normal_preds:
        return 0.0
    preds = (np.array(normal_preds) >= threshold).astype(int)
    return float(preds.mean())


def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        # --gt sobrescreve o ground truth default do dataset.
        if args.gt:
            gt = np.load(args.gt)
        elif args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        else:
            gt = np.load('list/gt-ucf.npy')
        gt_tmp = gt.copy()
        normal_preds = []

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

            # vídeo a vídeo (batch_size=1 no test_loader): acumula só os
            # 100% normais pra calcular FAR, igual ao PEL4VAD.
            seg_len = sig.shape[0] * 16
            video_labels = gt_tmp[:seg_len]
            if video_labels.sum() == 0:
                normal_preds.extend(np.repeat(sig.cpu().detach().numpy(), 16))
            gt_tmp = gt_tmp[seg_len:]

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)

        far = cal_false_alarm(normal_preds)

        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.plot_lines('far', far)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc, far, pr_auc

