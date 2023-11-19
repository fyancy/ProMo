import numpy as np
from sklearn import metrics
from utils.plot_utils import wrong_right_dist, wrong_right_train_dist


def auc(ind_conf, ood_conf, score_probability: bool):
    m_ind, m_ood = np.mean(ind_conf), np.mean(ood_conf)
    conf = np.concatenate((ind_conf, ood_conf))
    # # method 1:
    # if m_ind > m_ood:
    #     ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    # else:
    #     ind_indicator = np.concatenate((np.zeros_like(ind_conf), np.ones_like(ood_conf)))

    # method 2:
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    if not score_probability:  # m_ind <= m_ood:  # uncertainty --> probability
        conf = max(conf) + 0.1 - conf
    # 保证: score大-->positive

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def num_fp_at_recall(ind_conf, ood_conf, tpr, score_mode: bool):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    # m_ind, m_ood = np.mean(ind_conf), np.mean(ood_conf)
    recall_num = int(np.floor(tpr * num_ind))
    if score_mode:  # m_ind >= m_ood:
        thresh = np.sort(ind_conf)[-recall_num]  # score,升序;越小越属于OOD
        num_fp = np.sum(ood_conf >= thresh)
    else:
        thresh = np.sort(ind_conf)[recall_num]  # unc,升序;越大越属于OOD
        num_fp = np.sum(ood_conf <= thresh)

    return num_fp, thresh


def num_fp_at_recall_v2(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    m_ind, m_ood = np.mean(ind_conf), np.mean(ood_conf)
    # recall_num = int(np.floor(tpr * num_ind))
    q1 = np.quantile(ind_conf, 0.25)
    q3 = np.quantile(ind_conf, 0.75)
    IQR = q3 - q1
    # https://blog.csdn.net/qq_38121967/article/details/89919607

    if m_ind >= m_ood:
        thresh = q1 - 1.5 * IQR
        num_fp = np.sum(ood_conf >= thresh)
    else:
        thresh = q3 + 1.5 * IQR
        num_fp = np.sum(ood_conf <= thresh)

    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr, score_mode):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr, score_mode=score_mode)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh


def get_threshold(X, mode='3sigma'):
    if mode.lower() == 'iqr':
        q1 = np.quantile(X, 0.25)
        q3 = np.quantile(X, 0.75)
        IQR = q3 - q1
        thresh = [q1 - 1.5 * IQR, q3 + 1.5 * IQR]
    elif mode == '3sigma':
        mean, std = np.mean(X), np.std(X)
        thresh = [mean - 3 * std, mean + 3 * std]
    else:
        exit()

    return thresh


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def compute_roc_AccU(uncer, probs, y):
    """

    :param uncer: (B, )
    :param probs: (B, C)
    :param y: (B, )
    :return:
    """

    def convert_to_numpy(x):
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        return x

    uncer = convert_to_numpy(uncer)
    probs = convert_to_numpy(probs)
    y = convert_to_numpy(y)

    y_pred = np.argmax(probs, axis=1)
    wrong_pred = (y != y_pred).astype(int)
    right_pred = (y == y_pred).astype(int)
    umin, umax = np.min(uncer), np.max(uncer)
    N_tot = np.prod(y.shape)

    # precision_, recall_, threshold = metrics.precision_recall_curve(wrong_pred, uncer)
    if np.sum(wrong_pred) == 0:  # all samples are classified correctly
        wrong_pred[np.argmax(uncer)] = 1
    fpr, tpr, threshold_ = metrics.roc_curve(wrong_pred, uncer)
    rcc, recall, acc, precision, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # wrong_ids = wrong_pred.astype(bool)
    # wrong_right_dist(uncer[wrong_ids], uncer[~wrong_ids], title="Wrong_right", fig_save=False)
    # uT = threshold  # recommend
    # uT = threshold_
    uT = [get_threshold(uncer)[1]]
    # print(f"{len(uT)} thresholds")
    counter = 0
    for ut in uT:
        t = (ut - umin) / (umax - umin)
        counter += 1
        uncertain = (uncer >= ut).astype(int)
        certain = (uncer < ut).astype(int)

        TP = np.sum(uncertain * wrong_pred)  # iu
        TN = np.sum(certain * right_pred)  # cc
        N_w = np.sum(wrong_pred) if np.sum(wrong_pred) > 0 else 1e-5  # incorrect, iu+ic
        N_c = np.sum(certain) if np.sum(certain) > 0 else 1e-5
        N_unc = np.sum(uncertain) if np.sum(uncertain) > 0 else 1e-5
        recall = np.append(recall, TP / N_w)  # iu/(iu+ic)
        rcc = np.append(rcc, TN / N_c)  # Rcc = cc/(cc+ic)
        precision = np.append(precision, TP / N_unc)  # iu/(iu+cu)
        acc = np.append(acc, (TN + TP) / N_tot)  # (iu+cc)/N_all
        T = np.append(T, t)

    # auc_ = auc(recall_, precision_)  # pr_auc
    roc_auc = metrics.auc(fpr, tpr)
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    print(f"AccU: {acc_mean:.2%}+/-{acc_std:.2%}, AUROC: {roc_auc:.2%}")


def compute_roc_AccU_v2(uncer, probs, y, uncer_train, ood_score, ood_is_unc, th_correct, th_ind):
    """

    :param uncer: (B, )
    :param probs: (B, C)
    :param y: (B, )
    :return:
    """

    def convert_to_numpy(x):
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        return x

    uncer = convert_to_numpy(uncer)
    probs = convert_to_numpy(probs)
    y = convert_to_numpy(y)

    y_pred = np.argmax(probs, axis=1)
    wrong_pred = (y != y_pred).astype(int)
    right_pred = (y == y_pred).astype(int)
    # umin, umax = np.min(uncer), np.max(uncer)
    N_tot = np.prod(y.shape)

    # precision_, recall_, threshold = metrics.precision_recall_curve(wrong_pred, uncer)
    if np.sum(wrong_pred) == 0:  # all samples are classified correctly
        wrong_pred[np.argmax(uncer)] = 1
    fpr, tpr, threshold_ = metrics.roc_curve(wrong_pred, uncer)
    rcc, recall, acc, precision, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    wrong_ids = wrong_pred.astype(bool)
    wrong_right_train_dist(uncer[wrong_ids], uncer[~wrong_ids], uncer_train,
                           title="Wrong_right_train", fig_save=False)
    # uT = threshold  # recommend
    # uT = threshold_
    # uT = [get_threshold(uncer)[1]]
    # print(f"{len(uT)} thresholds")
    counter = 0
    # for ut in uT:
    # t = (ut - umin) / (umax - umin)
    counter += 1
    if ood_is_unc:
        uncertain = ((uncer >= th_correct) & (ood_score >= th_ind)).astype(int)
        certain = ((uncer < th_correct) & (ood_score < th_ind)).astype(int)
    else:
        uncertain = ((uncer >= th_correct) & (ood_score <= th_ind)).astype(int)
        certain = ((uncer < th_correct) & (ood_score > th_ind)).astype(int)

    TP = np.sum(uncertain * wrong_pred)  # iu
    TN = np.sum(certain * right_pred)  # cc
    # N_w = np.sum(wrong_pred) if np.sum(wrong_pred) > 0 else 1e-5  # incorrect, iu+ic
    # N_c = np.sum(certain) if np.sum(certain) > 0 else 1e-5
    # N_unc = np.sum(uncertain) if np.sum(uncertain) > 0 else 1e-5
    # recall = np.append(recall, TP / N_w)  # iu/(iu+ic)
    # rcc = np.append(rcc, TN / N_c)  # Rcc = cc/(cc+ic)
    # precision = np.append(precision, TP / N_unc)  # iu/(iu+cu)
    acc = np.append(acc, (TN + TP) / N_tot)  # (iu+cc)/N_all
    # T = np.append(T, t)

    # auc_ = auc(recall_, precision_)  # pr_auc
    roc_auc = metrics.auc(fpr, tpr)
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    print(f"AccU-v2: {acc_mean:.2%}+/-{acc_std:.2%}, AUROC: {roc_auc:.2%}")


def compute_roc_AccU_v3(uncer_ind, probs, y, uncer_train, score_train, ind_score, score_is_unc):
    """

    :param uncer_ind: (B, )
    :param probs: (B, C)
    :param y: (B, )
    :return:
    """

    def convert_to_numpy(x):
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        return x

    uncer_ind = convert_to_numpy(uncer_ind)
    probs = convert_to_numpy(probs)
    y = convert_to_numpy(y)

    y_pred = np.argmax(probs, axis=1)
    wrong_pred = (y != y_pred).astype(int)
    right_pred = (y == y_pred).astype(int)
    # umin, umax = np.min(uncer), np.max(uncer)
    N_tot = np.prod(y.shape)

    # precision_, recall_, threshold = metrics.precision_recall_curve(wrong_pred, uncer)
    if np.sum(wrong_pred) == 0:  # all samples are classified correctly
        wrong_pred[np.argmax(uncer_ind)] = 1
    fpr, tpr, threshold_ = metrics.roc_curve(wrong_pred, uncer_ind)
    rcc, recall, acc, precision, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    wrong_ids = wrong_pred.astype(bool)
    print(f"wrong predictions: {sum(wrong_ids)}/{len(uncer_ind)}")
    wrong_right_train_dist(uncer_ind[wrong_ids], uncer_ind[~wrong_ids], uncer_train,
                           title="Wrong_right_train", fig_save=False)

    # th_correct = get_threshold(uncer_ind)[1]
    th_correct = get_threshold(uncer_train)[1]
    counter = 0
    counter += 1
    if score_is_unc:  # use uncertainty as ood score, so certain InD data should be smaller than threshold(unc)
        th_ind_from_train = get_threshold(score_train)[1]
        uncertain = ((uncer_ind >= th_correct) & (ind_score >= th_ind_from_train)).astype(int)
        certain = ((uncer_ind < th_correct) & (ind_score < th_ind_from_train)).astype(int)
    else:  # use weight score as ood score, so certain InD data should be bigger than threshold(score)
        th_ind_from_train = get_threshold(score_train)[0]
        uncertain = ((uncer_ind >= th_correct) & (ind_score <= th_ind_from_train)).astype(int)
        certain = ((uncer_ind < th_correct) & (ind_score > th_ind_from_train)).astype(int)
        # uncertain = (uncer_ind >= th_correct).astype(int)
        # certain = (uncer_ind < th_correct).astype(int)
        # uncertain = (ind_score <= th_ind_from_train).astype(int)
        # certain = (ind_score > th_ind_from_train).astype(int)

    TP = np.sum(uncertain * wrong_pred)  # iu
    TN = np.sum(certain * right_pred)  # cc
    # N_w = np.sum(wrong_pred) if np.sum(wrong_pred) > 0 else 1e-5  # incorrect, iu+ic
    # N_c = np.sum(certain) if np.sum(certain) > 0 else 1e-5
    # N_unc = np.sum(uncertain) if np.sum(uncertain) > 0 else 1e-5
    # recall = np.append(recall, TP / N_w)  # iu/(iu+ic)
    # rcc = np.append(rcc, TN / N_c)  # Rcc = cc/(cc+ic)
    # precision = np.append(precision, TP / N_unc)  # iu/(iu+cu)
    acc = np.append(acc, (TN + TP) / N_tot)  # (iu+cc)/N_all
    # T = np.append(T, t)

    # auc_ = auc(recall_, precision_)  # pr_auc
    roc_auc = metrics.auc(fpr, tpr)
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    print(f"AccU-v3: {acc_mean:.2%}+/-{acc_std:.2%}, AUROC: {roc_auc:.2%}")
