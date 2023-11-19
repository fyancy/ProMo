import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interpolate
import os


def fit_xy(x, y, x_new):
    assert min(x_new) >= min(x), max(x_new) <= max(x)
    # filter some points, where the same x appears 2 times
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    x, y, x_new = np.asarray(x), np.asarray(y), np.asarray(x_new)
    f = interpolate.interp1d(x, y, kind='linear')
    # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
    #  'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
    y_new = f(x_new)

    x_smooth = np.linspace(x_new.min(), x_new.max(), 100)
    y_new = interpolate.make_interp_spline(x_new, y_new, k=3)(x_smooth)

    y_new = np.clip(y_new, 0., 1.)
    return y_new


def polyfit_xy(x, y, x_new):
    assert min(x_new) >= min(x), max(x_new) <= max(x)
    # filter some points, where the same x appears 2 times
    x, idx = np.unique(x, return_index=True)
    y = y[idx]
    x, y, x_new = np.asarray(x), np.asarray(y), np.asarray(x_new)
    coeff = np.polyfit(x, y, 3)
    # f = interpolate.interp1d(x, y, kind='quadratic')
    y_new = np.polyval(coeff, x_new)
    y_new = np.clip(y_new, 0., 1.)
    return y_new


def get_curve_value(ind_x, ood_xx, unc_mode, scale=1.):
    pos_label = 0 if unc_mode else 1
    # pos_label = 1
    FPR, TPR = [], []
    for ood_x in ood_xx:
        y_true = np.concatenate((np.ones_like(ind_x), np.zeros_like(ood_x)))
        y_score = np.concatenate([ind_x, ood_x], 0)
        # y_score = max(y_score) + 0.1 - y_score if unc_mode else y_score
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
        FPR.append(fpr)
        TPR.append(tpr)

    ood_x = np.concatenate(ood_xx, 0)
    y_true = np.concatenate((np.ones_like(ind_x), np.zeros_like(ood_x)))
    y_score = np.concatenate([ind_x, ood_x], 0)
    # y_score = max(y_score) + 0.1 - y_score if unc_mode else y_score
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    print(f"ind-ood auc-roc: {metrics.auc(fpr, tpr):.2%}")

    # draw:
    x_new = np.linspace(0, 1, 1000)
    Y_new = [fit_xy(FPR[i], TPR[i], x_new) for i in range(len(FPR))]
    Y_avg = fit_xy(fpr, tpr, x_new)

    Y_max = np.max(Y_new, 0)
    Y_min = np.min(Y_new, 0)

    Y_max = Y_avg + scale * (Y_max - Y_avg)
    Y_min = Y_avg - scale * (Y_avg - Y_min)

    # plt.plot(x_new, Y_avg, color='r')
    # x_new = np.linspace(0, 1, 100)
    # plt.fill_between(x_new, Y_min, Y_max, facecolor='r', edgecolor=None, alpha=0.1)
    # for i in range(len(FPR)):
    #     plt.plot(x_new, Y_new[i])
    # # # plt.plot(fpr, tpr, c='r')
    # # plt.plot(x_new, fit_xy(fpr, tpr, x_new), c='r')
    # # # plt.plot(x_new, polyfit_xy(fpr, tpr, x_new), c='r')
    # plt.show()

    return Y_min, Y_max, Y_avg


def draw_roc(x_new, Y_min, Y_max, Y_avg):
    plt.plot(x_new, Y_avg, color='r')
    plt.fill_between(x_new, Y_min, Y_max, facecolor='r', edgecolor=None, alpha=0.1)


def save_curve_and_plot(y_ood, score_ood, score_ind, unc_mode: bool, score_save_path):
    y_ood = y_ood.cpu().detach()
    ood_xx = []
    for i in np.unique(y_ood):
        s_ood_i = score_ood[y_ood == i]
        ood_xx.append(s_ood_i)

    if os.path.exists(score_save_path+r'.npy'):
        score_save_path = score_save_path + r'(copy)'
        print("path exists, new path: {}".format(score_save_path))

    np.save(score_save_path, {'ind': score_ind, 'ood': ood_xx})
    print(f"Data are saved at: \n{score_save_path}\n")
    x_new = np.linspace(0, 1, 100)
    y_min, y_max, y_avg = get_curve_value(score_ind, ood_xx, unc_mode=unc_mode, scale=0.4)
    draw_roc(x_new, y_min, y_max, y_avg)
    plt.show()


if __name__ == "__main__":
    SCALE = 0.4  # for exp3
    data = np.load(r"E:\fy_works\save_model\bcnn\discussion\ind_ood_scores\exp3\flipout.npy",
                   allow_pickle=True).item()

    # SCALE = 0.3  # for exp4
    # data = np.load(r"E:\fy_works\save_model\bcnn\discussion\ind_ood_scores\exp4\flipout.npy",
    #                allow_pickle=True).item()

    ind_x, ood_xx = data['ind'], data['ood']

    x_new = np.linspace(0, 1, 100)
    ymin, ymax, yavg = get_curve_value(ind_x, ood_xx, unc_mode=True, scale=SCALE)
    draw_roc(x_new, ymin, ymax, yavg)

    plt.legend(['flipout'])
    plt.show()
