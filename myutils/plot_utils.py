
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def ood_score_dist(train_score, ind_score, ood_score, title, fig_save=False):
    num_cols = 1
    fig = plt.figure(figsize=(int(3 * num_cols), 2 * 1), dpi=300)  # W x H, in inches
    plt.rc('font', family='Arial', weight='normal', size=8)
    plt.rc('lines', markersize=5)

    s_min = min(min(train_score), min(ind_score), min(ood_score)) - 0.05
    s_max = max(max(np.abs(train_score)), max(np.abs(ind_score)), max(np.abs(ood_score)))
    train_score, ind_score, ood_score = (train_score - s_min) / s_max, (ind_score - s_min) / s_max, (
                ood_score - s_min) / s_max

    bw = 2.0
    sns.kdeplot(train_score, fill=True, label="train", bw_adjust=bw)
    sns.kdeplot(ind_score, fill=True, label="ind", bw_adjust=bw)
    sns.kdeplot(ood_score, fill=True, label="ood", bw_adjust=bw)
    plt.legend(loc='upper right')
    plt.title(title)

    # plt.ylim([0, 30])
    # plt.tight_layout()
    path = fr"E:\桌面\MyWork\贝叶斯深度学习\实验图\{title}.pdf"
    if not os.path.exists(path) and fig_save:
        print(f"Save fig at \n {path}")
        plt.savefig(path, bbox_inches='tight')
    plt.show()


# 6 methods: NuSA, MSP, ViM, UaDE+, MCDrop, ProMo
def pred_ood_dist(pred_unc_ind, pred_unc_ood, ind_score, ood_score, title, fig_save=False):
    w = 4 / 2.54
    # plt.rc('font', family='Arial', weight='normal', size=8)
    plt.rc('lines', markersize=3)
    fig = plt.figure(1, figsize=(w, w))
    sns.set_style('darkgrid')

    s_min = min(min(ind_score), min(ood_score)) - 0.05
    s_max = max(max(np.abs(ind_score)), max(np.abs(ood_score)))
    ind_score, ood_score = (ind_score - s_min) / s_max, (ood_score - s_min) / s_max

    s_min = min(min(pred_unc_ind), min(pred_unc_ood)) - 0.05
    s_max = max(max(np.abs(pred_unc_ind)), max(np.abs(pred_unc_ood)))
    pred_unc_ind, pred_unc_ood = (pred_unc_ind - s_min) / s_max, (pred_unc_ood - s_min) / s_max

    data = {"pred_unc": np.concatenate([pred_unc_ind, pred_unc_ood]),
            "ood_score": np.concatenate([ind_score, ood_score]),
            "label": ['InD']*len(pred_unc_ind)+['OOD']*len(pred_unc_ood)}
    df = pd.DataFrame(data)
    # print(len(pred_unc_ind), len(pred_unc_ood))

    grid = sns.jointplot(x='pred_unc', y='ood_score', data=df, hue="label",
                         alpha=0.4, lw=0.3, legend=False)
    # https://seaborn.pydata.org/generated/seaborn.jointplot.html
    grid.fig.set_figwidth(w)
    grid.fig.set_figheight(w)
    grid.set_axis_labels('', '')
    # grid._legend.remove()

    # plt.ylim([0, 30])
    path = fr"E:\桌面\MyWork\贝叶斯深度学习\实验图\{title}.pdf"
    if not os.path.exists(path) and fig_save:
        print(f"Save fig at \n {path}")
        plt.savefig(path, bbox_inches='tight')
    # plt.savefig(path, bbox_inches='tight')
    plt.show()


def wrong_right_dist(wrong_score, right_score, title, fig_save=False):
    num_cols = 1
    fig = plt.figure(figsize=(int(3 * num_cols), 2 * 1), dpi=200)  # W x H, in inches
    plt.rc('font', family='Arial', weight='normal', size=8)
    plt.rc('lines', markersize=5)

    # s_min = min(min(wrong_score), min(right_score)) - 0.05
    # s_max = max(max(np.abs(wrong_score)), max(np.abs(right_score)))
    # wrong_score, right_score = (wrong_score - s_min) / s_max, (right_score - s_min) / s_max

    bw = 2.0
    score = np.concatenate([wrong_score, right_score], 0)
    sns.kdeplot(score, fill=True, label="all", bw_adjust=bw)
    sns.kdeplot(wrong_score, fill=True, label="wrong", bw_adjust=bw, warn_singular=False)
    sns.kdeplot(right_score, fill=True, label="right", bw_adjust=bw)
    plt.legend(loc='upper right')
    plt.title(title)
    # plt.ylim([0, 5])
    # plt.tight_layout()
    # path = fr"E:\桌面\MyWork\贝叶斯深度学习\实验图\{title}.pdf"
    # if not os.path.exists(path) and fig_save:
    #     print(f"Save fig at \n {path}")
    #     plt.savefig(path, bbox_inches='tight')
    plt.show()


def wrong_right_train_dist(wrong_score, right_score, train_score, title, fig_save=False):
    num_cols = 1
    fig = plt.figure(figsize=(int(3 * num_cols), 2 * 1), dpi=200)  # W x H, in inches
    plt.rc('font', family='Arial', weight='normal', size=8)
    plt.rc('lines', markersize=5)

    # s_min = min(min(wrong_score), min(right_score)) - 0.05
    # s_max = max(max(np.abs(wrong_score)), max(np.abs(right_score)))
    # wrong_score, right_score = (wrong_score - s_min) / s_max, (right_score - s_min) / s_max

    bw = 2.0
    score = np.concatenate([wrong_score, right_score], 0)
    sns.kdeplot(score, fill=True, label="all", bw_adjust=bw)
    sns.kdeplot(wrong_score, fill=True, label="wrong", bw_adjust=bw, warn_singular=False)
    sns.kdeplot(right_score, fill=True, label="right", bw_adjust=bw)
    sns.kdeplot(train_score, fill=True, label="train", bw_adjust=bw)
    plt.legend(loc='upper right')
    plt.title(title)
    # plt.ylim([0, 5])
    # plt.tight_layout()
    # path = fr"E:\桌面\MyWork\贝叶斯深度学习\实验图\{title}.pdf"
    # if not os.path.exists(path) and fig_save:
    #     print(f"Save fig at \n {path}")
    #     plt.savefig(path, bbox_inches='tight')
    plt.show()


def wrong_right_train_ind_dist(wrong_score, right_score, train_score, train95_score, ood_score, title,
                               fig_path=None):
    num_cols = 1
    plt.figure(figsize=(int(3 * num_cols), 2 * 1), dpi=200)  # W x H, in inches
    plt.rc('font', family='Arial', weight='normal', size=8)
    plt.rc('lines', markersize=5)

    # s_min = min(min(wrong_score), min(right_score)) - 0.05
    # s_max = max(max(np.abs(wrong_score)), max(np.abs(right_score)))
    # wrong_score, right_score = (wrong_score - s_min) / s_max, (right_score - s_min) / s_max

    bw = 2.0
    # score = np.concatenate([wrong_score, right_score], 0)
    # sns.kdeplot(score, fill=True, label="all", bw_adjust=bw)
    sns.kdeplot(train_score, fill=True, label="Seen InD", bw_adjust=bw)
    # sns.kdeplot(train95_score, fill=True, label="train95", bw_adjust=bw)
    sns.kdeplot(wrong_score, fill=True, label="Unseen InD Wrong", bw_adjust=bw, warn_singular=False)
    sns.kdeplot(right_score, fill=True, label="Unseen InD Right", bw_adjust=bw)
    sns.kdeplot(ood_score, fill=True, label="Unseen OOD", bw_adjust=bw)
    plt.legend(loc='upper right')
    # plt.title("Ind_Wrong_Right_OOD")
    plt.title(title)
    # plt.title("Train_Wrong_Right_OOD")
    # plt.ylim([0, 5])
    # plt.tight_layout()
    # path = fr"E:\桌面\MyWork\贝叶斯深度学习\实验图\{title}.pdf"
    # path = fr"C:\Users\asus\Desktop\MyWork\贝叶斯深度学习\实验图\exp4\{title}.pdf"
    if fig_path:
        if not os.path.exists(fig_path):
            print(f"Save fig at \n {fig_path}")
            plt.savefig(fig_path, bbox_inches='tight')
    plt.show()
