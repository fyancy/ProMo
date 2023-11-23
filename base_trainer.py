import os
import time

import torch
import numpy as np
import visdom
from collections import OrderedDict
from torch.utils.data import DataLoader

from comparison.cnn_based.netsV2 import WCNN, ConvNet1
from data.data_base_bogie import BogieDatasetTorch
from utils.init_utils import set_seed_torch
from data.data_gearbox import GearBoxDataset_load_0, GearBoxDataset_load_1
from data.data_base import GearBoxDatasetTrainL2L
from utils.unc_utils import moment_unc_Kwon
from utils.plot_utils import ood_score_dist, pred_ood_dist, wrong_right_train_ind_dist
from utils.ood_scores import vim, NuSA, Residual, energy, energy_react, KL_matching, Mahalanobis, MSP
from utils.vim_utils import auc, fpr_recall, compute_roc_AccU, compute_roc_AccU_v3, get_threshold
from utils.curve_fit import save_curve_and_plot


class BaseCNNTrainer:
    def __init__(self, exp_num, figure_save=False, ood_detect_single_cls=False):
        set_seed_torch()
        self.nway = 4
        self.exp_num = exp_num
        self.model = ConvNet1(self.nway, drop_rate=0.).cuda()
        self.SaveFigure = figure_save
        self.vim_alpha = 1.0
        self.score_fn_name = "vim"
        self.ood_detect_single_cls = ood_detect_single_cls
        self.use_preprocess = True

    def construct_dataset(self, mode="train"):
        if self.exp_num in [2, 3]:
            self.construct_seu(mode)
        elif self.exp_num == 4:
            self.construct_bogie(mode)
        else:
            exit("Exp number error")

        print(f"\n----- Exp {self.exp_num} -----\n")

    def construct_seu(self, mode="train"):
        train_cls = ["bearing_health", "bearing_ball", "bearing_comb", "bearing_outer"]
        if mode == 'train' or mode == 'ood':
            train_dict = OrderedDict({k: GearBoxDataset_load_0[k] for k in train_cls})
            self.train_dataset = GearBoxDatasetTrainL2L("train", train_dict, train_num_per_cls=100,
                                                        test_num_per_cls=200, chn_select=[1, 2, 3],
                                                        use_preprocess=self.use_preprocess)  # 50*4 = 200

        if self.exp_num == 3:
            valid_dict = OrderedDict({k: GearBoxDataset_load_1[k] for k in train_cls})  # exp3
            print("\n====== Exp: 3 ======\n")
            self.exp_name = "Exp3"
        elif self.exp_num == 2:
            valid_dict = OrderedDict({k: GearBoxDataset_load_0[k] for k in train_cls})  # exp2
            print("\n====== Exp: 2 ======\n\n")
            self.exp_name = "Exp2"
        else:
            exit("Exp number error")

        self.valid_dataset = GearBoxDatasetTrainL2L("valid", valid_dict, train_num_per_cls=100,
                                                    test_num_per_cls=200, chn_select=[1, 2, 3],
                                                    use_preprocess=self.use_preprocess)
        if mode == "ood":
            ood_cls = ["bearing_inner", "gear_miss", "gear_chipped", "gear_root"]
            ood_dict = OrderedDict({k: GearBoxDataset_load_0[k] for k in ood_cls})
            self.ood_dataset = GearBoxDatasetTrainL2L("valid", ood_dict, train_num_per_cls=100,
                                                      test_num_per_cls=200, chn_select=[1, 2, 3],
                                                      use_preprocess=self.use_preprocess)

    def construct_bogie(self, mode="train"):
        # https://github.com/cathysiyu/Mechanical-datasets
        # https://www.mitssolutions.asia/drivetrain-diagnostics-simulator-dds
        # todo: assume that we only simulated the crack faults, and got no info for pitting fault.
        train_classes = ["bear_norm", "bear_outer_crack_h", "bear_cage_crack", "bear_roller_crack_h"]
        ind_classes = ["bear_norm", "bear_outer_crack_l", "bear_cage_crack", "bear_roller_crack_h"]
        ood_classes_gear = ["gear_pitt", "gear_lack"]
        ood_classes_bear = ["bear_outer_pitt_m", "bear_roller_pitt_l"]
        # test samples have covariate shift, and have Unknown known classes and Unknown unknown classes
        # ood_classes_bear = ["bear_outer_pitt_m", "bear_roller_pitt_l", "bear_roller_pitt_m", "bear_roller_pitt_h"]
        self.exp_name = "Exp4"

        if mode == "train" or mode == 'ood':
            self.train_dataset = BogieDatasetTorch("train", train_classes,
                                                   file_name="BogieBear_rpm2000_load20",
                                                   train_num_per_cls=50, test_num_per_cls=0,
                                                   use_preprocess=self.use_preprocess)  # 50*4 = 200

        self.valid_dataset = BogieDatasetTorch("valid", ind_classes,
                                               file_name="BogieBear_rpm2000_load0",
                                               train_num_per_cls=0, test_num_per_cls=50,
                                               use_preprocess=self.use_preprocess)
        if mode == "ood":
            ood_bear = BogieDatasetTorch("valid", ood_classes_bear,
                                         file_name="BogieBear_rpm2000_load0",
                                         train_num_per_cls=0, test_num_per_cls=50,
                                         use_preprocess=self.use_preprocess)
            ood_gear = BogieDatasetTorch("valid", ood_classes_gear,
                                         file_name="BogieGear_rpm2000_load15",
                                         train_num_per_cls=0, test_num_per_cls=50,
                                         use_preprocess=self.use_preprocess)
            ood_gear.x = np.concatenate([ood_gear.x, ood_bear.x], 0)
            ood_gear.y = np.concatenate([ood_gear.y, ood_bear.y + len(ood_classes_gear)], 0)
            print(f"ood samples: {ood_gear.x.shape, ood_gear.y.shape}")
            self.ood_dataset = ood_gear

    @staticmethod
    def get_auc_fpr(score_ind, score_ood, score_prob):
        auc_ood = auc(score_ind, score_ood, score_prob)[0]
        fpr_ood, _ = fpr_recall(score_ind, score_ood, 0.95, score_mode=score_prob)
        # ======================= 注意
        # 这里在计算FPR的时候是以所有IND data为基础计算的，如果想要获得更一般的结果，
        # 应该使用correctly predicted IND data 这样可以排除模型预测不准的干扰，不苛求分类性能
        # 论文里没有这样做，因为我同时考虑了分类性能和OOD性能，分类也必须好。
        # ======================
        return auc_ood, fpr_ood

    def unc_fn(self, logits_train, logits_ind, logits_ood, eval_all_unc=True, ret_all_unc=False):
        t0 = time.time()
        all_train, ale_train, epi_train = moment_unc_Kwon(logits_train, logit=True)[-3:]
        all_ind, ale_ind, epi_ind = moment_unc_Kwon(logits_ind, logit=True)[-3:]
        all_ood, ale_ood, epi_ood = moment_unc_Kwon(logits_ood, logit=True)[-3:]
        tt = time.time() - t0
        print(f"time for three unc types: {tt} /s")
        # entropy_unc_Gal
        # all_train, ale_train, epi_train = entropy_unc_Gal(logits_train, logit=True)[-3:]
        # all_ind, ale_ind, epi_ind = entropy_unc_Gal(logits_ind, logit=True)[-3:]
        # all_ood, ale_ood, epi_ood = entropy_unc_Gal(logits_ood, logit=True)[-3:]

        if eval_all_unc:
            auc_ood, fpr_ood = self.get_auc_fpr(all_ind, all_ood, score_prob=False)
            print(f'[unc_all] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
            auc_ood, fpr_ood = self.get_auc_fpr(ale_ind, ale_ood, score_prob=False)
            print(f'[unc_ale] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
            auc_ood, fpr_ood = self.get_auc_fpr(epi_ind, epi_ood, score_prob=False)
            print(f'[unc_epi] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')

        # ood_score_dist(all_train, all_ind, all_ood, title="CNN_unc_all", fig_save=self.SaveFigure)
        # # ood_score_dist(ale_train, ale_ind, ale_ood, title="promo_w_unc_ale", fig_save=self.SaveFigure)
        # # ood_score_dist(epi_train, epi_ind, epi_ood, title="promo_w_unc_epi", fig_save=self.SaveFigure)

        if ret_all_unc:
            return (all_train, all_ind, all_ood), (ale_train, ale_ind, ale_ood), (epi_train, epi_ind, epi_ood)
        else:
            return all_train, all_ind, all_ood
        # return epi_train, epi_ind, epi_ood

    def get_ood_scores(self, logits, feats, labels, score_fn_name: str):
        feats_train, feats_ind, feats_ood = feats  # logits-(N, dim)
        logits_train, logits_ind, logits_ood = logits  # logits-(N, nc)
        labels_train, labels_ind, labels_ood = labels
        score_fn_name = score_fn_name.lower()
        s_train, s_ind, s_ood = None, None, None

        if score_fn_name == "energy":
            s_train, s_ind, s_ood = energy(logits_train, logits_ind, logits_ood)
        elif score_fn_name == "energy_react":
            s_train, s_ind, s_ood = energy_react(feats_train, feats_ind, feats_ood, self.model.classifier)
        elif score_fn_name == "vim":
            s_train, s_ind, s_ood = vim(feats_train, feats_ind, feats_ood,
                                        self.model.classifier.weight,
                                        self.model.classifier.bias, vim_alpha=self.vim_alpha)
        elif score_fn_name == "nusa":
            s_train, s_ind, s_ood = NuSA(self.model.classifier.weight, feats_train, feats_ind, feats_ood)
        elif score_fn_name == "mahalanobis":
            s_train, s_ind, s_ood = Mahalanobis(self.nway, feats_train, labels_train, feats_ind, feats_ood)
        elif score_fn_name == "residual":
            s_train, s_ind, s_ood = Residual(feats_train, feats_ind, feats_ood, self.model.classifier)
        elif score_fn_name == "kl_matching":
            probs_train, probs_ind, probs_ood = torch.softmax(logits_train, -1), torch.softmax(logits_ind, -1), \
                                                torch.softmax(logits_ood, -1)
            s_train, s_ind, s_ood = KL_matching(self.nway, probs_train, probs_ind, probs_ood)
        elif score_fn_name == "msp":
            s_train, s_ind, s_ood = MSP(logits_train, logits_ind, logits_ood)
        else:
            exit("score function type error!")

        return s_train, s_ind, s_ood

    def ood_fn(self, logits, feats, labels, score_fn_name: str, ensemble=False):
        if not ensemble:
            score_train, score_ind, score_ood = self.get_ood_scores(logits, feats, labels, score_fn_name)
        else:
            feats_train, feats_ind, feats_ood = feats
            logits_train, logits_ind, logits_ood = logits
            labels_train, labels_ind, labels_ood = labels
            score_train, score_ind, score_ood = [], [], []
            # t0 = time.time()
            for i in range(len(logits_train)):
                s_train, s_ind, s_ood = self.get_ood_scores([logits_train[i], logits_ind[i], logits_ood[i]],
                                                            [feats_train[i], feats_ind[i], feats_ood[i]],
                                                            [labels_train[i], labels_ind[i], labels_ood[i]],
                                                            score_fn_name)
                score_train.append(s_train)
                score_ind.append(s_ind)
                score_ood.append(s_ood)
            # tt = time.time() - t0
            # print(f"time for three sco types: {tt} /s")
            score_train, score_ind, score_ood = np.mean(score_train, 0), np.mean(score_ind, 0), np.mean(score_ood, 0)

        auc_ood = auc(score_ind, score_ood, score_probability=True)[0]
        fpr_ood, _ = fpr_recall(score_ind, score_ood, 0.95, score_mode=True)
        print(f'[{score_fn_name} score] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')

        ood_score_dist(score_train, score_ind, score_ood, title=f"{self.exp_name}_CNN-based_{score_fn_name}",
                       fig_save=self.SaveFigure)
        return score_train, score_ind, score_ood

    def train(self, model_path):
        self.construct_dataset(mode='ood')
        batch_size = 64
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        vis = visdom.Visdom(env='ProMo')
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fun = torch.nn.CrossEntropyLoss(reduction='mean')
        Epochs = 1000
        counter = 0
        Episodes = self.train_dataset.__len__() // batch_size
        print(f"Let's train {self.model.name}!")
        # early_stopping = EarlyStopping(patience=10, verbose=True, model_path=model_path)

        for ep in range(Epochs):
            valid_loader = iter(DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True))
            for epi, (bx, by) in enumerate(train_loader):
                bx, by = bx.cuda(), by.cuda().long()
                logits = self.model(bx)

                loss = loss_fun(logits, by)
                loss.backward()
                opt.step()
                opt.zero_grad()

                tr_acc = (logits.argmax(1) == by).float().mean().item()
                loss_train = loss.detach().cpu().item()

                # print()
                # print("[ep-{} Train] acc: {:.4f}, loss: {:.4f}".format(ep + 1, tr_acc, loss_train))

                # if (epi + 1) % 2 == 0:
                bx_ind, by_ind = valid_loader.__next__()
                bx_ind, by_ind = bx_ind.cuda(), by_ind.cuda().long()

                self.model.eval()
                with torch.no_grad():
                    lg = self.model(bx_ind)  # (N2, nc)
                    acc_ind = (lg.argmax(1) == by_ind).float().mean().item()
                    loss_ind = loss_fun(lg, by_ind).detach().cpu().item()
                self.model.train()

                # model, val_acc=None, val_loss=None, save_name=None
                print(f"[ep-{ep + 1}/{Epochs}, epi-{epi + 1}/{Episodes}] "
                      f"Valid Acc.:{acc_ind:.2%}, Loss: {loss_ind:.4f}")

                vis.line(Y=[[loss_train, loss_ind]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Loss_CNN_{self.exp_name}',
                         opts=dict(legend=['train', 'val'], title=f'Loss_CNN_{self.exp_name}'))
                vis.line(Y=[[tr_acc, acc_ind]], X=[counter],
                         update=None if counter == 0 else 'append', win=f'Acc_CNN_{self.exp_name}',
                         opts=dict(legend=['train', 'val'], title=f'Acc_CNN_{self.exp_name}'))
                counter += 1

            # early_stopping(self.model, acc_ind, None,
            #                save_name=f"[ES]{self.exp_name}_{self.model.name}_ep{ep + 1}.pth")
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            if (ep + 1) % 5 == 0:
                self.ood_test_online()
                save_order = input(f"Save model weights at Epoch {ep + 1}?\n").lower()
                if save_order == "y":
                    path = os.path.join(model_path, f"{self.exp_name}_{self.model.name}_ep{ep + 1}.pth")
                    torch.save(self.model.state_dict(), path)

                stop_order = input(f"Stop training at Epoch {ep + 1}?\n").lower()
                if stop_order == "y":
                    return

    def ood_test_online(self):
        self.model.eval()

        train_loader = iter(DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=True))
        valid_loader = iter(DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), shuffle=True))
        ood_loader = iter(DataLoader(self.ood_dataset, batch_size=len(self.ood_dataset), shuffle=True))

        # get logits and features
        num_mc = 1  # >=20
        logits_ind_train, feats_ind_train, labels_ind_train = self.get_logits_feats(train_loader, 'train', num_mc)
        logits_ind_val, feats_ind_val, labels_ind_val = self.get_logits_feats(valid_loader, 'valid', num_mc)
        logits_ood, feats_ood, labels_ood = self.get_logits_feats(ood_loader, 'ood', num_mc)

        # for alpha in [0, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        #     self.vim_alpha = alpha
        #     s_train, s_ind, s_ood = self.ood_fn([logits_ind_train[0], logits_ind_val[0], logits_ood[0]],
        #                                         [feats_ind_train[0], feats_ind_val[0], feats_ood[0]],
        #                                         [labels_ind_train[0], labels_ind_val[0], labels_ood[0]],
        #                                         score_fn_name='vim', ensemble=False)

        s_train, s_ind, s_ood = self.ood_fn([logits_ind_train[0], logits_ind_val[0], logits_ood[0]],
                                            [feats_ind_train[0], feats_ind_val[0], feats_ood[0]],
                                            [labels_ind_train[0], labels_ind_val[0], labels_ood[0]],
                                            score_fn_name=self.score_fn_name, ensemble=False)
        if self.ood_detect_single_cls:
            y_ood = labels_ood[0]  # (T, N)
            score_ood = s_ood
            score_ind = s_ind
            path = rf"...\exp4\{self.score_fn_name}_score"
            save_curve_and_plot(y_ood, score_ood, score_ind, unc_mode=False, score_save_path=path)
            exit()

        # u_train, u_ind, u_ood = self.unc_fn(logits_ind_train[0], logits_ind_val[0], logits_ood[0])

        # cimpute AccU
        probs = torch.softmax(logits_ind_val[0], -1)  # (B, C)
        print(probs.shape, s_ind.shape, labels_ind_val.shape)
        compute_roc_AccU(-s_ind, probs, labels_ind_val[0])
        compute_roc_AccU_v3(-s_ind, probs, labels_ind_val[0], -s_train, s_train, s_ind, False)
        # or: compute_roc_AccU_v3(-s_ind, probs, labels_ind_val[0], s_train, s_ind, ood_is_unc=False)
        pred_ood_dist(-s_ind, -s_ood, -s_ind, -s_ood,
                      title=f"pred_ood_dist@{self.score_fn_name}_exp{self.exp_num}")

        y_pred = logits_ind_val.mean(0).argmax(-1)
        wrong_pred = (labels_ind_val[0] != y_pred).detach().cpu().numpy()
        right_pred = (labels_ind_val[0] == y_pred).detach().cpu().numpy()
        self.visualize_right_wrong_ind(s_train, s_ind[right_pred], s_ind[wrong_pred],
                                       s_ood, unc_mode=False, title=f'exp4_wrong_right_ood_uncs_{self.score_fn_name}')

        self.model.train()

    def get_logits_feats(self, loader, mode='valid', num_mc=20):
        bx, by = loader.__next__()
        bx, by = bx.cuda(), by.cuda().long()
        # if mode == 'train':
        #     print(f"before aug: {bx.shape}")
        #     bx, by = self.sample_augment(bx, by)
        #     print(f"after aug: {bx.shape}")

        logits = []
        feats = []
        bys = []
        with torch.no_grad():
            for _ in range(num_mc):
                logits.append(self.model(bx))
                feats.append(self.model.features)
                bys.append(by)
        logits = torch.stack(logits, 0)  # (T, N, C)
        feats = torch.stack(feats, 0)
        bys = torch.stack(bys, 0)

        if mode != 'ood':
            acc_valid = (logits.mean(0).argmax(-1) == by).float().mean().item()
            print(f'Acc. under {len(bx)} {mode} samples: {acc_valid:.2%}')

        return logits, feats, bys

    def ood_test_offline(self, model_path):
        self.construct_dataset(mode='ood')
        state = torch.load(model_path)
        self.model.load_state_dict(state)
        self.ood_test_online()

    def visualize_right_wrong_ind(self, train_score, ind_right, ind_wrong, ood, unc_mode=True, title='unc'):
        def convert_to_numpy(x):
            if not isinstance(x, np.ndarray):
                x = x.detach().cpu().numpy()
            return x

        train_score = convert_to_numpy(train_score)
        ind_right = convert_to_numpy(ind_right)
        ind_wrong = convert_to_numpy(ind_wrong)
        ood = convert_to_numpy(ood)

        if unc_mode:
            wrong_right_thre = np.percentile(train_score, 95)
            train_95 = train_score[train_score < wrong_right_thre]
        else:
            wrong_right_thre = np.percentile(train_score, 5)
            train_95 = train_score[train_score > wrong_right_thre]

        wrong_right_train_ind_dist(ind_wrong, ind_right, train_score, train_95, ood, title)


if __name__ == "__main__":
    model_dir = r"..."
    ood_fns = ["energy", "vim", "nusa",
               "mahalanobis", "residual", "kl_matching", "msp"]  # "energy_react"-->energy
    # Exp4:
    # trainer = BaseCNNTrainer(exp_num=4, figure_save=False, ood_detect_single_cls=False)
    # trainer.score_fn_name = ood_fns[0]
    # trainer.train(model_dir)
    # load_pt = os.path.join(model_dir, r"Exp4_ConvNet1_ep100.pth")
    # trainer.ood_test_offline(load_pt)

    # # # Exp=2, 3:
    trainer = BaseCNNTrainer(exp_num=3, figure_save=False, ood_detect_single_cls=False)
    trainer.score_fn_name = ood_fns[1]
    # 1) train:
    # trainer.train(model_dir)

    # # 2) test:
    # load_pt = os.path.join(model_dir, r"Exp2_ConvNet1_ep15.pth")
    load_pt = os.path.join(model_dir, r"Exp3_ConvNet1_ep20.pth")
    trainer.ood_test_offline(load_pt)
