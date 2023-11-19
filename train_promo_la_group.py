import os
import numpy as np
import torch
import visdom
import time
from torch.utils.data import DataLoader

from models.nets import BWCNN_LA, BWCNN_LA_Group
from base_trainer import BaseCNNTrainer
from myutils.plot_utils import ood_score_dist, pred_ood_dist, wrong_right_train_ind_dist
from myutils.curve_fit import save_curve_and_plot
from data.data_base_bogie import BogieDatasetTorch
from myutils.vim_utils import compute_roc_AccU, compute_roc_AccU_v3, get_threshold


class BCNN_LA_Trainer(BaseCNNTrainer):
    def __init__(self, exp_num, figure_save=False, ood_detect_single_cls=False):
        super().__init__(exp_num, figure_save, ood_detect_single_cls)
        self.model = BWCNN_LA_Group(in_channels=6 if self.exp_num == 4 else 3, num_classes=4).cuda()
        self.model_name = "ProMoLAGroup"
        self.best_fpr = 100.
        self.best_epoch = 0

    def construct_bogie(self, mode="train"):
        # https://github.com/cathysiyu/Mechanical-datasets
        # https://www.mitssolutions.asia/drivetrain-diagnostics-simulator-dds
        # todo: assume that we only simulated the crack faults, and got no info for pitting fault.
        train_classes = ["bear_norm", "bear_outer_crack_h", "bear_cage_crack", "bear_roller_crack_h"]
        ind_classes = ["bear_norm", "bear_outer_crack_h", "bear_cage_crack", "bear_roller_crack_m"]
        ood_classes_gear = ["gear_pitt", "gear_lack"]
        ood_classes_bear = ["bear_outer_pitt_m", "bear_roller_pitt_l"]
        # test samples have covariate shift, and have Unknown known classes and Unknown unknown classes
        # ood_classes_bear = ["bear_outer_pitt_m", "bear_roller_pitt_l", "bear_roller_pitt_m", "bear_roller_pitt_h"]
        self.exp_name = "Exp4"

        if mode == "train" or mode == 'ood':
            self.train_dataset = BogieDatasetTorch("train", train_classes,
                                                   file_name="BogieBear_rpm2000_load20",
                                                   train_num_per_cls=40, test_num_per_cls=0)  # 50*4 = 200

        self.valid_dataset = BogieDatasetTorch("valid", ind_classes,
                                               file_name="BogieBear_rpm2000_load20",
                                               train_num_per_cls=0, test_num_per_cls=50)
        if mode == "ood":
            ood_bear = BogieDatasetTorch("valid", ood_classes_bear,
                                                 file_name="BogieBear_rpm2000_load0",
                                                 train_num_per_cls=0, test_num_per_cls=50)
            ood_gear = BogieDatasetTorch("valid", ood_classes_gear,
                                                 file_name="BogieGear_rpm2000_load15",  # 15
                                                 train_num_per_cls=0, test_num_per_cls=50)
            ood_gear.x = np.concatenate([ood_gear.x, ood_bear.x], 0)
            ood_gear.y = np.concatenate([ood_gear.y, ood_bear.y + len(ood_classes_gear)], 0)
            print(f"ood samples: {ood_gear.x.shape, ood_gear.y.shape}")
            self.ood_dataset = ood_gear

    def predict(self, bx, by, criterion, num_MC):
        # logits = self.model(bx)
        # loss, pred = 0., 0.
        # for logit in logits:
        #     loss += criterion(logit, by) / len(logits)
        #     pred += logit / len(logits)

        logits = 0.
        ls, kl = 0., 0.
        for j in range(num_MC):
            logit = self.model(bx)
            kl += self.model.kl_losses / num_MC
            logit_ = 0.
            for lgt in logit:
                ls += criterion(lgt, by) / num_MC / len(logit)
                logit_ += lgt / len(logit)

            logits += logit_ / num_MC  # (N, C)

        return logits, ls, kl

    def train(self, model_path):
        self.construct_dataset(mode='ood')
        batch_size = 64
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        vis = visdom.Visdom(env='base_cnn')
        Epochs = 1000
        counter = 0
        num_MC_valid = 10
        num_MC_train = 3
        # Episodes = self.train_dataset.__len__() // batch_size
        print(f"Let's train {self.model.name}!")

        # criterion = metrics.ELBO(batch_size).to(self.model.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        if self.exp_num == 4:
            optimizer = torch.optim.Adadelta(self.model.parameters())  # better
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # beta = 5e-7
        beta = 1e-6

        for ep in range(Epochs):
            valid_loader = iter(DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True))
            for epi, (bx, by) in enumerate(train_loader):
                bx, by = bx.cuda(), by.cuda().long()
                logits, ls, kl = self.predict(bx, by, criterion, num_MC_train)
                loss = ls + kl * beta
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tr_acc = (logits.argmax(1) == by).float().mean().item()
                loss_train = loss.detach().cpu().item()

                if (epi + 1) % 2 == 0:
                    bx_ind, by_ind = valid_loader.__next__()
                    bx_ind, by_ind = bx_ind.cuda(), by_ind.cuda().long()
                    logits1, logits2, logits3 = [], [], []

                    with torch.no_grad():
                        self.model.eval()
                        for i in range(num_MC_valid):
                            lg = self.model(bx_ind)  # list(lg1, lg2, lg3)
                            logits1.append(lg[0])  # (N2, nc)
                            logits2.append(lg[1])  # (N2, nc)
                            logits3.append(lg[2])  # (N2, nc)
                    self.model.train()

                    logits1, logits2, logits3 = torch.stack(logits1, 0), \
                                                torch.stack(logits2, 0), \
                                                torch.stack(logits3, 0)  # (T, N2, C)
                    preds1, preds2, preds3 = logits1.mean(0), logits2.mean(0), logits3.mean(0)
                    preds = (logits1 + logits2 + logits3).mean(0)
                    acc_1 = (preds1.argmax(1) == by_ind).float().mean().item()
                    acc_2 = (preds2.argmax(1) == by_ind).float().mean().item()
                    acc_3 = (preds3.argmax(1) == by_ind).float().mean().item()
                    acc_ind = (preds.argmax(1) == by_ind).float().mean().item()
                    loss_ind = criterion(preds, by_ind).detach().cpu().item()

                    vis.line(Y=[[kl.detach().cpu().data * beta, loss_train, loss_ind]], X=[counter],
                             update=None if counter == 0 else 'append', win='Loss_BCNN_LA',
                             opts=dict(legend=['kl', 'train', 'val'], title='Loss_BCNN_LA'))
                    vis.line(Y=[[tr_acc, acc_1, acc_2, acc_3, acc_ind]], X=[counter],
                             update=None if counter == 0 else 'append', win='Acc_BCNN_LA',
                             opts=dict(legend=['train', 'val1', 'val2', 'val3', 'val'], title='Acc_BCNN_LA'))
                    counter += 1

            if (ep + 1) >= 10 and (ep + 1) % 2 == 0:
                self.model.eval()
                self.ood_test_online(ep)
                self.model.train()
                save_order = input(f"Save model weights at epoch {ep + 1}? ").lower()
                if save_order == "y":
                    path = os.path.join(model_path, f"{self.exp_name}_{self.model.name}_ep{ep + 1}.pth")
                    if os.path.exists(path):
                        path = path[:-4] + r"(1).pth"
                    torch.save(self.model.state_dict(), path)

                stop_order = input(f"Stop training at epoch {ep + 1}? ").lower()
                if stop_order == "y":
                    return

    def ood_test_online(self, epoch=0):
        self.model.eval()
        train_loader = iter(DataLoader(self.train_dataset, batch_size=400, shuffle=True))
        valid_loader = iter(DataLoader(self.valid_dataset, batch_size=800, shuffle=True))
        ood_loader = iter(DataLoader(self.ood_dataset, batch_size=800, shuffle=True))

        num_mc = 30  # >=20, w.r.t "Unc in DL" thesis
        logits_train, feats_train, labels_train, W_train = self.get_logits_feats(train_loader, 'train', num_mc)
        logits_ind, feats_ind, labels_ind, W_ind = self.get_logits_feats(valid_loader, 'valid', num_mc)
        logits_ood, feats_ood, labels_ood, W_ood = self.get_logits_feats(ood_loader, 'ood', num_mc)

        logits = logits_ind[0] + logits_ind[1] + logits_ind[2]  # (T, N2, C)
        y_pred = logits.mean(0).argmax(-1)
        wrong_pred = (labels_ind[0] != y_pred).detach().cpu().numpy()
        right_pred = (labels_ind[0] == y_pred).detach().cpu().numpy()

        # 1) uncertainty:
        uncs_train, uncs_ind, uncs_ood = 0., 0., 0.
        ale_train, ale_ind, ale_ood = 0., 0., 0.
        epi_train, epi_ind, epi_ood = 0., 0., 0.
        for i in range(3):
            s1, s2, s3 = self.unc_fn(logits_train[i], logits_ind[i], logits_ood[i],
                                     eval_all_unc=False, ret_all_unc=True)
            # evaluate:
            auc_ood, fpr_ood = self.get_auc_fpr(s1[1], s1[2], score_prob=False)
            print(f'[{self.model_name}-layer-{i + 1}-all] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
            auc_ood, fpr_ood = self.get_auc_fpr(s2[1], s2[2], score_prob=False)
            print(f'[{self.model_name}-layer-{i + 1}-ale] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
            auc_ood, fpr_ood = self.get_auc_fpr(s3[1], s3[2], score_prob=False)
            print(f'[{self.model_name}-layer-{i + 1}-epi] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')

            uncs_train += s1[0] / 3
            uncs_ind += s1[1] / 3
            uncs_ood += s1[2] / 3

            ale_train += s2[0] / 3
            ale_ind += s2[1] / 3
            ale_ood += s2[2] / 3

            epi_train += s3[0] / 3
            epi_ind += s3[1] / 3
            epi_ood += s3[2] / 3

        auc_ood, fpr_ood = self.get_auc_fpr(uncs_ind, uncs_ood, score_prob=False)
        print(f'[uncs_all] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        # ood_score_dist(uncs_train, uncs_ind, uncs_ood, title=f"{self.exp_name}_repar_{self.model_name}_uncs_all",
        #                fig_save=self.SaveFigure)
        auc_ood, fpr_ood = self.get_auc_fpr(ale_ind, ale_ood, score_prob=False)
        print(f'[uncs_ale] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        auc_ood, fpr_ood = self.get_auc_fpr(epi_ind, epi_ood, score_prob=False)
        print(f'[uncs_epi] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')
        ood_score_dist(epi_train, epi_ind, epi_ood, title=f"{self.exp_name}_repar_{self.model_name}_uncs_epi",
                       fig_save=self.SaveFigure)

        if fpr_ood < self.best_fpr:
            self.best_fpr = fpr_ood
            self.best_epoch = epoch+1
            print(f"***** Best FPR95: {self.best_fpr} at epoch {self.best_epoch} *****\n")

        # 2) ood scores:
        scores_train, scores_ind, scores_ood = 0., 0., 0.
        S1, S2, S3 = [], [], []
        for i in range(3):
            # s1, s2, s3 = self.ood_fn([logits_train[i], logits_ind[i], logits_ood[i]],
            #                          [feats_train[i], feats_ind[i], feats_ood[i]],
            #                          [labels_train, labels_ind, labels_ood], classifier_id=i)
            print(f"score: Layer-{i+1}")
            (s1, s2, s3), _ = self.get_ood_scores([logits_train[i], logits_ind[i], logits_ood[i]],
                                                  [feats_train[i], feats_ind[i], feats_ood[i]],
                                                  [labels_train[i], labels_ind[i], labels_ood[i]],
                                                  [W_train[i], W_ind[i], W_ood[i]])
            scores_train += s1 / 3
            scores_ind += s2 / 3
            scores_ood += s3 / 3
            S1.append(s1), S2.append(s2), S3.append(s3)
        auc_ood, fpr_ood = self.get_auc_fpr(scores_ind, scores_ood, score_prob=True)
        print(f'[B-NuSA_all] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')
        ood_score_dist(scores_train, scores_ind, scores_ood, title=f"{self.exp_name}_repar_{self.model_name}_scores_all",
                       fig_save=self.SaveFigure)

        # get AccU:
        # ours:
        logits_ind = logits_ind[0] + logits_ind[1] + logits_ind[2]  # (T, B, C)
        probs = torch.softmax(logits_ind, -1).mean(0)  # (B, C)
        labels_ind = labels_ind[0]
        print(probs.shape, uncs_ind.shape, labels_ind.shape)
        print("unc-all:")
        compute_roc_AccU(uncs_ind, probs, labels_ind)
        compute_roc_AccU_v3(uncs_ind, probs, labels_ind, uncer_train=epi_train, score_train=scores_train,
                            ind_score=scores_ind, score_is_unc=False)
        print("unc-epi:")
        compute_roc_AccU(epi_ind, probs, labels_ind)
        compute_roc_AccU_v3(epi_ind, probs, labels_ind, uncer_train=epi_train, score_train=scores_train,
                            ind_score=scores_ind, score_is_unc=False)
        compute_roc_AccU_v3(epi_ind, probs, labels_ind, uncer_train=epi_train, score_train=uncs_train,
                            ind_score=uncs_ind, score_is_unc=True)
        # print("unc-all:")
        # # compute_roc_AccU_v3(uncs_ind, probs, labels_ind, uncer_train=uncs_train, score_train=scores_train,
        # #                     ind_score=scores_ind, score_is_unc=False)
        # compute_roc_AccU_v3(uncs_ind, probs, labels_ind, uncer_train=uncs_train, score_train=uncs_train,
        #                     ind_score=uncs_ind, score_is_unc=False)

        # self.visualize_distribution(epi_train, epi_ind[right_pred], epi_ind[wrong_pred],
        #                             epi_ood, unc_mode=True, title='exp4_wrong_right_ood_uncs')
        # self.visualize_distribution(scores_train, scores_ind[right_pred], scores_ind[wrong_pred],
        #                             scores_ood, unc_mode=False, title='exp4_wrong_right_ood_scores')

        # pred_ood_dist(epi_ind, epi_ood, -scores_ind, -scores_ood,
        #               title=f"pred_ood_dist@{self.model_name}_exp{self.exp_num}")

    def get_logits_feats(self, loader, mode='valid', num_mc=20):
        bx, by = loader.__next__()
        bx, by = bx.cuda(), by.cuda().long()

        logits1, logits2, logits3 = [], [], []
        feats1, feats2, feats3 = [], [], []
        w1, w2, w3 = [], [], []
        bys = []
        with torch.no_grad():
            for _ in range(num_mc):
                lg = self.model(bx)
                fe = self.model.features
                wt = [[self.model.classifiers[0].weight, self.model.classifiers[0].bias],
                      [self.model.classifiers[1].weight, self.model.classifiers[1].bias],
                      [self.model.classifiers[2].weight, self.model.classifiers[2].bias]]
                bys.append(by)

                logits1.append(lg[0])
                logits2.append(lg[1])
                logits3.append(lg[2])

                feats1.append(fe[0])
                feats2.append(fe[1])
                feats3.append(fe[2])

                w1.append(wt[0])
                w2.append(wt[1])
                w3.append(wt[2])

        logits1, logits2, logits3 = torch.stack(logits1, 0), \
                                    torch.stack(logits2, 0), torch.stack(logits3, 0)  # (T, N2, C)
        feats1, feats2, feats3 = torch.stack(feats1, 0), \
                                 torch.stack(feats2, 0), torch.stack(feats3, 0)  # (T, N2, C)
        # w1, w2, w3 = torch.stack(w1, 0), torch.stack(w2, 0), torch.stack(w3, 0)
        if mode != "ood":
            logits = (logits1 + logits2 + logits3) / 3.  # (T, N2, C)
            acc_valid = (logits.mean(0).argmax(-1) == by).float().mean().item()
            print(f'Acc. under {len(bx)} {mode} samples: {acc_valid:.2%}')

        return (logits1, logits2, logits3), (feats1, feats2, feats3), bys, (w1, w2, w3)

    def get_ood_scores(self, logits, feats, labels, weights):
        feats_train, feats_ind, feats_ood = feats
        # logits_train, logits_ind, logits_ood = logits
        # labels_train, labels_ind, labels_ood = labels
        w_train, w_ind, w_ood = weights

        w_avg = 0.
        for i in range(len(w_train)):
            w_avg += (w_train[i][0][0] + w_ind[i][0][0] + w_ood[i][0][0]) / 3 / len(w_train)

        score_train, score_ind, score_ood = [], [], []
        score_train2, score_ind2, score_ood2 = [], [], []
        t0 = time.time()
        for i in range(len(feats_train)):
            s_train, s_ind, s_ood = self.bnn_nusa([w_train[i][0][0], w_ind[i][0][0], w_ood[i][0][0]],
                                                  [feats_train[i], feats_ind[i], feats_ood[i]])
            # s_train, s_ind, s_ood = self.bnn_nusa([w_avg, w_avg, w_avg],
            #                                       [feats_train[i], feats_ind[i], feats_ood[i]])
            s_train2, s_ind2, s_ood2 = self.bnn_nusa([w_train[i][0][1], w_ind[i][0][1], w_ood[i][0][1]],
                                                     [feats_train[i] ** 2, feats_ind[i] ** 2, feats_ood[i] ** 2])
            # weight [[W_mu, W_sigma**2], [[bias_mu], [bias_rho]]]

            score_train.append(s_train)
            score_ind.append(s_ind)
            score_ood.append(s_ood)

            score_train2.append(s_train2)
            score_ind2.append(s_ind2)
            score_ood2.append(s_ood2)
        tt = time.time() - t0
        print(f"time for three sco types: {tt} /s")
        score_train, score_ind, score_ood = np.mean(score_train, 0), np.mean(score_ind, 0), np.mean(score_ood, 0)
        score_train2, score_ind2, score_ood2 = np.mean(score_train2, 0), np.mean(score_ind2, 0), np.mean(score_ood2, 0)

        auc_ood, fpr_ood = self.get_auc_fpr(score_ind, score_ood, score_prob=True)
        print(f'[B-NuSA] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        # ood_score_dist(score_train, score_ind, score_ood, title=f"{self.exp_name}_{self.model_name}_nusa{classifier_id}",
        #                fig_save=self.SaveFigure)

        return (score_train, score_ind, score_ood), (score_train2, score_ind2, score_ood2)

    @staticmethod
    def bnn_nusa(weights, features):
        w_train, w_ind, w_ood = weights
        f_train, f_ind, f_ood = features

        def get_nusa_score(W, feats):
            proj = torch.mm((torch.mm(W.T, torch.inverse(torch.mm(W, W.T)))), W)
            proj_x = torch.mm(feats, proj)  # (None, m1)x(m1, m1)
            score = torch.norm(proj_x, p=2, dim=1) / torch.norm(feats, p=2, dim=1)
            return score.cpu().detach().numpy()

        score_train = get_nusa_score(w_train, f_train)
        score_ind = get_nusa_score(w_ind, f_ind)
        score_ood = get_nusa_score(w_ood, f_ood)

        return score_train, score_ind, score_ood

    def visualize_distribution(self, train_score, ind_right, ind_wrong, ood, unc_mode=True, title='unc'):
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

        # wrong_right_train_ind_dist(ind_wrong, ind_right, train_score, train_95, ood, title, fig_save=True)
        wrong_right_train_ind_dist(ind_wrong, ind_right, train_score, train_95, ood, title, fig_path=None)


if __name__ == "__main__":
    model_dir = ...  # your directory
    trainer = BCNN_LA_Trainer(exp_num=3, figure_save=False, ood_detect_single_cls=False)

    # 1) train:
    trainer.train(model_dir)

    # 2) test:
    # ********** in BWCNN_LA_Group: self.num_channels = (16, 32, 64)
    load_pt = ...  # your path
    trainer.ood_test_offline(load_pt)  # for visualize_distribution

