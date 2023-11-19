import os
import numpy as np
import torch
import visdom
from torch.utils.data import DataLoader

from models.nets import BWCNN
from base_trainer import BaseCNNTrainer
from myutils.vim_utils import compute_roc_AccU, compute_roc_AccU_v3, get_threshold
from myutils.plot_utils import ood_score_dist, pred_ood_dist
from myutils.curve_fit import save_curve_and_plot


class BCNNTrainer(BaseCNNTrainer):
    def __init__(self, exp_num, figure_save=False, ood_detect_single_cls=False):
        super().__init__(exp_num, figure_save, ood_detect_single_cls)
        self.model = BWCNN(in_channels=6 if self.exp_num == 4 else 3, num_classes=4).cuda()

    def train(self, model_path):
        self.construct_dataset(mode='ood')
        batch_size = 64
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        vis = visdom.Visdom(env='base_cnn')
        Epochs = 1000
        counter = 0
        num_MC_valid = 10
        num_MC_train = 5
        # Episodes = self.train_dataset.__len__() // batch_size
        print(f"Let's train {self.model.name}!")

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        optimizer = torch.optim.Adadelta(self.model.parameters())  # better
        beta = 1e-6

        for ep in range(Epochs):
            valid_loader = iter(DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True))
            val_loss = []
            for epi, (bx, by) in enumerate(train_loader):
                bx, by = bx.cuda(), by.cuda().long()

                kl = 0.0
                ls = 0.0
                logits = []
                self.model.eval()
                for j in range(num_MC_train):
                    logit = self.model(bx)
                    logits.append(logit)
                    kl += self.model.kl_losses / num_MC_train
                    ls += criterion(logit, by) / num_MC_train
                self.model.train()
                loss = ls + kl * beta
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                logits = torch.stack(logits, 0).mean(0)
                tr_acc = (logits.argmax(1) == by).float().mean().item()
                loss_train = loss.detach().cpu().item()

                if (epi + 1) % 2 == 0:
                    bx_ind, by_ind = valid_loader.__next__()
                    bx_ind, by_ind = bx_ind.cuda(), by_ind.cuda().long()
                    logits_ind = []
                    Acc = []

                    with torch.no_grad():
                        for i in range(num_MC_valid):
                            lg = self.model(bx_ind)
                            logits_ind.append(lg)  # (N2, nc)
                            acc = (lg.argmax(1) == by_ind).float().mean().item()
                            Acc.append(acc)
                    logits_ind = torch.stack(logits_ind, 0)  # (T, N2, C)
                    preds_ind = logits_ind.mean(0)  # (N2, C)
                    acc_ind = (preds_ind.argmax(1) == by_ind).float().mean().item()
                    loss_ind = criterion(preds_ind, by_ind).detach().cpu().item()
                    val_loss.append(loss_ind)

                    vis.line(Y=[[kl.detach().cpu().data * beta, loss_train, loss_ind]], X=[counter],
                             update=None if counter == 0 else 'append', win='Loss_CNN',
                             opts=dict(legend=['kl', 'train', 'val'], title='Loss_CNN'))
                    vis.line(Y=[[tr_acc, acc_ind]], X=[counter],
                             update=None if counter == 0 else 'append', win='Acc_CNN',
                             opts=dict(legend=['train', 'val'], title='Acc_CNN'))
                    counter += 1

            # lr_sched.step(np.mean(val_loss))

            if (ep + 1) % 20 == 0:
                self.model.eval()
                self.ood_test_online()
                self.model.train()
                save_order = input("Save model weights? ").lower()
                if save_order == "y":
                    path = os.path.join(model_path, f"{self.exp_name}_{self.model.name}_ep{ep + 1}.pth")
                    torch.save(self.model.state_dict(), path)

                stop_order = input("Stop training? ").lower()
                if stop_order == "y":
                    return

    def ood_test_online(self):
        self.model.eval()
        train_loader = iter(DataLoader(self.train_dataset, batch_size=400, shuffle=True))
        valid_loader = iter(DataLoader(self.valid_dataset, batch_size=800, shuffle=True))
        ood_loader = iter(DataLoader(self.ood_dataset, batch_size=800, shuffle=True))

        num_mc = 50  # >=20, w.r.t "Unc in DL" thesis
        logits_train, feats_train, labels_train, W_train = self.get_logits_feats(train_loader, 'train', num_mc)
        logits_ind, feats_ind, labels_ind, W_ind = self.get_logits_feats(valid_loader, 'valid', num_mc)
        logits_ood, feats_ood, labels_ood, W_ood = self.get_logits_feats(ood_loader, 'ood', num_mc)

        # ********** obtain result uncertainty ***********
        # u_train, u_ind, u_ood = self.unc_fn(logits_train, logits_ind, logits_ood)
        (all_train, all_ind, all_ood), (ale_train, ale_ind, ale_ood), (epi_train, epi_ind, epi_ood) = \
            self.unc_fn(logits_train, logits_ind, logits_ood, ret_all_unc=True)
        # evaluate:
        # auc_ood, fpr_ood = self.get_auc_fpr(epi_ind, epi_ood, score_prob=False)
        # print(f'[uncs_epi] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
        ood_score_dist(epi_train, epi_ind, epi_ood, title=f"{self.exp_name}_repar_promo_w_uncs_epi",
                       fig_save=self.SaveFigure)
        if self.ood_detect_single_cls:
            y_ood = labels_ood[0]  # (T, N)
            score_ood = epi_ood
            score_ind = epi_ind
            if self.exp_num == 3:
                path = rf"E:\fy_works\save_model\bcnn\discussion\ind_ood_scores\exp3\promoW_unc"
            elif self.exp_num == 4:
                path = rf"E:\fy_works\save_model\bcnn\discussion\ind_ood_scores\exp4\promoW_unc"
            save_curve_and_plot(y_ood, score_ood, score_ind, unc_mode=True, score_save_path=path)

        # ********** obtain weight score ***********
        ood_s1, ood_s2 = self.get_ood_scores([logits_train, logits_ind, logits_ood],
                                             [feats_train, feats_ind, feats_ood],
                                             [labels_train, labels_ind, labels_ood],
                                             [W_train, W_ind, W_ood])
        # ood_s1: (score_train, score_ind, score_ood)
        # evaluate:
        auc_ood, fpr_ood = self.get_auc_fpr(ood_s1[1], ood_s1[2], score_prob=True)
        print(f'[nusa-s1] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')
        ood_score_dist(ood_s1[0], ood_s1[1], ood_s1[2], title=f"{self.exp_name}_repar_promo_w_nusa_s1",
                       fig_save=self.SaveFigure)
        score_ood = ood_s1[2]
        score_ind = ood_s1[1]

        # auc_ood, fpr_ood = self.get_auc_fpr(ood_s2[1], ood_s2[2])
        # print(f'[nusa-s2] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')
        # ood_score_dist(ood_s2[0], ood_s2[1], ood_s2[2], title=f"{self.exp_name}_promo_w_nusa_s2",
        #                fig_save=self.SaveFigure)

        # auc_ood, fpr_ood = self.get_auc_fpr(ood_s1[1]-ood_s2[1], ood_s1[2]-ood_s2[2])
        # print(f'[nusa-s1-s2] auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}\n')
        # ood_score_dist(ood_s1[0]-ood_s2[0], ood_s1[1]-ood_s2[1], ood_s1[2]-ood_s2[2],
        #                title=f"{self.exp_name}_promo_w_nusa_s12",
        #                fig_save=self.SaveFigure)

        # get AccU:
        # ours:
        probs = torch.softmax(logits_ind, -1).mean(0)  # (B, C)
        labels_ind = labels_ind[0]
        print(probs.shape, all_ind.shape, labels_ind.shape)
        # print("unc-all:")
        # compute_roc_AccU(all_ind, probs, labels_ind)
        # compute_roc_AccU_v3(all_ind, probs, labels_ind, uncer_train=epi_train,
        #                     ind_ood_score=ood_s1[1], ood_is_unc=False)
        print("unc-epi:")
        compute_roc_AccU(epi_ind, probs, labels_ind)
        compute_roc_AccU_v3(epi_ind, probs, labels_ind, uncer_train=epi_train, score_train=ood_s1[0], ind_score=ood_s1[1],
                            score_is_unc=False)
        print("unc-all:")
        compute_roc_AccU_v3(all_ind, probs, labels_ind, uncer_train=all_train, score_train=ood_s1[0], ind_score=ood_s1[1],
                            score_is_unc=False)
        # pred_ood_dist(epi_ind, epi_ood, -score_ind, -score_ood,
        #               title=f"pred_ood_dist@promoW_exp{self.exp_num}")

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
        weights = []
        with torch.no_grad():
            for _ in range(num_mc):
                logits.append(self.model(bx))
                feats.append(self.model.features)
                bys.append(by)
                weights.append([self.model.classifier.weight, self.model.classifier.bias])  # (T, 2, 2)
        logits = torch.stack(logits, 0)  # (T, N, C)
        feats = torch.stack(feats, 0)
        bys = torch.stack(bys, 0)

        if mode != 'ood':
            acc_valid = (logits.mean(0).argmax(-1) == by).float().mean().item()
            print(f'Acc. under {len(bx)} {mode} samples: {acc_valid:.2%}')

        return logits, feats, bys, weights

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

        score_train, score_ind, score_ood = np.mean(score_train, 0), np.mean(score_ind, 0), np.mean(score_ood, 0)
        score_train2, score_ind2, score_ood2 = np.mean(score_train2, 0), np.mean(score_ind2, 0), np.mean(score_ood2, 0)

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


if __name__ == "__main__":
    model_dir = ...  # your directory

    trainer = BCNNTrainer(exp_num=2, figure_save=False, ood_detect_single_cls=False)
    trainer.train(model_dir)

    # load_pt = os.path.join(model_dir, r"Exp4_BWCNN_ep60.pth")  # only exp_num=4
    # load_pt = os.path.join(model_dir, r"Exp3_BWCNN_ep40.pth")  # exp_num=2 and 3
    # trainer.ood_test_offline(load_pt)
