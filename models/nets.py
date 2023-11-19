
import torch
import torch.nn as nn
from promo.models.layers import BBB_Linear, BBB_Conv1d
from promo.models.layers import BBB_LRT_Linear, BBB_LRT_Conv1d
from promo.models.layers import ModuleWrapper


class BWCNN(ModuleWrapper):
    name = "BWCNN"

    def __init__(self,
                 in_channels,
                 num_classes,
                 bn_affine: bool = False,
                 layer_type='lrt',
                 activation_type='softplus',
                 track_state=False
                 ) -> None:
        super().__init__()

        self.priors = {
            'prior_mu': 0,
            'prior_sigma': 0.1,
            'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
            'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
        }

        self.num_classes = num_classes
        self.padding = 1
        self.num_channels = (16, 32, 64)
        self.kernel_size = (65, 17, 3)
        self.pool_size = (8, 8, 8)
        self.bn_affine = bn_affine
        self.track_state=track_state

        if layer_type == 'lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv1d = BBB_LRT_Conv1d
        elif layer_type == 'bbb':
            BBBLinear = BBB_Linear
            BBBConv1d = BBB_Conv1d
        else:
            raise ValueError("Undefined layer_type")

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.FeatLayers = nn.Sequential(
            BBBConv1d(
                in_channels=in_channels,
                out_channels=self.num_channels[0], kernel_size=self.kernel_size[0],
                stride=1, padding=(self.kernel_size[0] - 1) // 2, priors=self.priors
            ),
            nn.BatchNorm1d(num_features=self.num_channels[0],
                           momentum=1,
                           track_running_stats=self.track_state,
                           affine=self.bn_affine
                           ),
            # nn.Dropout(self.drop_rate),
            self.act(),
            nn.MaxPool1d(self.pool_size[0]),

            BBBConv1d(
                in_channels=self.num_channels[0],
                out_channels=self.num_channels[1], kernel_size=self.kernel_size[1],
                stride=1, padding=(self.kernel_size[1] - 1) // 2, priors=self.priors
            ),
            nn.BatchNorm1d(num_features=self.num_channels[1],
                           momentum=1,
                           track_running_stats=self.track_state,
                           affine=self.bn_affine
                           ),
            # nn.Dropout(self.drop_rate),
            self.act(),
            nn.MaxPool1d(self.pool_size[1]),

            BBBConv1d(
                in_channels=self.num_channels[1],
                out_channels=self.num_channels[2], kernel_size=self.kernel_size[2],
                stride=1, padding=(self.kernel_size[2] - 1) // 2, priors=self.priors
            ),
            nn.BatchNorm1d(num_features=self.num_channels[2],
                           momentum=1,
                           track_running_stats=self.track_state,
                           affine=self.bn_affine
                           ),
            # nn.Dropout(self.drop_rate),
            self.act(),
            nn.MaxPool1d(self.pool_size[2]),

            nn.Flatten(),
            BBBLinear(4*self.num_channels[-1], 128, priors=self.priors),
            self.act(),
        )
        self.classifier = BBBLinear(128, self.num_classes, priors=self.priors)


class BWCNN_LA(ModuleWrapper):
    name = "BWCNN_LA"

    def __init__(self,
                 in_channels,
                 num_classes,
                 bn_affine: bool = False,
                 layer_type='lrt',
                 activation_type='softplus',
                 track_state=False,
                 ) -> None:
        super().__init__()

        self.priors = {
            'prior_mu': 0,
            'prior_sigma': 0.1,
            'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
            'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
        }

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.padding = 1
        self.num_channels = (16, 32, 64)
        self.kernel_size = (65, 17, 3)
        self.pool_size = (4, 4, 4)
        self.bn_affine = bn_affine
        self.track_state = track_state

        if layer_type == 'lrt':
            self.BBBLinear = BBB_LRT_Linear
            self.BBBConv1d = BBB_LRT_Conv1d
        elif layer_type == 'bbb':
            self.BBBLinear = BBB_Linear
            self.BBBConv1d = BBB_Conv1d
        else:
            raise ValueError("Undefined layer_type")

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.backbone = self.construct_feats()
        self.flatten = nn.Flatten()
        self.construct_heads()
        # self.classifier = BBBLinear(128, self.num_classes, priors=self.priors)
        self.classifiers = nn.ModuleList([self.BBBLinear(256, self.num_classes, priors=self.priors),
                                          self.BBBLinear(512, self.num_classes, priors=self.priors),
                                          self.BBBLinear(1024, self.num_classes, priors=self.priors),
                                          ])

    def control_BN(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.BatchNorm1d):
                layer.training = training

    def construct_feats(self):
        net = nn.Sequential()
        temp = nn.Sequential(
            self.BBBConv1d(
                in_channels=self.in_channels,
                out_channels=self.num_channels[0], kernel_size=self.kernel_size[0],
                stride=1, padding=(self.kernel_size[0] - 1) // 2,
                bias=not self.bn_affine, priors=self.priors
            ),
            nn.BatchNorm1d(
                num_features=self.num_channels[0],
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(self.pool_size[0])
        )
        net.add_module(name='layer0', module=temp)

        for i in range(1, len(self.num_channels)):
            temp = nn.Sequential(
                self.BBBConv1d(
                    in_channels=self.num_channels[i - 1],
                    out_channels=self.num_channels[i],
                    kernel_size=self.kernel_size[i],
                    stride=1, padding=(self.kernel_size[i] - 1) // 2,
                    bias=not self.bn_affine, priors=self.priors
                ),
                nn.BatchNorm1d(
                    num_features=self.num_channels[i],
                    momentum=1,
                    track_running_stats=self.track_state,
                    affine=self.bn_affine
                ),
                self.act(),
                nn.MaxPool1d(self.pool_size[i])
            )
            net.add_module(name='layer{0:d}'.format(i), module=temp)

        # net.add_module(name='Flatten', module=nn.Flatten())
        # net.add_module(name='fc1', module=nn.Sequential(
        #     nn.LazyLinear(out_features=self.hidden_fc),
        #     nn.Dropout(0.3),
        #     nn.ReLU(True),
        # ))

        return net

    def construct_heads(self):
        self.branch1_1 = nn.Sequential(
            self.BBBConv1d(64, 32, 1, 1, 0, priors=self.priors),
            nn.BatchNorm1d(
                num_features=32,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.up1 = nn.ConvTranspose1d(32, 32, 3, 2, 1, 1)
        self.branch1_2 = nn.Sequential(
            self.BBBConv1d(32, 32, 1, 1, 0, priors=self.priors),
            nn.BatchNorm1d(
                num_features=32,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.branch3_1 = nn.Sequential(
            self.BBBConv1d(64+32, 64, 3, 1, 1, priors=self.priors),
            nn.BatchNorm1d(
                num_features=64,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.up3 = nn.ConvTranspose1d(64, 64, 3, 2, 1, 1)
        self.branch3_2 = nn.Sequential(
            self.BBBConv1d(64+32, 64, 3, 1, 1, priors=self.priors),
            nn.BatchNorm1d(
                num_features=64,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.branch5_1 = nn.Sequential(
            self.BBBConv1d(64+64, 128, 5, 1, 2, priors=self.priors),
            nn.BatchNorm1d(
                num_features=128,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.branch5_2 = nn.Sequential(
            self.BBBConv1d(128+64, 128, 5, 1, 2, priors=self.priors),
            nn.BatchNorm1d(
                num_features=128,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = self.backbone(x)  # (N, 64, 2048->32)
        feats1_1 = self.branch1_1(x)  # (N, 32, 16), (N, C, L)
        feats1_1_up = self.up1(feats1_1)  # (N, 32, 32)
        feats3_1 = self.branch3_1(torch.cat([x, feats1_1_up], 1))  # /2, c-64
        feats3_1_up = self.up3(feats3_1)  # c-64
        feats5_1 = self.branch5_1(torch.cat([x, feats3_1_up], 1))  # /2, c-128

        feats1_2 = self.branch1_2(feats1_1)  # (N, 32, 8)
        feats3_2 = self.branch3_2(torch.cat([feats3_1, feats1_1], 1))  # (N, 64, 8)
        feats5_2 = self.branch5_2(torch.cat([feats5_1, feats3_1], 1))  # (N, 128, 8)
        # print(feats1_2.shape)
        # print(feats3_2.shape)
        # print(feats5_2.shape)
        # print("=========")
        # exit()

        feats1_2, feats3_2, feats5_2 = self.flatten(feats1_2), self.flatten(feats3_2), self.flatten(feats5_2)
        features = [feats1_2, feats3_2, feats5_2]
        logits = []
        for i in range(3):
            logits.append(self.classifiers[i](features[i]))

        # print(feats1_2.shape, feats3_2.shape, feats5_2.shape)
        self.features = features

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        self.kl_losses = kl

        return logits
        # return logits[0]  # 1st layer for CAM, 2023-11-19


class BWCNN_LA_Group(ModuleWrapper):
    name = "BWCNN_LA_Group"

    def __init__(self,
                 in_channels,
                 num_classes,
                 bn_affine: bool = False,
                 layer_type='lrt',
                 activation_type='softplus',
                 track_state=False,
                 ) -> None:
        super().__init__()

        self.priors = {
            'prior_mu': 0,
            'prior_sigma': 0.1,
            'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
            'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
        }

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.padding = 1

        # self.num_channels = (in_channels*4, in_channels*8, in_channels*12)
        self.num_channels = (16, 32, 64)

        self.kernel_size = (65, 17, 3)
        self.pool_size = (4, 4, 4)
        self.bn_affine = bn_affine
        self.groups = [2, 2] if in_channels % 2 == 0 else [1, 1]
        self.track_state = track_state

        if layer_type == 'lrt':
            self.BBBLinear = BBB_LRT_Linear
            self.BBBConv1d = BBB_LRT_Conv1d
        elif layer_type == 'bbb':
            self.BBBLinear = BBB_Linear
            self.BBBConv1d = BBB_Conv1d
        else:
            raise ValueError("Undefined layer_type")

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.backbone = self.construct_feats()
        self.flatten = nn.Flatten()
        self.construct_heads()
        # self.classifier = BBBLinear(128, self.num_classes, priors=self.priors)
        self.classifiers = nn.ModuleList([self.BBBLinear(256, self.num_classes, priors=self.priors),
                                          self.BBBLinear(512, self.num_classes, priors=self.priors),
                                          self.BBBLinear(1024, self.num_classes, priors=self.priors),
                                          ])

    def control_BN(self, training=False):
        for layer in self.children():
            if isinstance(layer, nn.BatchNorm1d):
                layer.training = training

    def construct_feats(self):
        net = nn.Sequential()
        temp = nn.Sequential(
            self.BBBConv1d(
                in_channels=self.in_channels,
                out_channels=self.num_channels[0], kernel_size=self.kernel_size[0],
                stride=1, padding=(self.kernel_size[0] - 1) // 2,
                bias=not self.bn_affine, priors=self.priors, groups=self.groups[0]
            ),
            nn.BatchNorm1d(
                num_features=self.num_channels[0],
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(self.pool_size[0])
        )
        net.add_module(name='layer0', module=temp)

        for i in range(1, len(self.num_channels)):
            temp = nn.Sequential(
                self.BBBConv1d(
                    in_channels=self.num_channels[i - 1],
                    out_channels=self.num_channels[i],
                    kernel_size=self.kernel_size[i],
                    stride=1, padding=(self.kernel_size[i] - 1) // 2,
                    bias=not self.bn_affine, priors=self.priors, groups=self.groups[0]
                ),
                nn.BatchNorm1d(
                    num_features=self.num_channels[i],
                    momentum=1,
                    track_running_stats=self.track_state,
                    affine=self.bn_affine
                ),
                self.act(),
                nn.MaxPool1d(self.pool_size[i])
            )
            net.add_module(name='layer{0:d}'.format(i), module=temp)

        # net.add_module(name='Flatten', module=nn.Flatten())
        # net.add_module(name='fc1', module=nn.Sequential(
        #     nn.LazyLinear(out_features=self.hidden_fc),
        #     nn.Dropout(0.3),
        #     nn.ReLU(True),
        # ))

        return net

    def construct_heads(self):
        inch = self.num_channels[-1]
        outc = 32
        self.branch1_1 = nn.Sequential(
            self.BBBConv1d(inch, outc, 1, 1, 0, priors=self.priors, groups=self.groups[1]),
            nn.BatchNorm1d(
                num_features=outc,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.up1 = nn.ConvTranspose1d(outc, outc, 3, 2, 1, 1)
        self.branch1_2 = nn.Sequential(
            self.BBBConv1d(outc, outc, 1, 1, 0, priors=self.priors, groups=self.groups[1]),
            nn.BatchNorm1d(
                num_features=outc,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.branch3_1 = nn.Sequential(
            self.BBBConv1d(inch+outc, outc*2, 3, 1, 1, priors=self.priors, groups=self.groups[1]),
            nn.BatchNorm1d(
                num_features=outc*2,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.up3 = nn.ConvTranspose1d(outc*2, outc*2, 3, 2, 1, 1)
        self.branch3_2 = nn.Sequential(
            self.BBBConv1d(outc*2+outc, outc*2, 3, 1, 1, priors=self.priors, groups=self.groups[1]),
            nn.BatchNorm1d(
                num_features=outc*2,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.branch5_1 = nn.Sequential(
            self.BBBConv1d(inch+outc*2, outc*4, 5, 1, 2, priors=self.priors, groups=self.groups[1]),
            nn.BatchNorm1d(
                num_features=outc*4,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.branch5_2 = nn.Sequential(
            self.BBBConv1d(outc*6, outc*4, 5, 1, 2, priors=self.priors, groups=self.groups[1]),
            nn.BatchNorm1d(
                num_features=outc*4,
                momentum=1,
                track_running_stats=self.track_state,
                affine=self.bn_affine
            ),
            self.act(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = self.backbone(x)  # (N, 64, 2048->32)
        feats1_1 = self.branch1_1(x)  # (N, 32, 16), (N, C, L)
        feats1_1_up = self.up1(feats1_1)  # (N, 32, 32)
        feats3_1 = self.branch3_1(torch.cat([x, feats1_1_up], 1))  # /2, c-64
        feats3_1_up = self.up3(feats3_1)  # c-64
        feats5_1 = self.branch5_1(torch.cat([x, feats3_1_up], 1))  # /2, c-128

        feats1_2 = self.branch1_2(feats1_1)  # (N, 32, 8)
        feats3_2 = self.branch3_2(torch.cat([feats3_1, feats1_1], 1))  # (N, 64, 8)
        feats5_2 = self.branch5_2(torch.cat([feats5_1, feats3_1], 1))  # (N, 128, 8)
        # print(feats1_2.shape)
        # print(feats3_2.shape)
        # print(feats5_2.shape)
        # print("=========")
        # exit()

        feats1_2, feats3_2, feats5_2 = self.flatten(feats1_2), self.flatten(feats3_2), self.flatten(feats5_2)
        features = [feats1_2, feats3_2, feats5_2]
        logits = []
        for i in range(3):
            logits.append(self.classifiers[i](features[i]))

        # print(feats1_2.shape, feats3_2.shape, feats5_2.shape)
        self.features = features

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        self.kl_losses = kl

        return logits


if __name__ == "__main__":

    # net = BWCNN(in_channels=6, num_classes=4).cuda()
    # inp = torch.zeros(32, 6, 2048).cuda()
    # out = net(inp)
    # print(out.shape)

    zeros = torch.zeros([32, 6, 2048])
    cnn = BWCNN_LA(6, 4)
    out = cnn(zeros)
    print(cnn.kl_losses.data)

