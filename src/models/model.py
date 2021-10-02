import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
import os
import wget
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, pretrain_dir, num_classes=16):
        '''
        :param pretain_dir: location of pretrained model resnet-50 (../../data/resnet50-19c8e357.pth)
        :param num_classes: label dims
        '''
        super(ResNet, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet152()

        # download or use cached model
        if not os.path.exists(self.pretrain_dir):
            # download
            # ResNet-18: https://download.pytorch.org/models/resnet18-5c106cde.pth
            # ResNet-50: https://download.pytorch.org/models/resnet50-19c8e357.pth
            # ResNet-152: https://download.pytorch.org/models/resnet152-b121ed2d.pth
            url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            wget.download(url, "../../data/resnet50-19c8e357.pth")

        # load state dict (params) of the model
        state_dict_load = torch.load(self.pretrain_dir, map_location='cpu')
        self.resnet.load_state_dict(state_dict_load)

        # modify the last FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for i in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lins.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.relu(self.lins[i](x))
            x = self.bns[i](x)  # batch norm
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return x


class EyeNet(nn.Module):
    def __init__(self, pretrain_dir, num_classes=64, feat_dim=5, hidden_dim=64, output_dim=12):
        super(EyeNet, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.resnet_pre = ResNet(self.pretrain_dir, num_classes=num_classes)
        self.resnet_after = ResNet(self.pretrain_dir, num_classes=num_classes)

        # freeze grad in resnet-50
        for param in self.resnet_pre.parameters():
            param.requires_grad = False
        for param in self.resnet_after.parameters():
            param.requires_grad = False

        self.mlp = MLP(num_classes * 2 + hidden_dim, hidden_dim, output_dim, num_layers=4)
        # linear for feat
        self.fc_feat = nn.Linear(feat_dim, hidden_dim)

    def forward(self, pre_img, after_img, feat):
        pre_img = self.resnet_pre(pre_img)
        after_img = self.resnet_after(after_img)
        feat = self.fc_feat(feat)
        x = torch.cat([feat, pre_img, after_img], dim=1)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    # resnet_18 = ResNet('../../data/resnet18-5c106cde.pth', num_classes=16)
    # summary(resnet_18, input_size=(3, 224, 224))
    model = EyeNet(
        '../../data/resnet50-19c8e357.pth',
        num_classes=64,
        feat_dim=5,
        hidden_dim=64,
        output_dim=12
    )
    print(model)

    pre_img = torch.randn(16, 3, 224, 244)
    after_img = torch.randn(16, 3, 224, 244)
    feat = torch.randn(16, 5)

    out = model(pre_img, after_img, feat)

    print(out.shape)
    # torch.Size([16, 12])
    print(out)
