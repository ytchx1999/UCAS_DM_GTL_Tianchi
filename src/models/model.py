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
        :param pretain_dir: location of pretrained model resnet-18 (../../data/resnet18-5c106cde.pth)
        :param num_classes: label dims
        '''
        super(ResNet, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.num_classes = num_classes
        self.resnet_18 = torchvision.models.resnet18()

        # download or use cached model
        if not os.path.exists(self.pretrain_dir):
            # download
            # https://download.pytorch.org/models/resnet18-5c106cde.pth
            url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
            wget.download(url, "../../data/resnet18-5c106cde.pth")

        # load state dict (params) of the model
        state_dict_load = torch.load(self.pretrain_dir, map_location='cpu')
        self.resnet_18.load_state_dict(state_dict_load)

        # modify the last FC layer
        num_features = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        x = self.resnet_18(x)
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
            x = self.bns[i](x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return x


class EyeNet(nn.Module):
    def __init__(self, pretrain_dir, num_classes=16, feat_dim=5, hidden_dim=16, output_dim=12):
        super(EyeNet, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.resnet_pre = ResNet(self.pretrain_dir, num_classes=num_classes)
        self.resnet_after = ResNet(self.pretrain_dir, num_classes=num_classes)

        self.mlp = MLP(num_classes * 2 + feat_dim, hidden_dim, output_dim, num_layers=3)

    def forward(self, pre_img, after_img, feat):
        pre_img = self.resnet_pre(pre_img)
        after_img = self.resnet_after(after_img)
        x = torch.cat([pre_img, after_img, feat], dim=1)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    # resnet_18 = ResNet('../../data/resnet18-5c106cde.pth', num_classes=16)
    # summary(resnet_18, input_size=(3, 224, 224))
    model = EyeNet(
        '../../data/resnet18-5c106cde.pth',
        num_classes=16,
        feat_dim=5,
        hidden_dim=16,
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
