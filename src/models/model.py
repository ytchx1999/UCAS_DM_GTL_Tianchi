import torch
import torchvision
import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn
import os
import wget


class ResNet(nn.Module):
    def __init__(self, pretrain_dir, num_classes=1000):
        '''
        :param pretain_dir: location of pretrained model resnet-18 (../../data/resnet18-5c106cde.pth)
        :param num_classes: label dims
        '''
        super(ResNet, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.num_classes = num_classes
        self.resnet_50 = torchvision.models.resnet50()

        # download or use cached model
        if not os.path.exists(self.pretrain_dir):
            # download
            # https://download.pytorch.org/models/resnet18-5c106cde.pth
            url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            wget.download(url, "../../data/resnet50-19c8e357.pth")

        # load state dict (params) of the model
        state_dict_load = torch.load(self.pretrain_dir, map_location='cpu')
        self.resnet_50.load_state_dict(state_dict_load)

        # modify the last FC layer
        self.resnet = nn.Sequential(*list(self.resnet_50.children())[:-5])
        # num_features = self.resnet_50.fc.in_features
        # self.resnet_50.fc = nn.Linear(num_features, 16)

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


class Mixup(nn.Module):
    '''
    将图像混合后进行一个两层卷积混合，最后再作为两个分支输出
    '''

    def __init__(self, channels, classes1, classes2, base):
        super(Mixup, self).__init__()
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(channels, base*4, 3, 2, 1),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(base*4, base*2, 3, 2, 1),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 2, 1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 2, 1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.linear1 = nn.Linear(base, classes1)
        self.linear2 = nn.Linear(base, classes2)

        self.bn1 = nn.BatchNorm1d(classes1)
        self.bn2 = nn.BatchNorm1d(classes2)

        self.ConvBlock.apply(self.init_weights)
        self.branch1.apply(self.init_weights)
        self.branch2.apply(self.init_weights)

    def forward(self, x):
        x = self.ConvBlock(x)
        x1 = self.branch1(x.clone())
        x2 = self.branch2(x)

        x1 = self.bn1(F.relu(self.linear1(torch.flatten(x1, 1))))
        x2 = self.bn2(F.relu(self.linear2(torch.flatten(x2, 1))))

        out = torch.cat([x1, x2], dim=1)

        return out

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class EYENet(nn.Module):
    def __init__(self, pretrained, channels, classes1, classes2, base, feature_dims):
        super(EYENet, self).__init__()
        self.pre_resnet = ResNet(pretrained)
        self.aft_resnet = ResNet(pretrained)
        self.mix = Mixup(channels, classes1, classes2, base)

        for param in self.pre_resnet.parameters():
            param.requires_grad = False
        for param in self.aft_resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(classes1 + classes2 + feature_dims, 12)
        # self.linear.apply(self.init_weights)

        self.mlp = MLP(classes1 + classes2 + feature_dims, 64, 12, num_layers=4)

    def forward(self, x1, x2, x3):
        x1 = self.pre_resnet(x1)
        x2 = self.aft_resnet(x2)
        x = torch.cat([x1, x2], dim=1)

        x = self.mix(x)
        x = torch.cat([x, x3], dim=1)

        # x = self.linear(x)
        x = self.mlp(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    # resnet_18 = ResNet('../../data/resnet18-5c106cde.pth', num_classes=2)
    eye = EYENet('../../data/resnet50-19c8e357.pth', 512, 10, 10, 64, 5)
    print(eye)
    # print(resnet_18)
    summary(eye, input_size=[(3, 32, 32), (3, 32, 32), (1, 5)], device='cpu')
    # print(os.path.exists('./resnet18-5c106cde.pth'))
