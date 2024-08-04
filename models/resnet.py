# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64, norm_layer=None, expansion=4):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.expansion = expansion

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 width_per_group=64, in_planes=64, block_expansion=4, norm_layer=nn.BatchNorm2d, regress=False, channels_in=1):
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.block_expansion = block_expansion
        self.layer_dims = layers
        
        self.inplanes = in_planes
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channels_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if len(layers) == 5:
            self.layer5 = self._make_layer(block, 512, layers[4], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.regress = regress

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block_expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self.block_expansion, stride),
                self.norm_layer(planes * self.block_expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.base_width, self.norm_layer, expansion=self.block_expansion))
        self.inplanes = planes * self.block_expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride= 1, base_width=self.base_width, norm_layer=self.norm_layer, expansion=self.block_expansion))

        return nn.Sequential(*layers)

    def latent_code(self, x):
        # Downsample to latent space
        #print('conv1', x.shape)
        x = self.conv1(x)
        #print('conv1', x.shape)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('maxpool', x.shape)

        x = self.layer1(x)
        #print('layer1', x.shape)
        x = self.layer2(x)
        #print('layer2', x.shape)
        x = self.layer3(x)
        #print('layer3', x.shape)
        x = self.layer4(x)
        #print('layer4', x.shape)

        if len(self.layer_dims) == 5:
            x = self.layer5(x)

        x = self.avgpool(x)
        #print('avgpool', x.shape)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        z = self.latent_code(x)

        return z

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50 =  ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    feature_tokens_model = torch.nn.Sequential(*list(resnet50.children())[:-1])
    x = torch.zeros([3, 1, 384, 384]).to(device)
    c, z = resnet50(x)
    print(z.shape)
