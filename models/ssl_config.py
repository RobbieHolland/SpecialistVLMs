from models.resnet import ResNet, Bottleneck

def ssl_backbone(num_outs=1, regression=False, channels_in=1, blocks=None):
    if blocks is None:
        blocks = [3, 4, 6, 3]

    encoder = ResNet(Bottleneck, blocks, num_classes=num_outs, regress=regression, channels_in=channels_in)

    return encoder
