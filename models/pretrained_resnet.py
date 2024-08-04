import torch
from timm.models.layers import trunc_normal_
import hydra
import sys
import os
import pytorch_lightning as pl
from torchvision import transforms
from models.util import ResizeImage, ExpandChannels
from models.ssl_config import ssl_backbone

class PretrainedResNet(pl.LightningModule):
    def __init__(self, config):
        super(PretrainedResNet, self).__init__()

        checkpoint = os.path.join(config.pretrained_model_dir, config.model.vision_encoder.checkpoint)

        sd = torch.load(checkpoint)['state_dict']
        sd = {k.split('encoder.')[-1]: v for (k, v) in sd.items() if 'encoder' in k and 'tail' not in k}
        if 'blocks' not in config.model.vision_encoder.keys():
            blocks = None
        else:
            blocks = config.model.vision_encoder.blocks
        encoder = ssl_backbone(channels_in=1, blocks=blocks)
        encoder.load_state_dict(sd, strict=True)

        self.model = encoder.eval()
        self.feature_tokens_model = torch.nn.Sequential(*list(self.model.children())[:-1])

    def embed_image(self, xs):
        return self.model.latent_code(xs)

    def feature_tokens(self, images):
        tokens = self.feature_tokens_model(images)
        return tokens.view(tokens.shape[0], -1, tokens.shape[1])
    
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    model = PretrainedResNet(config)
    print(model)
    y = model.feature_tokens(torch.randn([3, 1, 192, 192]))
    print(y)
    
    print('Success')


if __name__ == "__main__":
    test()