import torch
from timm.models.layers import trunc_normal_
import hydra
import sys
import os
from timm.models.layers import trunc_normal_
import pytorch_lightning as pl
from torchvision import transforms
from models.util import ResizeImage, ExpandChannels
import torch.nn.functional as F
import numpy as np

# Efficient implementation equivalent to the following:
import math
def custom_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    attn_bias = attn_bias.to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

class RETFound(pl.LightningModule):
    def __init__(self, config):
        super(RETFound, self).__init__()
        sys.path.append(config.retfound_dir)
        import models_vit
        # from util.pos_embed import interpolate_pos_embed
        # call the model
        self.config = config
        self.model = models_vit.__dict__['vit_large_patch16'](
            num_classes=2,
            drop_path_rate=0.2,
            global_pool=True,
        )
        self.config = config

        # load RETFound weights
        checkpoint = torch.load(os.path.join(config.pretrained_model_dir, 'RETFound_oct_weights.pth'))#, map_location='cuda:0')
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        self.model = self.model.to(torch.device('cuda:0'))

        self.preprocess = torch.nn.Sequential(ResizeImage(), ExpandChannels(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        if config.model.vision_encoder.p_tokens:
            self.learnable_p_tokens = torch.nn.Parameter(torch.empty(config.model.vision_encoder.p_tokens, config.model.vision_encoder.feature_dim))
            torch.nn.init.xavier_normal_(self.learnable_p_tokens)
            self.learnable_p_tokens = torch.nn.Parameter(self.learnable_p_tokens.unsqueeze(0))

    def reshape_and_concatenate(self, activations):
        N, num_patches, feature_dim = activations.shape

        # Extract class token and image patches
        class_token = activations[:, :1, :]
        image_patches = activations[:, 1:, :].reshape(N, 14, 14, feature_dim)

        # Reshape the patches by combining 2x2 neighborhoods
        image_patches = image_patches.unfold(1, 2, 2).unfold(2, 2, 2)
        image_patches = image_patches.permute(0, 1, 3, 2, 4, 5).reshape(N, -1, feature_dim * 4)

        # Repeat the class token across the feature dimension to match new patches
        class_token_expanded = class_token.repeat(1, 1, 4)
        final_output = torch.cat([class_token_expanded, image_patches], dim=1)

        return final_output

    def forward_features(self, x, tokens=True):
        B = x.shape[0]
        x = self.model.patch_embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        # Pass through all but the last Transformer block
        x = self.model.blocks(x)

        # if self.config.model.vision_encoder.p_tokens:
        #     p_tokens = self.learnable_p_tokens.expand(B, -1, -1)
        #     x = torch.cat((x, p_tokens), dim=1)  # Concatenate existing features with p_tokens

        # Final Transformer block processing
        # block = self.model.blocks[-1]
        # attention_block = block.attn
        # attention = block.attn(block.norm1(x))

        # if return_attentions:
        #     y = x.clone()
        #     y = block.norm1(y)
        #     B, N, C = y.shape
        #     qkv = attention_block.qkv(y).reshape(B, N, 3, attention_block.num_heads, attention_block.head_dim).permute(2, 0, 3, 1, 4)
        #     q, k, v = qkv.unbind(0)
        #     q, k = attention_block.q_norm(q), attention_block.k_norm(k)
        #     if attention_block.fused_attn:
        #         y = F.scaled_dot_product_attention(
        #             q, k, v,
        #             dropout_p=self.attn_drop.p if self.training else 0.,
        #         )
        #     else:
        #         raise Exception('Not fused attention!')

        #     y_new, attention_weight = custom_scaled_dot_product_attention(
        #             q, k, v,
        #             dropout_p=self.attn_drop.p if self.training else 0.,
        #         )

        # x = self.model.blocks[-1](x)

        # if self.config.model.vision_encoder.p_tokens:
        #     p_tokens_output = x[:, -p_tokens.shape[1]:, :]
        # else:
        #     p_tokens_output = x

        # if tokens:
        #     outcome = x[:, 1:, :]
        #     return self.model.fc_norm(outcome)

        # if self.model.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.model.fc_norm(x)
        # else:
        #     x = self.model.norm(x)
        #     outcome = x[:, 0]

        # if return_attentions:
        #     return p_tokens_output, attention_weight
        if self.config.model.vision_encoder.concatenate_tokens:
            x = self.reshape_and_concatenate(x)

        return x

    def embed_image(self, xs):
        return self.model.forward_features(self.preprocess(xs))
    
    def feature_tokens(self, xs):
        tokens = self.forward_features(self.preprocess(xs), tokens=True)
        # if self.config.model.vision_encoder.concatenate_tokens:
        #     kernel = self.config.model.vision_encoder.concatenate_tokens
        #     tokens = tokens.view(2, 14//kernel, kernel, 14//kernel, kernel, 1024).permute(0, 1, 3, 2, 4, 5).reshape(2, (14//kernel) ** 2, -1)
        # if self.config.model.vision_encoder.average_tokens:
        #     kernel = self.config.model.vision_encoder.average_tokens
        #     tokens = tokens.view(2, 14//kernel, kernel, 14//kernel, kernel, 1024).permute(0, 1, 3, 2, 4, 5).mean((3, 4)).reshape(2, (14//kernel) ** 2, -1)

        return tokens

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    device = torch.device('cuda:0')
    model = RETFound(config).to(device)
    print(model)

    img = torch.randn([3, 1, 384, 384]).to(device)
    out = model.feature_tokens(img)

    print('Success')

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def attention(config):
    sys.path.append(config['octlatent_dir'])
    from dataset.retinal_text_dataset import RetinalTextDataset

    device = torch.device('cuda:0')
    dataset = RetinalTextDataset(config, set_='train')

    import matplotlib.pyplot as plt
    import numpy as np
    from evaluation.figure_util import save_fig_path_creation

    import random
    rand_images = [random.randint(0, 2000) for _ in range(16)]
    print(rand_images)
    images = torch.stack([dataset.__getitem__(i)[0] for i in rand_images])

    device = torch.device('cuda:0')
    model = RETFound(config).to(device).eval()

    images = images.to(model.device)
    out = model.feature_tokens(images)

    x = 3

if __name__ == "__main__":
    # test()
    attention()

    # tokens = torch.arange(2 * 8 * 8).view(2, -1)
    # kernel = 2
    # tokens = tokens.view(2, 8//kernel, kernel, 8//kernel, kernel).permute(0, 1, 3, 2, 4).reshape(2, (8//kernel) ** 2, -1)