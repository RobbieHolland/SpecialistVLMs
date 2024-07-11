import torch
from timm.models.layers import trunc_normal_
import hydra
import sys
import os
import pytorch_lightning as pl
from torchvision import transforms
from models.util import ResizeImage, ExpandChannels
from evaluation.self_supervised.ssl_config import ssl_backbone

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
    model.feature_tokens(torch.randn([3, 1, 192, 192]))
    print(model)

    print('Success')


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def transformer(config):
    import torch
    import torch.nn as nn

    class TransformerLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
            super(TransformerLayer, self).__init__()
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                                dim_feedforward=dim_feedforward, dropout=dropout)
            
            self.learnable_p_tokens = torch.nn.Parameter(torch.empty(50, dim_feedforward))
            torch.nn.init.xavier_normal_(self.learnable_p_tokens)
            self.learnable_p_tokens = torch.nn.Parameter(self.learnable_p_tokens.unsqueeze(0))

        def forward(self, x, mask=None):

            learnable_tokens_batch = self.learnable_p_tokens.repeat(x.shape[0], 1, 1)
            x = torch.cat((learnable_tokens_batch, x), dim=1)  # Concatenate learnable tokens

            output = self.transformer_layer(x)
            output = output[:,:learnable_tokens_batch.shape[1]]
            return output

    # Example usage:
    # Define the input tensor
    input_tensor = torch.randn(10, 36, 2048)  # (sequence length, batch size, embedding dimension)

    # Initialize the transformer layer
    transformer_layer = TransformerLayer(d_model=2048, nhead=8)
    print('Trainable parameters', sum(p.numel() for p in transformer_layer.parameters() if p.requires_grad))

    # Pass the input tensor through the transformer layer
    output_tensor = transformer_layer(input_tensor)

    # Check the shape of the output tensor
    print(output_tensor.shape)  # Output: torch.Size([10, 32, 2048])

    x = 3

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def global_project(config):
    import torch
    import torch.nn as nn

    class TokenModelWithLearnableTokens(nn.Module):
        def __init__(self, feature_dim, num_tokens_in, num_learnable_tokens, num_heads):
            super(TokenModelWithLearnableTokens, self).__init__()
            self.num_learnable_tokens = num_learnable_tokens

            self.learnable_p_tokens = torch.nn.Parameter(torch.empty(num_learnable_tokens, feature_dim))
            torch.nn.init.xavier_normal_(self.learnable_p_tokens)
            self.learnable_p_tokens = torch.nn.Parameter(self.learnable_p_tokens.unsqueeze(0))

            # Multihead attention layer
            self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        def forward(self, x):
            # x should have shape (batch_size, num_tokens_in, feature_dim)
            batch_size = x.size(0)
            
            # Repeat the learnable tokens for the batch and combine with input
            learnable_tokens_batch = self.learnable_p_tokens.repeat(batch_size, 1, 1)
            x = torch.cat((learnable_tokens_batch, x), dim=1)  # Concatenate learnable tokens
            
            # Permute x to fit (seq_len, batch_size, feature_dim) expected by nn.MultiheadAttention
            x = x.permute(1, 0, 2)
            
            # Self-attention
            attn_output, _ = self.self_attn(x, x, x)
            
            # Extract the learnable tokens after attention
            # Assuming they are still at the beginning
            learnable_output = attn_output[:self.num_learnable_tokens, :, :]
            
            return learnable_output.permute(1, 0, 2)

    # Parameters
    batch_size = 10
    feature_dim = 1024
    num_tokens_in = 36
    num_learnable_tokens = 50
    num_heads = 8

    # Initialize model
    model = TokenModelWithLearnableTokens(feature_dim, num_tokens_in, num_learnable_tokens, num_heads)
    print('Trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Example input
    input_tokens = torch.randn(batch_size, num_tokens_in, feature_dim)

    # Forward pass
    output_tokens = model(input_tokens)
    print("Output shape:", output_tokens.shape)  # Expected shape (batch_size, num_learnable_tokens, feature_dim)

    x = 3

if __name__ == "__main__":
    # test()
    # global_project()
    transformer()