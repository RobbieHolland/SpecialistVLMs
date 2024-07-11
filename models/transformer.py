import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
import math

class Miniformer(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        config = config.model.miniformer

        transformer_config = BertConfig(
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.nhead,
            hidden_size=hidden_size,
            intermediate_size=config.dim_feedforward,
            pooler_output=False,
        )

        self.transformer = BertModel(transformer_config)
        del self.transformer.embeddings.word_embeddings
        del self.transformer.pooler
        self.transformer.pooler = None

        self.query_tokens = nn.Parameter(torch.randn(config.learnable_queries, hidden_size))
        nn.init.kaiming_uniform_(self.query_tokens, a=math.sqrt(5))

    def forward(self, embeddings):
        # Prepend query tokens
        query_tokens = self.query_tokens.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        combined_input = torch.cat([query_tokens, embeddings], dim=1)

        outputs = self.transformer(inputs_embeds=combined_input)
        return outputs[0][:,:self.query_tokens.shape[0]]

# Define configuration with a single layer

import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    model = Miniformer(config, 2048)
    x = torch.randn([5, 36, 2048])
    p_y = model(x)
    print(p_y.shape)

if __name__ == "__main__":
    test()
