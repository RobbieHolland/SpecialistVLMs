import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

class Perceiver(nn.Module):
    def __init__(self, config, feature_dim):
        super().__init__()
        
        # Initialize the learnable query tokens as described
        self.queries = nn.Parameter(torch.empty(config.model.miniformer.learnable_queries, feature_dim // 2))
        nn.init.xavier_normal_(self.queries)
        self.queries = nn.Parameter(self.queries.unsqueeze(0))  # Adding a batch dimension for broadcasting
        
        self.projection = nn.Linear(feature_dim, feature_dim // 2)
        self.key_projection = nn.Linear(feature_dim // 2, feature_dim // 2)

        if config.model.miniformer.linear_upsample:
            output_dim = config.model.language_model.hidden_dim
        else:
            output_dim = feature_dim
        self.value_projection = nn.Linear(feature_dim // 2, output_dim)

    def forward(self, x):
        x = self.projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        # Expand queries to match the batch size of x
        queries_expanded = self.queries.expand(x.size(0), -1, -1)
        
        # Compute attention scores using batch matrix multiplication
        attn_weights = torch.matmul(queries_expanded, keys.transpose(-1, -2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Compute weighted sum of values based on attention weights
        output = torch.matmul(attn_weights, values)
        
        return output

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    model = Perceiver(config, 2048)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'MiniGPT-4 has {trainable_params} trainable parameters')
    
    x = torch.randn([5, 144, 2048])
    p_y = model(x)
    print(p_y.shape)
    x = 3

if __name__ == "__main__":
    test()
