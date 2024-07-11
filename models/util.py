import torch

class ResizeImage(torch.nn.Module):
    def __init__(self, size=(224, 224)):
        super().__init__()
        self.upsample = torch.nn.Upsample(size, mode='bilinear')

    def forward(self, x):
        return self.upsample(x)

class ExpandChannels(torch.nn.Module):
    def forward(self, x):
        # Repeat the single channel three times along the channel dimension
        return x.repeat(1, 3, 1, 1)
    
def set_llama3_pad_token(tokenizer):
    tokenizer.pad_token = '<|end_of_text|>'
    tokenizer.pad_token_id = 128001
    return tokenizer