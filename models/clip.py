import torch
import hydra
import pytorch_lightning as pl
import numpy as np
from models.biomedclip import BiomedCLIP
from models.pubmedclip import PubmedCLIP

class CLIPModule(pl.LightningModule):
    def __init__(self, config, type):
        super(CLIPModule, self).__init__()

        self.config = config
        self.is_generative = False

        if type == 'pubmed':
            self.clip = PubmedCLIP(config)
        elif type == 'biomed':
            self.clip = BiomedCLIP(config)

    def forward(self, images, texts):
        return self.clip(images, texts)
    
    def closed_ended(self, images, texts, labels, options):
        logits = self.forward(images, texts)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        logits = logits.detach().cpu().numpy()
        sorted_indices = sorted_indices.detach().cpu().numpy()
        # if len(images) == 1:
        #     sorted_indices = np.expand_dims(sorted_indices, 0)

        pred = [options[i[0]] for i in sorted_indices]
        return pred
    
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    model = BiomedCLIP(config)
    preds = model(torch.rand((10, 1, 192, 192)), ['Male', 'Female'])
    x = 3

if __name__ == "__main__": 
    test()