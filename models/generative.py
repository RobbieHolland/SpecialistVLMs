import torch
import hydra
import pytorch_lightning as pl
import torch
import os

import sys
from models.mini_gpt4 import MiniGPT4
from models.med_flamingo import MedFlamingo
from run.vision_language_pretraining import MiniGPT4Module
import numpy as np

class GenerativeModule(pl.LightningModule):
    def __init__(self, config, type=None, vlm=None):
        super(GenerativeModule, self).__init__()

        self.config = config
        self.is_generative = True

        if vlm is not None:
            self.vlm = vlm
        else:
            if type == 'medflamingo':
                self.vlm = MedFlamingo(config)
            elif type == 'mini_gpt4':
                self.vlm = MiniGPT4(config, 'biomed')

    def forward(self, images, texts, output_only=False):
        return self.vlm.query(images, texts, output_only=output_only, max_new_tokens=30)
    
    def closed_ended(self, task, images, texts, labels, options):
        # responses = self.forward(images, texts, output_only=True)
        responses = self.few_shot(task, images, texts, labels, output_only=True)
        print('First text:', texts[0], 'Response:', responses[0])
        # answer = response.split('Answer: ')[1]
        # matching_responses = [next((option for option in options if option in response), None) for response in responses]
        matching_responses = [[option for option in options if option in response] for response in responses]
        # next((label for label in labels if text.startswith(label)), None)
        return matching_responses
    
    def few_shot(self, task, images, texts, labels, output_only=True, k=3):
        few_shot_ix = np.array([np.concatenate((np.random.permutation(np.delete(np.arange(len(images)), i))[:k], [i])) for i in range(len(images))])

        responses = self.vlm.few_shot_query(images, texts, few_shot_ix, output_only=output_only, max_new_tokens=30)
        return responses
    
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    # model = MedFlamingo(config)
    # preds = model(torch.rand((1, 1, 192, 192)), {'Question': 'What is in this image? Options are A) Nothing, B) Something.', 'Answer': 'Nothing'}, ['Something', 'Nothing'])
    device = torch.device('cuda:0')
    vlm = MiniGPT4Module(config, MiniGPT4(config)).model.to(device)
    model = GenerativeModule(config, vlm=vlm)

    preds = model.closed_ended(torch.rand((100, 1, 192, 192)).half().to(device), [{'Question': 'What is in this image <ImageHere>? Options are A) Nothing, B) Something.', 'Answer': 'Nothing'}]*100, ['Something']*100, ['Something', 'Nothing'])
    print(preds)
    os.system('nvidia-smi')
    x = 3

if __name__ == "__main__": 
    test()