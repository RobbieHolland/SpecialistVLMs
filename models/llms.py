# Load model directly
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Vicuna(pl.LightningModule):
    def __init__(self, config, device_8bit=0):
        super().__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.language_model.model_id,
            torch_dtype=torch.float16,
            load_in_8bit=config.model.language_model.load_in_8bit,
            device_map={'': device_8bit},
            cache_dir=config.pretrained_model_dir,
        )
        self.tokenizer = self.create_tokenizer()

        self.stop_words = [self.tokenizer.eos_token]
        self.stop_words_ids = [torch.Tensor(self.tokenizer.encode(t)).to(self.model.device) for t in self.stop_words]

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.language_model.model_id, use_fast=False)
        tokenizer.padding_side = "left"
        return tokenizer

    def build_prompt(self, config, sample):
        return f"<s>{config.model.language_model.prompt.preamble} {config.model.language_model.prompt.question.replace('<Question>', sample['Question']).replace('<Answer>', sample['Answer'])}"

import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    llm = Vicuna(config)
    x = 3

if __name__ == "__main__":
    test()
    