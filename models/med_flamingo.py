import sys
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
import os
import pytorch_lightning as pl
import torch
from PIL import Image
import numpy as np

class MedFlamingo(pl.LightningModule):
    def __init__(self, config, device):
        super(MedFlamingo, self).__init__()
        self.config = config
        # self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        accelerator = Accelerator(cpu=False) #when using cpu: cpu=True

        self.dev = accelerator.device
        self.is_generative = True

        print('Loading model..', self.dev)

        # >>> add your local path to Llama-7B (v1) model here:
        llama_path = self.config['llama_path']
        print(llama_path)
        if not os.path.exists(llama_path):
            raise ValueError('Llama model not yet set up, please check README for instructions!')

        self.model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            cross_attn_every_n_layers=4
        )
        from src.utils import FlamingoProcessor
        self.processor = FlamingoProcessor(tokenizer, image_processor)

        # load med-flamingo checkpoint:
        checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt", cache_dir=self.config['pretrained_model_dir'])
        print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.dev), strict=False)

        # go into eval model and prepare:
        self.model = accelerator.prepare(self.model)
        is_main_process = accelerator.is_main_process
        self.model.eval()

    def query(self, images, texts, max_new_tokens=150, answer_preamble=None, output_only=False, return_samples=False):
        return self.forward(images, texts, max_new_tokens=max_new_tokens, answer_preamble=answer_preamble, output_only=output_only, return_samples=return_samples)

    def forward(self, images, texts, answer_preamble=None, max_new_tokens=150, output_only=False, return_samples=False):
        from scripts.demo_utils import clean_generation
        
        # example few-shot prompt:
        # prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|><image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|><image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|><image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|><image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|><image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|><image>Question: Where is the largest mass located in the cerebellum? Answer:"
        # prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What image modality is this? Answer:"
        # prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What signs, if any, of age-related macular degneration are present in the image? Answer:"
        samples = {'Question': texts}

        # Preprocess images
        images = (255 * images.cpu()).numpy().astype(np.uint8)
        pixels = self.processor.preprocess_images([T.ToPILImage()(image.squeeze()) for image in images]).to(self.dev)
        pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1).transpose(0,1)

        answer_preamble = answer_preamble if answer_preamble is not None else [''] * len(texts)

        def build_prompt(inp, preamble):
            # return f"You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: {inp['Question']} Answer: {inp['Answer']}"
            return f"You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: {inp} Answer:{preamble}"
        # tokenized_data = [self.processor.encode_text(build_prompt(text)) for text in texts]

        self.processor.tokenizer.padding_side = "left" 
        # For generation padding tokens should be on the left
        prompts = [build_prompt(text, preamble) for (text, preamble) in zip(texts, answer_preamble)]
        samples['Input'] = prompts

        print('Prompts', prompts)
        tokenized_data = self.processor.tokenizer(prompts, return_tensors="pt", padding=True)
        # tokenized_data = {k: torch.stack([d[k].squeeze(0) for d in tokenized_data]) for k in tokenized_data[0]}
        """
        Step 4: Generate response 
        """

        # actually run few-shot prompt through model:
        print('Generate from multimodal few-shot prompt')
        # print(tokenized_data["input_ids"].shape, tokenized_data["attention_mask"].shape, pixels.shape)

        generated_text = self.model.generate(
            vision_x=pixels.to(self.dev),
            lang_x=tokenized_data["input_ids"].to(self.dev),
            attention_mask=tokenized_data["attention_mask"].to(self.dev),
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        responses = []
        for text in generated_text:
            response = self.processor.tokenizer.decode(text)
            response = clean_generation(response)
            if output_only:
                response = response.split('Answer:')[1]
            responses.append(response)

        if return_samples:
            return responses, samples
        return responses

import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test_preamble(config):
    device = torch.device('cuda:0')
    model = MedFlamingo(config, device=device)

    from torch.utils.data import DataLoader
    from dataset.retinal_text_dataset import RetinalTextDataset
    dataset = RetinalTextDataset(config.copy(), set_='validation')

    data_loader = DataLoader(dataset, batch_size=config.model.batch_size, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                        persistent_workers=False, pin_memory=False, num_workers=6, drop_last=False)
    batch = next(iter(data_loader))
    images = batch[0]
    queries = ["Describe this image in detail"] * len(images)
    answer_preambles = [""] * len(images)

    output = model.query(images, queries, answer_preamble=answer_preambles, output_only=True, return_samples=True)
    print(output)
    x = 3


if __name__ == "__main__":
    test_preamble()