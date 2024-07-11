# Code taken from https://github.com/microsoft/LLaVA-Med
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from PIL import Image
import os
import pytorch_lightning as pl
import numpy as np

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

class LLavaMed(pl.LightningModule):
    def __init__(self, config, device):
        super(LLavaMed, self).__init__()
        self.config = config
        self.is_generative = True

        print('Adding path')
        sys.path.append(config.llava_med_dir)
        from llava.model import LlavaLlamaForCausalLM

        print('Loading llava-med')
        self.model = LlavaLlamaForCausalLM.from_pretrained(config.llava_med_model_path, cache_dir=config.pretrained_model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(config.llava_med_model_path, cache_dir=config.pretrained_model_dir)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)
        self.model = self.model.to(device)
        self.model.model.vision_tower[0] = self.model.model.vision_tower[0].to(device)

    def query(self, images, texts, answer_preamble=None, max_new_tokens=200, output_only=True, return_samples=False):
        DEFAULT_IMAGE_TOKEN = "<image>"
        DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
        DEFAULT_IM_START_TOKEN = "<im_start>"
        DEFAULT_IM_END_TOKEN = "<im_end>"

        answer_preamble = [''] * len(images) if answer_preamble is None else answer_preamble

        samples = {'Question': texts}
        from llava.conversation import conv_templates

        vision_tower = self.model.model.vision_tower[0]

        # Convert to PIL images
        images = (255 * images.cpu()).numpy().astype(np.uint8)
        images = [Image.fromarray(image.squeeze()) for image in images]
        images = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in images]
        images = torch.stack([image.half() for image in images], dim=0).to(self.model.device)
        # vision_tower.to(device='cuda', dtype=torch.float16)

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        prompts = []
        for qs, ap in zip(texts, answer_preamble):

            qs = qs.replace('<image>', '').strip()
            cur_prompt = qs

            if getattr(self.model.config, 'mm_use_im_start_end', False):
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            cur_prompt = cur_prompt + '\n' + '<image>'

            # if args.conv_mode == 'simple_legacy':
                # qs += '\n\n### Response:'

            # conv = default_conversation.copy()
            conv = conv_templates["simple"].copy()
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt() + ' ' + ap
            prompts.append(prompt)

        samples['Input'] = prompts
        inputs = self.tokenizer(prompts)

        input_ids = torch.as_tensor(inputs.input_ids).to(self.model.device)

        keywords = []
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images.to(self.model.device),
                do_sample=False,
                # temperature=0.7,
                max_new_tokens=max_new_tokens,
                # max_length=input_ids.shape[1] + max_new_tokens,
                # stopping_criteria=[stopping_criteria]
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        # outputs = outputs[:index].strip()
        outputs = [out.strip().replace('Assistant: ', '').replace('#', '') for out in outputs]
        outputs = [out.split('Human:')[0] for out in outputs]
        # outputs = outputs_reasoning + '\n The answer is ' + outputs
        if return_samples:
            return outputs, samples
        return outputs

import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test_input(config):
    device = torch.device('cuda:0')
    model = LLavaMed(config, device=device)

    from torch.utils.data import DataLoader
    from dataset.retinal_text_dataset import RetinalTextDataset
    dataset = RetinalTextDataset(config.copy(), set_='validation')

    data_loader = DataLoader(dataset, batch_size=config.model.batch_size, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                        persistent_workers=False, pin_memory=False, num_workers=6, drop_last=False)
    batch = next(iter(data_loader))
    images = batch[0].to(device)
    queries = ["Describe the OCT image list any biomarkers or abnormalities. Detail if there are any signs indicating that subretinal fluid might be present, even if there is only a small amount."] * len(images)
    answer_preambles = ["This appears"] * len(images)

    output = model.query(images, queries, answer_preamble=answer_preambles, output_only=True, return_samples=True)
    print(output)
    x = 3

if __name__ == "__main__":
    test_input()