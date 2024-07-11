# Based off of https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/models/minigpt4.py

import logging
import random
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from models.get_model import get_vision_model, get_language_model
from models.transformer import Miniformer
from models.perceiver import Perceiver
from models.position_embeddings import PositionalEncoding

import hydra

class MiniGPT4(nn.Module):
    def __init__(
        self,
        config,
        freeze_vit=True,
        device=None,
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        self.config = config
        self.is_generative = True
        
        if config.model.checkpoint_path:
            self.name = config.model.checkpoint_path
            print(f'Loading model from {config.model.checkpoint_path}')
        else:
            self.name = 'unnamed'
            print(f'Creating new MiniGPT4 model')

        print('Loading image encoder')
        self.visual_encoder = get_vision_model(config)#.to(self.device)
        self.visual_encoder = self.maybe_to(self.visual_encoder, device)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if name != 'learnable_p_tokens' and (name.split('.')[1] not in config.model.vision_encoder.unfrozen_layers):
                    param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()

            logging.info("freeze vision encoder")
        print('Loading image encoder Done')

        # img_f_dim = self.visual_encoder.num_features * 4
        img_f_dim = self.config.model.vision_encoder.feature_dim
        img_f_dim *= self.config.model.vision_encoder.concatenate_tokens if self.config.model.vision_encoder.concatenate_tokens else 1

        print('Loading language model')
        # llm = LLM(config)
        self.llm = get_language_model(config)
        self.llm = self.maybe_to(self.llm, device)

        self.llama_model = self.llm.model
        self.llama_tokenizer = self.llm.tokenizer
        self.tokenizer = None
        self.tokenizers = {}
        self.image_patch_position_embedding = None #PositionalEncoding(img_f_dim, max_len=self.config.dataset.n_vision_tokens, device=self.device)

        # Freeze language model
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading language model Done')

        # Mapper

        if config.model.miniformer.name:
            print('Using Miniformer')
            miniformer = Perceiver(config, img_f_dim).to(self.device)

            if config.model.miniformer.linear_upsample:
                self.llama_proj = miniformer
            else:
                self.upsampler = nn.Linear(
                    img_f_dim, self.llama_model.config.hidden_size
                )
                self.llama_proj = nn.Sequential(miniformer, self.upsampler)
        else:
            self.llama_proj = nn.Linear(
                img_f_dim, self.llama_model.config.hidden_size
            )#.to(self.device)

        self.llama_proj = self.maybe_to(self.llama_proj, device)
        
        print('Linear projection', self.llama_proj)

        if self.config.model.p_tokens:
            print('Using p-tokens', self.config.model.p_tokens)
            self.learnable_p_tokens = nn.Parameter(torch.empty(self.config.model.p_tokens, self.llama_model.config.hidden_size))
            nn.init.xavier_normal_(self.learnable_p_tokens)
            self.learnable_p_tokens = nn.Parameter(self.learnable_p_tokens.unsqueeze(0))
            # self.maybe_to(self.learnable_p_tokens, device)

        if self.config.model.s_tokens:
            print('Using s-tokens', self.config.model.s_tokens)
            self.learnable_s_tokens = nn.Parameter(torch.empty(self.config.model.s_tokens, self.llama_model.config.hidden_size))
            nn.init.xavier_normal_(self.learnable_s_tokens)
            self.learnable_s_tokens = nn.Parameter(self.learnable_s_tokens.unsqueeze(0))
            # self.maybe_to(self.learnable_s_tokens, device)

        self.max_txt_len = config.dataset.task.max_txt_len

        self.prompt_list = None

        self.softmax = nn.Softmax(dim=2)
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #             torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[sw.to(self.device) for sw in self.llm.stop_words_ids])])

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'MiniGPT-4 has {trainable_params} trainable parameters')

    def get_tokenizer(self):
        if self.config.devices < 2:
            if self.tokenizer is None:
                self.tokenizer = self.llm.create_tokenizer()
            return self.tokenizer
        else:
            if self.llm.device.index not in self.tokenizers.keys():
                self.tokenizers[self.llm.device.index] = self.llm.create_tokenizer()
            return self.tokenizers[self.llm.device.index]

    def maybe_to(self, module, device):
        module = module.to(device) if device else module
        torch.cuda.empty_cache()
        return module

    @property
    def device(self):
        return self.llm.device

    def encode_img(self, image, with_p_s_tokens=True):
        device = image.device

        with self.maybe_autocast():
            image_embeds = self.visual_encoder.feature_tokens(image).to(device)

            if self.config.model.image_patch_position_embedding:
                image_embeds += self.image_patch_position_embedding(image_embeds)

            inputs_llama = self.llama_proj(image_embeds)

            if self.config.model.p_tokens and with_p_s_tokens:
                inputs_llama = torch.cat((self.learnable_p_tokens.to(device).repeat(inputs_llama.shape[0], 1, 1), inputs_llama), dim=1)

            if self.config.model.s_tokens and with_p_s_tokens:
                inputs_llama = torch.cat((inputs_llama, self.learnable_s_tokens.to(device).repeat(inputs_llama.shape[0], 1, 1)), dim=1)

            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.llm.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def prompt_wrap(self, img_embeds, atts_img, prompts, tokenizer, pad_side=None):
        pad_side = pad_side if pad_side is not None else tokenizer.padding_side

        if prompts:
            token_lists = []
            emb_lists = []
            indices = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')

                p_before_tokens = tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids).to(img_embeds.device)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids).to(img_embeds.device)

                # Combine pre, image, and post tokens
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)

                wrapped_token = torch.cat([p_before_tokens.input_ids, -torch.ones(each_img_embed.shape[0]).to(p_before_tokens.input_ids.device).unsqueeze(0), p_after_tokens.input_ids], dim=1).to(torch.int32)
                token_lists.append(wrapped_token)

                indices += [(p_before_embed.shape[1], each_img_embed[None].shape[1], p_after_embed.shape[1])]
                
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_token = torch.tensor(tokenizer.pad_token_id, device=img_embeds.device)
            pad_emb = self.embed_tokens(pad_token)

            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_tokens = pad_token.unsqueeze(0).expand(len(emb_lens), max(emb_lens), -1).clone().squeeze(-1)
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)

            for i, emb in enumerate(emb_lists):
                if pad_side == 'left':
                    wrapped_embs[i, -emb_lens[i]:] = emb  # Notice the change here for left-padding
                    wrapped_tokens[i, -emb_lens[i]:] = token_lists[i]  # Notice the change here for left-padding
                    wrapped_atts[i, -emb_lens[i]:] = 1   # Notice the change here for left-padding
                    # wrapped_atts[i, -emb_lens[i]:] = (token_lists[i] != tokenizer.pad_token_id).long()   # Notice the change here for left-padding
                elif pad_side == 'right':
                    wrapped_embs[i, :emb_lens[i]] = emb  # Notice the change here for left-padding
                    wrapped_tokens[i, :emb_lens[i]] = token_lists[i]  # Notice the change here for left-padding
                    wrapped_atts[i, :emb_lens[i]] = 1   # Notice the change here for left-padding
                    # wrapped_atts[i, :emb_lens[i]] = (token_lists[i] != tokenizer.pad_token_id).long()   # Notice the change here for left-padding
            return wrapped_embs, wrapped_tokens, wrapped_atts, indices
        else:
            return img_embeds, atts_img

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens
    
    # Input is {'image': ..., 'instruction_input': ..., 'answer': ...}
    def form_input(self, samples, answer_preamble=None, query=False, return_indices=False, pad_side=None):
        image = samples["Image"]
        tokenizer = self.get_tokenizer()

        if torch.is_tensor(samples['Question']):
            samples['Question'] = [tokenizer.decode([t for t in x if t != tokenizer.pad_token_id]) for x in samples['Question']]
            samples['Answer'] = [tokenizer.decode([t for t in x if t != tokenizer.pad_token_id]) for x in samples['Answer']]

        answer_preamble = [''] * len(samples['Question']) if answer_preamble is None else answer_preamble
        samples['Input'] = [self.llm.build_prompt(self.config, {'Question': text, 'Answer': answer}) for text, answer in zip(samples['Question'], answer_preamble)]
        samples['Input'] = [text.replace('encoding of a retinal OCT image', 'encoding of a retinal OCT image <Img><ImageHere></Img>', 1) if '<ImageHere>' not in text else text for text in samples['Input']]

        img_embeds, atts_img = self.encode_img(image)

        inputs_embeds, inputs_tokens, attention_mask, subsequence_indices = self.prompt_wrap(img_embeds, atts_img, samples["Input"], tokenizer, pad_side=pad_side)

        if query:
            if return_indices:
                return inputs_embeds, inputs_tokens, attention_mask, subsequence_indices
            return inputs_embeds, inputs_tokens, attention_mask

        # self.llama_tokenizer.padding_side = "left"
        text = [t + self.llm.stop_words[0] for t in samples["Answer"]]

        to_regress_tokens = tokenizer(
            text,
            return_tensors="pt",
            # padding="longest",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)


        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(inputs_embeds, attention_mask, to_regress_embeds, to_regress_tokens.attention_mask)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]:input_lens[i] + len(target)] = target

        return inputs_embeds, inputs_tokens, attention_mask, targets

    def forward(self, samples):
        inputs_embeds, inputs_tokens, attention_mask, targets = self.form_input(samples)

        if not self.config.model.language_model.load_in_8bit:
            self.llama_model = self.llama_model.to(inputs_embeds.device)
            self.llama_model.model = self.llama_model.model.to(inputs_embeds.device)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        outputs.loss = outputs.loss.to(targets.device)
        
        return outputs.loss

    def embed_tokens(self, token_ids):
        device = token_ids.device
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
            embeds = embeds.to(device)
        return embeds
    
    # Returns the probability distribution for the next token
    def softmax_logits(self, input_tensor, texts=None, answer_preambles=None, class_tokens_positions=[-1], class_tokens=[0]):
        images = input_tensor.to(self.llama_model.device)

        samples = {
            'Image': images,
            'Question': texts,
            'Answer': [''],
        }

        inputs_embeds, inputs_tokens, _ = self.form_input(samples, answer_preamble=answer_preambles, query=True, pad_side='left')
        outputs = self.llama_model(inputs_embeds=inputs_embeds)
        token_distribution = self.softmax(outputs.logits)

        cl = inputs_tokens[:,torch.Tensor(class_tokens_positions).long() + 1]
        class_tokens_distribution = token_distribution[:, torch.Tensor(class_tokens_positions).long(), class_tokens].mean(1, keepdims=True)

        return class_tokens_distribution

    def word_attention(self, images, word=' fluid'):
        samples = {
            'Image': images,
            'Question': [''],
            'Answer': [''],
        }

        inputs_llama, atts_llama = self.encode_img(images, with_p_s_tokens=False)
        word_tokens = self.llama_tokenizer(word, return_tensors="pt", add_special_tokens=False).to(inputs_llama.device)
        word_embedding = self.embed_tokens(word_tokens.input_ids).to(inputs_llama.device)

        inputs_embeds = torch.cat((inputs_llama, word_embedding.repeat(images.shape[0], 1, 1)), dim=1)

        attention_mask = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.shape[1])
        attention_mask[:, -word_embedding.shape[1]:, :-word_embedding.shape[1]] = 1

        self.llama_model.config.output_attentions = True
        outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=True)
        self.llama_model.config.output_attentions = False

        sequence_attentions = torch.stack(outputs.attentions).detach().cpu()
        word_queries = sequence_attentions[:,:,:,-word_tokens.input_ids.shape[1]:,:-word_tokens.input_ids.shape[1]].mean(-2)
        language_image_attention = word_queries

        image_attention = None
        if self.config.model.vision_encoder.name == 'retfound':
            with self.maybe_autocast():
                _, image_attention = self.visual_encoder.feature_tokens(images, return_attentions=True)
                
                p_tokens = self.config.model.vision_encoder.p_tokens
                n_tokens = self.config.dataset.n_vision_tokens
                p_token_image_attention = image_attention[:,:,-p_tokens:,1:1+n_tokens]

                word_queries = word_queries.transpose(0, 1)
                p_token_image_attention = p_token_image_attention.transpose(1, 2)
                language_image_attention = torch.einsum('abcd,adef->abcef', word_queries.float().detach().cpu(), p_token_image_attention.float().detach().cpu())

                s = language_image_attention.shape
                language_image_attention = language_image_attention.reshape(s[0], s[1], s[2] * s[3], s[4])
                language_image_attention.transpose(0, 1)

        return language_image_attention.detach().cpu()

    def attention(self, images, texts, answer_preamble=None):
        samples = {
            'Image': images,
            'Question': texts,
            'Answer': [''],
        }
        
        image_attention = None
        if self.config.model.vision_encoder.name == 'retfound':
            with self.maybe_autocast():
                _, image_attention = self.visual_encoder.feature_tokens(images, return_attentions=True)

        inputs_embeds, inputs_tokens, attention_mask, subsequence_indices = self.form_input(samples, query=True, return_indices=True, answer_preamble=answer_preamble)

        self.llama_model.config.output_attentions = True
        outputs = self.llama_model(inputs_embeds=inputs_embeds, output_attentions=True)
        self.llama_model.config.output_attentions = False

        sequence_attentions = torch.stack(outputs.attentions).detach().cpu()
        return samples, inputs_tokens, subsequence_indices, sequence_attentions, image_attention
    
    def query(self, images, texts, answer_preamble=None, max_new_tokens=150, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1e-3, output_only=False, return_samples=False):
        samples = {
            'Image': images,
            'Question': texts,
            'Answer': [''],
        }

        inputs_embeds, inputs_tokens, _ = self.form_input(samples, answer_preamble=answer_preamble, query=True, pad_side='left')

        samples['Input tokens'] = inputs_tokens

        # Was not previously using attention mask for generate
        attention_mask = (inputs_tokens != self.get_tokenizer().pad_token_id).long()

        self.llama_model.generation_config.temperature = None
        self.llama_model.generation_config.top_p = None
        
        print('Inputs', inputs_embeds.shape)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            num_beams=1,
            do_sample=False,
            # pad_token_id=self.llama_tokenizer.pad_token_id,
            # eos_token_id=self.llama_tokenizer.encode(self.llm.stop_words[0], add_special_tokens=False)[0],
            # num_beams=1,  # Set to 1 for greedy decoding
            # min_length=min_length,
            # stopping_criteria=self.stopping_criteria,
            # repetition_penalty=repetition_penalty,
            # length_penalty=length_penalty,
            # max_length=max_length,
            top_p=None,
            temperature=None,
        )

        samples['Output tokens'] = outputs
        # print('Outputs', outputs.shape)

        responses = []
        for instruction_input, output_token in zip(samples['Question'], outputs):
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.get_tokenizer().decode(output_token, add_special_tokens=False, skip_special_tokens=True)
            if not output_only:
                output_text = instruction_input + output_text
            responses.append(output_text)

        if return_samples:
            return responses, samples
        else:
            return responses

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def debug(config):
    model = MiniGPT4(config, 'biomed')
    # img = torch.randn([1, 1, 192, 192]).half().to(model.device)
    n = 20
    img = torch.randn([n, 1, 192, 192]).half().to(model.device)
    # print('Image embeddings', embeds.shape)

    samples = {
        'image': img,
        'instruction_input': ['Here is an image <ImageHere>. List what you see in it.']*n,
        'answer': ['I see nothing']*n,
        }
    inputs_embeds, _, _, _ = model.form_input(samples)
    print('Input embeds', inputs_embeds.shape)

    outputs = model(samples)
    print(outputs.loss)
    print(outputs.logits.shape)


    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    print(output_text)
    # embeds, atts = model.encode_img(img)

if __name__ == "__main__":
    debug()