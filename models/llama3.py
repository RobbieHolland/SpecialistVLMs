import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import transformers
import hydra
import warnings
from models.util import set_llama3_pad_token
import warnings

from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class Llama3(pl.LightningModule):
    def __init__(self, config, device_8bit=0):
        super().__init__()
        self.config = config

        self.tokenizer = self.create_tokenizer()

        if self.config.model.language_model.initialize:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model.language_model.model_id,
                torch_dtype=torch.float16,
                load_in_8bit=config.model.language_model.load_in_8bit,
                # device_map={'': device_8bit},
                cache_dir=config.pretrained_model_dir,
            )
        else:
            warnings.warn('Loading 8-bit quantized models from HuggingFace can lead to errors when loading LLama3 from_config instead of from_pretrained.')
            config_model = AutoConfig.from_pretrained(
                config.model.language_model.model_id,  # Model ID for config
                cache_dir=config.pretrained_model_dir,
            )
            self.model = AutoModelForCausalLM.from_config(
                config_model,
                torch_dtype=torch.float16,
            )

        self.stop_words = ["<|eot_id|>"]
        self.stop_words_ids = [torch.Tensor(self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)).to(self.model.device), torch.Tensor(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)).to(self.model.device)]

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.language_model.model_id, cache_dir=self.config.pretrained_model_dir, use_fast=False)
        with warnings.catch_warnings():
            tokenizer = set_llama3_pad_token(tokenizer)
        return tokenizer

    def build_prompt(self, config, sample):
        messages = [
            {"role": "system", "content": "You are a helpful ophthalmological specialist chatbot capable of interpreting retinal OCT images."},
            {"role": "user", "content": f'Here is an encoding of a retinal OCT image.\n{sample["Question"]}'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt = prompt + sample['Answer']
    
        return prompt


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(myconfig):
    llama_model = Llama3(myconfig)
    questions = ['What color is grass?', 'What\'s going on with age-related macular degeneation?', 'Who is bigger out of 6 and 2?']
    answer_preambles = ['', '', '']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    llama_model.model = llama_model.model

    messages = [
        {"role": "system", "content": "You are a helpful ophthalmological specialist chatbot capable of interpreting retinal OCT images."},
        {"role": "user", "content": "Who are you?"},
    ]

    prompt = llama_model.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    
    # sentences = [build_prompt(myconfig, {'Question': text, 'Answer': answer}) for text, answer in zip(questions, answer_preambles)]
    input_ids = llama_model.tokenizer(prompt, padding=True, return_tensors="pt", truncation=True).to(device)
    input_embeds = llama_model.model.base_model.embed_tokens(input_ids['input_ids'])

    # stop_words_ids = [torch.Tensor(llama_model.tokenizer.encode(llama_model.tokenizer.eos_token, add_special_tokens=False)).to(device), torch.Tensor(llama_model.tokenizer.encode("<|eot_id|>", add_special_tokens=False)).to(device)]
    stop_words_ids = llama_model.stop_words_ids
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    outputs = llama_model.model.generate(
        # input_ids=input_ids, 
        # input_ids=input_ids['input_ids'].to(torch.int32), 
        inputs_embeds=input_embeds,
        attention_mask=input_ids['attention_mask'],
        max_new_tokens=200,
        # max_length=max_length,
        stopping_criteria=stopping_criteria,
        eos_token_id=llama_model.tokenizer.eos_token_id,
        num_beams=1,  # Set to 1 for greedy decoding
        do_sample=False,  # Disable sampling
        # top_p=top_p,
        # temperature=temperature,
        min_length=1,
        length_penalty=1.0,
        repetition_penalty=1,
    )

    responses = [llama_model.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    x = 3


if __name__ == "__main__":
    test()