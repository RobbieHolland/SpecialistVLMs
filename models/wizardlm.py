import hydra
import torch

def load_wizardlm(config):
    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained(config.model.language_model.path)
    model = LlamaForCausalLM.from_pretrained(
        config.model.language_model.path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    print(f'Loaded WizardLM onto {model.device}')
    return model, tokenizer

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(myconfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    import transformers
    import torch

    # Assuming 'config.model.language_model.path' is a string that points to the directory containing 'config.json'
    import transformers
    import torch

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    cache_dir=myconfig.pretrained_model_dir,
    )


    model, tokenizer = load_wizardlm(myconfig)

    prompt = "Tell me about AI"
    prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:'''

    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    print(tokenizer.decode(output[0]))

if __name__ == "__main__":
    test()