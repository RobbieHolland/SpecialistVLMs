import hydra
import sys
# from evaluation.tasks.general import Modality, PatientSex, AMDStage, SilverBiomarker, IntraretinalSubretinalFluid, FluidDetection, HypertransmissionDetection, BiomarkerClassification, HypertransmissionVsFluid
from dataset.text.tabular_to_prompt import TabularToPrompt
from dataset.text_util import list_and, valid_variable
import random
import numpy as np
import torch 
from models.wizardlm import load_wizardlm
import os
from datetime import datetime
import wandb 
from tqdm import tqdm
from tqdm import tqdm
from dataset.df_util import try_load

def generate_list_description(variables):
    wraps = [
        f"the {{selected_attribute}} is {{value}}",
        f"the {{selected_attribute}} are {{value}}",
    ]

    attribute_keys = ["Imaging biomarkers visible in the image", "Imaging biomarkers not present/detected", "AMD stage of the patient", "Visual acuity (measured in letter score)", "Subject's age", "Sex of the patient", "Image quality rating", "Eye scanned"]
    valid_indices = [i for i, var in enumerate(variables) if valid_variable(var)]
    # random.shuffle(valid_indices)

    selected_attributes = np.array(attribute_keys)[valid_indices]
    wrapped_attributes = [(attr, str(variables[attribute_keys.index(attr)])) for attr in selected_attributes.tolist()]

    formatted_output = '\n'.join([f'- {attr[0]}: {attr[1]}' for attr in wrapped_attributes])

    return formatted_output

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(config):
    wandb.init(project=config.wandb_project, config=dict(config), mode=config.wandb_mode)

    sys.path.append(config['octlatent_dir'])
    # Split my module across 4 gpus, one layer each
    dataset_version = config.dataset.task.version if config.dataset.task.version else (os.environ.get('SLURM_JOB_ID') if os.environ.get('SLURM_JOB_ID') else datetime.now().strftime('%Y-%m-%d'))
    from dataset.retinal_text_dataset import RetinalTextDataset

    # Load datasets
    dataset = RetinalTextDataset(config.copy(), set_='all')
    variable_gen = TabularToPrompt(config, dataset.data_csv)
    variables = dataset.data_csv[config.dataset.task.target]
    variables = variables.apply(variable_gen.generate_variables, axis=1)
    variables = variables.apply(generate_list_description)

    dataset.data_csv['Tabular report'] = variables.apply(lambda s: config.dataset.task.curriculum.annotation_prompt.replace('<Variables>', s))

    from utils.prompt import build_prompt
    dataset.data_csv['prompt'] = dataset.data_csv['Tabular report'].apply(lambda s: build_prompt(config, {'Question': s, 'Answer': ''}))

    # Load LLM
    device = torch.device('cuda:0')
    
    llm, tokenizer = load_wizardlm(config)
    from auto_gptq import exllama_set_max_input_length
    llm = exllama_set_max_input_length(llm, 8192)

    folder_path = os.path.join(config.dataset.task.output_path, f'{config.model.language_model.name}_annotations_{dataset_version}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # n = len(os.listdir(folder_path)) + 1
    file_path = os.path.join(folder_path, f'base.pkl')
    output_path = os.path.join(folder_path, f'{config.dataset.task.worker_id}.pkl')

    if not os.path.exists(file_path):
        dataset.data_csv = dataset.data_csv.sample(frac=1).reset_index(drop=True)
        initial_dataframe = dataset.data_csv[['ImageId', 'prompt']].copy()
        initial_dataframe[f'{config.model.language_model.name}_response'] = None
        initial_dataframe.to_pickle(file_path)
    
    global_dataframe = try_load(file_path)
    local_dataframe = global_dataframe.copy()

    # Create tqdm progress bars
    pbar_local = tqdm(desc="Current Job Progress", position=0, leave=True)
    # pbar_global = tqdm(total=len(dataset.data_csv), desc="Overall Progress", position=1, leave=True)

    while True:
        unprocessed_indices = global_dataframe[global_dataframe[f'{config.model.language_model.name}_response'].isnull() & local_dataframe[f'{config.model.language_model.name}_response'].isnull()].index.tolist()
        unprocessed_indices = [x for x in unprocessed_indices if x % config.dataset.task.total_workers == config.dataset.task.worker_id]

        if not unprocessed_indices:
            break

        selected_indices = np.random.choice(unprocessed_indices, min(config.dataset.task.batch_size, len(unprocessed_indices)), replace=False)
        subset = local_dataframe.loc[selected_indices]['prompt']

        # Generate Qs and As
        input_tokens = tokenizer(list(subset), return_tensors="pt", padding="longest", truncation=True, add_special_tokens=False)['input_ids'].to(device)
        outputs = llm.generate(inputs=input_tokens, temperature=config.dataset.task.temperature, do_sample=True, top_p=0.95, top_k=40, max_length=config.dataset.task.max_length, eos_token_id=tokenizer.encode(tokenizer.eos_token))
        
        # Parse and write result
        responses = []
        for output in outputs:
            original_length = len(input_tokens[0])
            new_tokens = output[original_length:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

        local_dataframe.loc[selected_indices, f'{config.model.language_model.name}_response'] = responses
        local_dataframe.to_pickle(output_path)

        # Update the tqdm progress bars
        pbar_local.update(len(selected_indices))

    pbar_local.close()
    # pbar_global.close()

if __name__ == "__main__":
    run()
    # save_load()