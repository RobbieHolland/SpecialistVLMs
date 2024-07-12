import hydra
import sys
import yaml
import os
from datetime import datetime
from tqdm import tqdm
from models.chatgpt import ChatGPT
from dataset.df_util import try_load
from clinical_capabilities.clinical_capabilities_util import add_schema

def format_dataframe(df):
    formatted_strings = []
    
    for index, row in df.iterrows():
        formatted_string = f"{int(row['Annotation_Id'])}\t{row['Annotation']}"
        formatted_strings.append(formatted_string)
    
    return "\n".join(formatted_strings)

def prepare_dataframe(config):
    sys.path.append(config['octlatent_dir'])
    # Split my module across 4 gpus, one layer each
    dataset_version = config.dataset.task.version if config.dataset.task.version else (os.environ.get('SLURM_JOB_ID') if os.environ.get('SLURM_JOB_ID') else datetime.now().strftime('%Y-%m-%d'))
    from dataset.retinal_text_dataset import RetinalTextDataset

    # Load datasets
    dataset = RetinalTextDataset(config.copy(), set_='all')
    dataset.data_csv = dataset.data_csv.sort_values(by='Annotation_Id')
    
    folder_path = os.path.join(config.dataset.task.output_path, f'{config.model.language_model.name}_annotations_{dataset_version}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # n = len(os.listdir(folder_path)) + 1
    file_path = os.path.join(folder_path, f'base.pkl')
    output_path = os.path.join(folder_path, f'{config.dataset.task.curriculum.output_column_name}_{config.dataset.task.worker_id}.pkl')

    if not os.path.exists(file_path):
        initial_dataframe = dataset.data_csv[['ImageId', 'Annotation_Id', 'Annotation']].copy()
        initial_dataframe[f'{config.model.language_model.name}_response'] = None
        initial_dataframe.to_pickle(file_path)
    
    global_dataframe = try_load(file_path)
    local_dataframe = global_dataframe.copy()
    return local_dataframe, output_path

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def annotate(config):
    print(f'Running: {config.dataset.task.curriculum.output_column_name}')
    dataset_version = config.dataset.task.version if config.dataset.task.version else (os.environ.get('SLURM_JOB_ID') if os.environ.get('SLURM_JOB_ID') else datetime.now().strftime('%Y-%m-%d'))
    folder_path = os.path.join(config.dataset.task.output_path, f'{config.model.language_model.name}_annotations_{dataset_version}')
    output_path = os.path.join(folder_path, f'{config.dataset.task.curriculum.output_column_name}_{config.dataset.task.worker_id}.pkl')


    if os.path.exists(output_path):
        local_dataframe = try_load(output_path)
    else:
        local_dataframe, output_path = prepare_dataframe(config)
        local_dataframe[config.dataset.task.curriculum.output_column_name] = None
    
    print(f'Output will be written to {output_path}')
    
    # Initialize column 'a' with None or some default value
    chatgpt = ChatGPT(config.model.language_model.openai_api_key)

    for index, row in tqdm(local_dataframe.loc[local_dataframe[config.dataset.task.curriculum.output_column_name].isna()].iterrows()):
        chatgpt_input = config.dataset.task.curriculum.annotation_prompt.replace('<Variables>', row['Annotation'])
        chatgpt_input = add_schema(chatgpt_input)

        reply = chatgpt.generate(chatgpt_input, temperature=config.model.language_model.temperature, endpoint=config.model.language_model.endpoint)

        # Update the 'a' column with the value of reply for the current row
        local_dataframe.at[index, config.dataset.task.curriculum.output_column_name] = reply

        # Save the DataFrame to a pickle file
        local_dataframe.to_pickle(output_path)

    x = 3

if __name__ == "__main__":
    annotate()
