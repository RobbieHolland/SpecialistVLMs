import numpy as np
import hydra
from dataset.oct_dataset import OCTDataset
from dataset.image_transforms import augmentation_suites
import torch
import random
import copy
import pandas as pd
from dataset.text_util import parse_qa
import os 
from dataset.df_util import merge_unique
from dataset.text_util import valid_variable
from transformers import AutoTokenizer
from models.util import set_llama3_pad_token

def fix_location(p, l):
    if not (valid_variable(p) and l in ['Left', 'Right', 'Center']):
        return l
    if l == 'Center':
        return 'Fovea'
    if p == 0:
        if l == 'Left':
            return 'Nasal'
        return 'Temporal'
    elif p == 1:
        if l == 'Right':
            return 'Nasal'
        return 'Temporal'

class RetinalTextDataset(OCTDataset):
    def __init__(self,
                 config,
                 set_='validation',
                ):
        data_augment, central_crop = augmentation_suites['standard'](config.dataset)
        # image_transform = central_crop
        image_transform = {'train': data_augment, 'validation': central_crop, 'test': central_crop, 'all': central_crop}[set_]

        def format_transform(x):
            return torch.Tensor((np.array(x))).float()
        self.config = config

        metadata = config.dataset.metadata
        if '.csv' not in config.dataset.metadata:
            metadata = f"{config.dataset.metadata}/{set_}.pkl"

        dataset_config = copy.copy(dict(config.dataset))
        dataset_config['target'] = dataset_config['task']['target']

        def full_image_transform(x):
            return image_transform(format_transform(x))

        super().__init__(dataset_config, set_, metadata, 'ImageId', full_image_transform, None, config.seed)

        silver_biomarkers = pd.read_pickle(config.tabular_biomarker_variables)
        self.data_csv = merge_unique(self.data_csv, silver_biomarkers, on='ImageId')

        specialist_reports = pd.read_csv(config.specialist_description_annotations)[['ImageId', 'Annotation', 'DescriptionAnnotated', 'Description annotating clinician']]
        specialist_reports = specialist_reports.loc[~specialist_reports['DescriptionAnnotated'].isna()]
        tabular_annotations = pd.read_csv(config.specialist_tabular_annotations)
        self.data_csv = merge_unique(self.data_csv, specialist_reports, on='ImageId')
        self.data_csv = merge_unique(self.data_csv, tabular_annotations, on='ImageId')

        for location_column in [c for c in self.data_csv.columns if isinstance(c, str) and 'location' in c]:
            self.data_csv[location_column] = self.data_csv[['EyePosition', location_column]].apply(
                lambda p: fix_location(p['EyePosition'], p[location_column]), axis=1
            )

        if 'llm_qs_as' in self.config.dataset.task.keys() and len(self.config.dataset.task.llm_qs_as) > 0:
            self.load_and_process_llm_annotations()

        self.data_csv = self.filter_dataset(None, self.target_columns[0])

        x = 3
    
    def load_and_process_llm_annotations(self):
        all_llm_annotations = None
        for llm_annotations_set in self.config.dataset.task.llm_qs_as:
            # Check if it's a file
            if os.path.isfile(llm_annotations_set):
                llm_annotations = pd.read_pickle(llm_annotations_set)

            # Check if it's a directory
            elif os.path.isdir(llm_annotations_set):
                all_files = [os.path.join(llm_annotations_set, f) for f in os.listdir(llm_annotations_set) if f.endswith('.pkl')]
                
                # Load the first dataframe from the directory to initialize llm_annotations
                llm_annotations = pd.read_pickle(all_files[0])
                
                # Combine the rest of the dataframes in the directory with llm_annotations
                for file in all_files[1:]:
                    df = pd.read_pickle(file)
                    llm_annotations = llm_annotations.combine_first(df)

            else:
                raise Exception(f'{llm_annotations_set} does not exist')

            if all_llm_annotations is None:
                all_llm_annotations = llm_annotations
            else:
                all_llm_annotations = merge_unique(all_llm_annotations, llm_annotations, 'ImageId')

        # Merge and process
        self.data_csv = merge_unique(self.data_csv, all_llm_annotations, on='ImageId')

        self.data_csv[self.config.dataset.task.target] = None
        for input_col in self.config.dataset.task.qa_input:
            qs_as = self.data_csv[input_col].apply(parse_qa)
            self.data_csv[self.config.dataset.task.target] = self.data_csv[self.config.dataset.task.target].combine(qs_as, lambda x, y: x + y if x is not None and y is not None else y or x)
        x = 3 

    def class_balance(self, column):
        min_count = self.data_csv[column].value_counts().min()
        return pd.concat([self.data_csv[self.data_csv[column] == val].sample(min_count, random_state=self.config.seed) for val in self.data_csv[column].unique()]).sample(frac=1, random_state=self.config.seed).reset_index(drop=True)

    def standard_get(self, index):
        image = torch.Tensor(self._get_images(index)[0]).unsqueeze(0)
        targets = self._get_targets(index)

        return image, targets
    
    def __getitem__(self, index):
        return self.standard_get(index)

    # [ast.literal_eval(item) if isinstance(item, str) else None for item in [variables] if isinstance(item, str) or math.isnan(item)][0]

    def _get_targets(self, index, cols=None, random_crop_params=None):
        return self.data_csv.loc[index, self.target_columns[0]]
    
    @staticmethod
    def custom_collate(batch):
        # Separate images and variables
        images, variables = zip(*batch)
        images = torch.stack(images)
        
        # Transpose variables to handle mixed types
        transposed_vars = zip(*variables)
        collated_vars = []
        
        for items in transposed_vars:
            if all(isinstance(item, (float, int, bool, np.bool_)) for item in items):
                collated_vars.append(torch.tensor(items))
            else:  # Assuming the rest are strings
                collated_vars.append(list([y for y in items]))
        
        # Return collated images and variables
        return images, collated_vars

    
class RetinalLLMDataset(RetinalTextDataset):
    def __init__(self,
                config,
                set_='test',
                deterministic=False,
                model_config=None):

        super().__init__(config, set_)
        self.standard = False

        self.deterministic = deterministic
        self.model_config = model_config

    def random_closed_ended(self, index):
        if (self.target_columns[0] == 'WizardLM_Qs_As') or (self.target_columns[0] == 'LLM_Qs_As'):
            choices = self.data_csv.loc[index, self.target_columns[0]]
            if self.deterministic:
                deterministic_index = hash(str(index)) % len(choices)
                sample = choices[deterministic_index]
            else:
                sample = random.choice(choices)
        else:
            raise Exception("Not implemented.")

        return sample
    
    def __getitem__(self, index):
        if self.standard:
            return self.standard_get(index)
        image = torch.Tensor(self._get_images(index)[0]).unsqueeze(0)
        samples = self.random_closed_ended(index)
        samples['Image'] = image
        return samples
    
    def create_custom_collate(self):
        @staticmethod
        def custom_collate(batch):
            # Not optimal for memory but is important if using DDP training
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model.language_model.model_id, cache_dir=self.model_config.pretrained_model_dir, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer = set_llama3_pad_token(tokenizer)

            keys = batch[0].keys()
            collated_batch = {key: [] for key in keys}
            
            for item in batch:
                for key in keys:
                    collated_batch[key].append(item[key])
            
            for key in collated_batch.keys():
                if isinstance(collated_batch[key][0], str):
                    collated_batch[key] = tokenizer(collated_batch[key], return_tensors="pt", padding="longest", truncation=False, add_special_tokens=False).input_ids

            collated_batch['Image'] = torch.stack(collated_batch['Image'])
            return collated_batch
        return custom_collate

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    dataset = RetinalLLMDataset(config, set_='validation')
    sample = dataset.__getitem__(1)
    print(sample)
    print('Success')
    x = 3

if __name__ == "__main__": 
    test()