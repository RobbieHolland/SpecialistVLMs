
import hydra
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.gridspec as gridspec
from evaluation.figure_util import save_fig_path_creation
from models.vlm import VLM
import matplotlib.pyplot as plt
import numpy as np
import os
from clinical_capabilities.clinical_capabilities_util import add_schema
from tqdm import tqdm
import shutil

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    from slurm.util import record_job_id
    config = record_job_id(config)

    sys.path.append(config['flamingo_dir'])
    sys.path.append(config['octlatent_dir'])
    from dataset.retinal_text_dataset import RetinalTextDataset

    # Load data
    device = torch.device('cuda:0')
    dataset = RetinalTextDataset(config, set_='validation')
    if 'DescriptionAnnotated' in dataset.data_csv:
        dataset.data_csv = dataset.data_csv[dataset.data_csv['DescriptionAnnotated'] == True]
    dataset.data_csv = dataset.data_csv.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    data_loader = DataLoader(dataset, batch_size=config.dataset.task.n_images, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                        persistent_workers=False, pin_memory=False, num_workers=6, drop_last=False)

    images, targets = next(iter(data_loader))
    images = images.half().to(device)

    # Create results directory
    results_dir = os.path.join('evaluation/results/open_ended', config['job_id'])
    from evaluation.figure_util import make_folder_if_not_exists
    make_folder_if_not_exists(results_dir)

    # Save specialist annotations
    specialist_df = pd.DataFrame({'Retinal specialist': targets[2]}, index=targets[1])
    results_path = os.path.join(results_dir, 'specialist_annotations.csv')
    print('Saving results to', results_path)
    specialist_df.to_csv(results_path)

    # shrm_image = dataset._get_images(image_name='16044837-25-MFTXWLVHEJQQPLDBUKHLTVQPR+XTKMUXZYDPLGJSELBASSNPBNSBQZLPAIMKDYY+LETDCP.png')[0].unsqueeze(0).unsqueeze(0)
    # question = ["Describe the OCT image in detail and note any biomarkers or abnormalities, including the presence or absence of any subretinal hyperreflective material (SHRM). Then tell me if the image \'does\' or \'does not\' contain any subretinal hyperreflective material (SHRM)?"]
    # preamble = [" The OCT image shows a large area of subretinal hyperreflective material (SHRM) with a hyporeflective core, indicating the presence of a retinal pigment epithelial (RPE) detachment. Additionally, there is a small area of subretinal fluid and minimal intraretinal fluid. No intraretinal hyperreflective material (IRM) or intraretinal cystic rupture is observed. To conclude my findings, the OCT image does"]

    # Save images
    os.makedirs(results_dir, exist_ok=True)
    
    # Iterate through pretrained models
    all_outputs = []

    for model_spec in config.pretrained_models:
        print(f'Running {model_spec}')


        # Load model
        config.model.checkpoint_path = model_spec
        model = VLM(config.copy()).load(device=device)

        # shrm_output, samples = model.query(shrm_image.to(model.device), question, answer_preamble=preamble, output_only=True, return_samples=True)

        # Split images into batches
        for ix in tqdm(range(len(images))):
            batch_images = images[ix:ix+1]
            image_ids = targets[1][ix]
            specialist_annotation = targets[2][ix]

            image_dir = os.path.join(results_dir, f'image-{image_ids}')
            model_dir = os.path.join(image_dir, model_spec[0])

            print(model_dir)
            os.makedirs(os.path.join(model_dir), exist_ok=True)

            with open(os.path.join(model_dir, 'specialist_report' + '.txt'), 'w') as text_file:
                text_file.write(specialist_annotation)

            # shutil.copy(f'{config.dataset.task.figure_ready_images}/{image_ids}', f'{model_dir}/{image_ids}')
            # Save images
            plt.imshow(batch_images.cpu().squeeze().numpy(), cmap='gray', vmin=np.percentile(batch_images.cpu().squeeze().numpy(), 0.5), vmax=np.percentile(batch_images.cpu().squeeze().numpy(), 99.5))
            plt.axis('off')
            save_fig_path_creation(f'{model_dir}/{image_ids}')

            # Ask each set of questions
            for question in config.dataset.task.all_questions:
                print(f'On question {question["name"]}')

                query = add_schema(question['query'])
                query_output = model.query(batch_images, [query]*len(batch_images), max_new_tokens=config.dataset.task.max_new_tokens_cot, output_only=True)

                # Save response
                with open(os.path.join(model_dir, question['name'] + '.txt'), 'w') as text_file:
                    text_file.write(query_output[0])

                # if 'query2' in question:
                #     answer_preambles = query_output
                #     answer_preambles = [a + question['preamble'] for a in answer_preambles]
                #     full_query = add_schema(question['query'] + question['query2'])
                #     cot_output, sample = model.query(image, [full_query], answer_preamble=answer_preambles, max_new_tokens=2000, output_only=True, return_samples=True)
                #     query_output = [a + o for (a, o) in zip(answer_preambles, cot_output)]
        
        del model
        torch.cuda.empty_cache()

    x = 3

if __name__ == "__main__":
    test()