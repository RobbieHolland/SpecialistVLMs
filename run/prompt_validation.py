
import hydra
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
from evaluation.figure_util import save_fig_path_creation
from models.vlm import VLM
import matplotlib.pyplot as plt
import numpy as np
import os
from clinical_capabilities.clinical_capabilities_util import add_schema
from tqdm import tqdm
from dataset.retinal_text_dataset import RetinalTextDataset
from run.closed_ended_figures import aggregate_predictions, aggregate_f1

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def open_ended_prompt_testing(config):
    from slurm.util import record_job_id
    config = record_job_id(config)

    sys.path.append(config['flamingo_dir'])
    sys.path.append(config['octlatent_dir'])

    # Load data
    device = torch.device('cuda:0')
    dataset = RetinalTextDataset(config, set_='validation')
    dataset.data_csv = dataset.data_csv[dataset.data_csv['TabularAnnotated'] == True]

    data_loader = DataLoader(dataset, batch_size=config.dataset.task.n_images, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                        persistent_workers=False, pin_memory=False, num_workers=6, drop_last=False)

    images, targets = next(iter(data_loader))
    images = images.half().to(device)

    # Create results directory
    results_dir = os.path.join('evaluation/test_bed/', config['job_id'])
    from evaluation.figure_util import make_folder_if_not_exists
    make_folder_if_not_exists(results_dir)

    # Save specialist annotations
    if len(targets) > 2:
        specialist_df = pd.DataFrame({'Retinal specialist': targets[2]}, index=targets[1])
        results_path = os.path.join(results_dir, 'specialist_annotations.csv')
        print('Saving results to', results_path)
        specialist_df.to_csv(results_path)

    for model_spec in config.pretrained_models:
        print(f'Running {model_spec}')

        # Load model
        config.model.checkpoint_path = model_spec
        model = VLM(config.copy()).load(device=device)

        model_dir = os.path.join(results_dir, model_spec[3])

        # Split images into batches
        for ix in tqdm(range(0, len(images))):
            image_dir = os.path.join(model_dir, f'Image-{ix}')
            image = images[ix:ix+1]
            image_ids = targets[1][ix:ix+1]
            print(f'On images {image_ids}')

            plt.imshow(image.cpu().squeeze().numpy(), cmap='gray', vmin=np.percentile(image.cpu().numpy(), 0.5), vmax=np.percentile(image.cpu().numpy(), 99.5))
            plt.axis('off')
            save_fig_path_creation(os.path.join(image_dir, str(ix) + '-' + image_ids[0].split('.png')[0] + '.png'))

            # Ask each set of questions
            for question in config.dataset.task.all_questions:
                print('Current question', question['name'])

                query = add_schema(question['query'])
                query_output = model.query(image, [query], max_new_tokens=1200, output_only=True)

                if 'query2' in question:
                    answer_preambles = query_output
                    answer_preambles = [a + question['preamble'] for a in answer_preambles]
                    full_query = add_schema(question['query'] + question['query2'])
                    cot_output, sample = model.query(image, [full_query], answer_preamble=answer_preambles, max_new_tokens=2000, output_only=True, return_samples=True)
                    query_output = [a + o for (a, o) in zip(answer_preambles, cot_output)]

                # Save response
                with open(os.path.join(image_dir, question['name'] + '-question.txt'), 'w') as text_file:
                    text_file.write(sample['Input'][0])

                with open(os.path.join(image_dir, question['name'] + '-response.txt'), 'w') as text_file:
                    text_file.write(query_output[0])

                # show_display_image(output, batch_images, image_ids, model_spec[2], results_dir, question_set)
        
        del model
        torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def closed_ended_prompt_testing(config):
    # Closed ended performance
    from run.closed_ended_evaluation import ClosedEndedEvaluator
    tabular_validation_dataset = RetinalTextDataset(config, set_='validation')
    evaluator = ClosedEndedEvaluator(config, tabular_validation_dataset, specific_tasks=config.dataset.task.validation_tasks)
    device = torch.device('cuda:0')

    df = pd.DataFrame()
    for model_spec in config.pretrained_models:
        print(f'Running {model_spec}')

        # Load model
        config.model.checkpoint_path = model_spec
        model = VLM(config.copy()).load(device=device)

        results = evaluator.run_tasks(model)
        results_df = pd.DataFrame(results)
        results_df['model_display_name'] = model_spec[3]

        df = pd.concat((df, results_df))

        del model
        torch.cuda.empty_cache()

    results_ixed = df.set_index(['task_type', 'task_name'])

    staging_results = results_ixed.loc[results_ixed.index[0][0]]
    for i, res in staging_results.iterrows():
        aggregated_staging_results = aggregate_predictions(pd.DataFrame(res).T)

        # results_ixed.loc['SpecialistOther'].loc[i][f'F1'] = aggregate_f1(aggregated_staging_results, positive_classes=aggregated_staging_results.iloc[0]['options'], n_bootstraps=False)
        f1 = aggregate_f1(aggregated_staging_results, positive_classes=aggregated_staging_results.iloc[0]['options'], n_bootstraps=False)
        print(i, np.round(100 * f1, 2))

        # Save images

    comparison_df = [pd.DataFrame({k: result[k] for k in ['labels', 'predictions', 'inputs', 'outputs']}) for result in results]

    # results_dir = os.path.join('evaluation/test_bed/', config['job_id'], results_ixed.index[0][0])
    # for i, (image_id) in enumerate(results[0]['ImageId']):
    #     image = tabular_validation_dataset._get_images(image_name=image_id)[0]
    #     image = image.cpu().squeeze().numpy()
    #     plt.imshow(image, cmap='gray', vmin=np.percentile(image, 0.5), vmax=np.percentile(image, 99.5))
    #     plt.axis('off')
    #     save_fig_path_creation(os.path.join(results_dir, 'images', str(i) + '-' + image_id.split('/')[-1]))

    x = 3

if __name__ == "__main__":
    open_ended_prompt_testing()
    # closed_ended_prompt_testing()