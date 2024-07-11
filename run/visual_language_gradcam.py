
import hydra
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
from evaluation.figure_util import save_fig_path_creation
from models.vlm import VLM
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from clinical_capabilities.clinical_capabilities_util import add_schema

class SpatialArrange(nn.Module):
    def __init__(self):
        super(SpatialArrange, self).__init__()

    def forward(self, x):
        # Slicing
        x = x[:, 1:, :]
        # Reshaping
        x = x.reshape(-1, 1024, 14, 14)
        # Transposing from [batch_size, channels, height, width] to [batch_size, height, width, channels]
        # for hypothetical processing reasons or to match a specific data format requirement
        x = x.permute(0, 2, 3, 1)
        return x

def plot_and_save_image(ax, masked_image, mask_1, mask_2, title, save_path=None):
    ax.imshow(masked_image, cmap='gray', vmin=np.percentile(masked_image, 0.5), vmax=np.percentile(masked_image, 99.5))
    ax.contour(np.where(mask_2 >= 0.85, 1, 0), colors='red', linewidths=1.5)
    combined_mask = np.where(mask_2 >= 0.85, 2)
    ax.contourf(combined_mask, levels=[0, 0.5, 1.0], colors=['none', 'red'], alpha=0.2)

    ax.axis('off')
    
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def find_subsequences(k_ids, tokens):
    return [i for i in range(len(tokens) - len(k_ids) + 1) if tokens[i:i+len(k_ids)].tolist() == k_ids.tolist()]

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def generative_gradcam(config):
    import random
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    from slurm.util import record_job_id
    config = record_job_id(config)

    sys.path.append(config['flamingo_dir'])
    sys.path.append(config['octlatent_dir'])
    from dataset.retinal_text_dataset import RetinalTextDataset

    device = torch.device('cuda:0')
    dataset = RetinalTextDataset(config, set_='validation')
    dataset.data_csv = dataset.data_csv[dataset.data_csv['TabularAnnotated'] == True]
    dataset.data_csv = dataset.data_csv.sample(frac=1, random_state=config.seed+1).reset_index(drop=True)

    data_loader = DataLoader(dataset, batch_size=config.model.batch_size, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                        persistent_workers=False, pin_memory=False, num_workers=6, drop_last=False)
    images, targets = next(iter(data_loader))

    images = images.half().to(device)

    # Iterate through pretrained models
    for model_spec in config.pretrained_models:
        results_dir = os.path.join(config.figure_path, 'visual_language_attentions', config['job_id'], model_spec[3])

        print(f'Running {model_spec}')

        # Load model
        config.model.checkpoint_path = model_spec
        model = VLM(config.copy()).load(device=device)

        folder_path = f'visual_language_attentions/'
        output = {}

        for name, param in model.visual_encoder.named_parameters():
            param.requires_grad = True

        if model.config.model.vision_encoder.name == 'retfound':
            layers = [model.visual_encoder.model.blocks[-2].norm1]
        else:
            layers = [model.visual_encoder.model.layer3[-1], model.visual_encoder.model.layer4[-1]]
        # layers = [model.visual_encoder.model.layer4[-1]]

        cam = GradCAM(model=model, target_layers=layers)

        phrases = config.dataset.task.phrases
        phrase_tokens = {
            k: model.llama_tokenizer.encode(k, add_special_tokens=False)
            for k in phrases
        }

        for question_set, questions in config.dataset.task.all_questions.items():
            print(f'On questions {question_set}')
            for question in questions:
                question_dir = f"{results_dir}/{question['name']}"

                # Iterate over each image
                for i, image in tqdm(enumerate(images)):
                    output_dir = os.path.join(question_dir, f"image_{i}")
                    os.makedirs(output_dir, exist_ok=True)

                    image = image.unsqueeze(0)  # Prepare image for processing

                    # Query the model for each image
                    questions = [add_schema(question['query'])] * len(image)
                    outputs, samples = model.query(image, questions, answer_preamble=None, max_new_tokens=config.dataset.task.max_new_tokens, output_only=True, return_samples=True)

                    with open(os.path.join(output_dir, 'report.txt'), 'w') as text_file:
                        text_file.write(outputs[0])

                    image_cpu = image.cpu()
                    # Save the original image
                    fig = plt.figure().add_subplot(111)
                    plt.imshow(image_cpu.squeeze().numpy(), cmap='gray', vmin=np.percentile(image_cpu.squeeze().numpy(), 0.5), vmax=np.percentile(image_cpu.squeeze().numpy(), 99.5))
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, 'original.png'), bbox_inches='tight', pad_inches=0)
                    plt.close('all')

                    # Create a new figure for each image
                    ncols = len(phrase_tokens) + 1
                    fig, axs = plt.subplots(1, ncols, figsize=(2.5 * ncols, 2.5))  # Single row, multiple columns

                    # Display the original image in the first subplot
                    axs[0].imshow(image_cpu.squeeze(), cmap='gray', vmin=np.percentile(image_cpu, 0.5), vmax=np.percentile(image_cpu, 99.5))
                    axs[0].axis('off')
                    axs[0].set_title('Original Image')

                    # Iterate over each phrase and token
                    for j, (phrase, class_tokens) in enumerate(phrase_tokens.items()):
                        
                        targets = [ClassifierOutputTarget(0)]

                        keyword_positions = find_subsequences(np.array(class_tokens), samples['Output tokens'][0].cpu())
                        if len(keyword_positions) == 0:
                            continue

                        output_position = keyword_positions[0] - samples['Output tokens'].shape[1]
                        class_tokens_positions = np.arange(output_position, output_position + len(class_tokens))

                        model.forward = lambda x: model.softmax_logits(x, texts=questions, answer_preambles=outputs, class_tokens_positions=class_tokens_positions, class_tokens=class_tokens)

                        grayscale_cam = cam(input_tensor=image, targets=targets, eigen_smooth=True, aug_smooth=True)

                        masked_image = image_cpu.squeeze().clone()
                        mask_1 = grayscale_cam.squeeze() > 0.25
                        mask_2 = grayscale_cam.squeeze() > 0.5

                        # Plot in the subplot
                        plot_and_save_image(axs[j + 1], masked_image, mask_1, mask_2, phrase)

                        # Save the masked image with contours
                        plot_and_save_image(plt.figure().add_subplot(111), masked_image, mask_1, mask_2, phrase, os.path.join(output_dir, f'{phrase}.png'))

                    # Set the main title of the figure to the 'outputs'
                    fig.suptitle(f"Report: {outputs[0]}")

                    # Save the figure for each image
                    save_fig_path_creation(os.path.join(output_dir, f"composite.png"))

                    plt.close(fig)  # Close the figure to free memory
                    
        del model
        torch.cuda.empty_cache()  # Clear CUDA memory cache to prevent memory issues

if __name__ == "__main__":
    generative_gradcam()