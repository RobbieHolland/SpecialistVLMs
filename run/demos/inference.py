import hydra
from PIL import Image
import numpy as np
import textwrap
from glob import glob
import os
import sys

sys.path.append(os.getcwd())
from models.retinavlm_wrapper import load_from_api

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def demo_inference(config):
    sample_image_dir = 'dataset/processed_images'
    png_files = glob(os.path.join(sample_image_dir, '*.png'))

    instruction = textwrap.dedent(f'''Write an extensive report describing the OCT image and listing any visible biomarkers or other observations. Do not provide disease stage or patient referral recommendations yet.''')
    print(f'Running textual instruction: \"{instruction}\"')

    retinavlm = load_from_api(config).eval()

    for image_path in png_files:
        image = np.array(Image.open(image_path))
        print(f'On image: {image_path}')
        
        output = retinavlm.forward(image, [instruction])
        print(f'VLM response: \"{output[0]}\"')

if __name__ == "__main__":
    demo_inference()
