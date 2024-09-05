import hydra
import os
from tqdm import tqdm
from models.chatgpt import ChatGPT
from dataset.text_util import parse_qa

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def generate_curriculum(config):
    print(f'Running: {config.dataset.task.curriculum.output_column_name}')
    
    # Initialize column 'a' with None or some default value
    chatgpt = ChatGPT(config.model.language_model.openai_api_key)

    sample_annotations = [
        'In this OCT image there is a large irregular RPE elevation in the centre of the image with a heterogenously reflective core. There are overlying intraretinal hyperreflective foci and a small amount of intraretinal fluid. There is no subretinal fluid.',
        'This OCT image shows a large druse with a hyporeflective core in the left side of the image. There is a small druse to the right of it. There are a small number of hyperreflective foci in the outer retinal layers. There is no evidence of intraretinal or subretinal fluid.',
        'This OCT image shows a large drusenoid PED in the left and centre of the image. There is a large volume of intraretinal fluid across the image. There is no obvious subretinal fluid and there are no intraretinal hyperreflective foci. There is mild signal hypertransmission to the choroid in the right of the image. This image is consistent with wet AMD.',
        'This OCT shows intraretinal fluid in the outer plexiform layer with an overlying lamellar macular hole. This may be due to released vitreomacularr traction, but the fluid may be related to the fibrovascular PED in the right of the image. There is patchy ellipsoid zone (EZ) loss and likely SDD. This is likely late wet AMD, but could also be cystoid macular edema (CMO) due to another pathology, and further investigation is warranted',
        'This OCT shows the right macula and fovea. There is patchy ellipsoid zone (EZ) loss including at the fovea, and in the left of the image a possible SDD. There is a shallow elevation of the RPE on the right of the image possibly consistent with drusen, but could also represent a double layer sign. There is no intraretinal fluid or subretinal fluid. There is overlying ellipsoid zone (EZ) loss but no clear RPE atrophy. This is intermediate AMD but at high risk of progression to late AMD',
        'This OCT shows the fovea of the right eye. There are likely several small drusen affecting the fovea and temporal retina. There is no sign of late AMD. This is likely early AMD but if there is pigmentary change could be considered intermediate',
    ]

    for i, annotation in tqdm(enumerate(sample_annotations)):
        print(f'------ Generating VQA for sample {i} with annotation:')
        print(annotation)
        chatgpt_input = config.dataset.task.curriculum.annotation_prompt.replace('<Variables>', annotation)

        reply = chatgpt.generate(chatgpt_input, temperature=config.model.language_model.temperature, endpoint=config.model.language_model.endpoint)
        vqa = parse_qa(reply)

        print('Resulting VQA')
        print(vqa)
        print()

if __name__ == "__main__":
    generate_curriculum()
