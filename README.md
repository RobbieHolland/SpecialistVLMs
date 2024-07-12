# SpecialistVLMs
Developing VLMs for expert-level performance in specific medical specialties.

Paper preprint [https://arxiv.org/abs/2407.08410](https://arxiv.org/abs/2407.08410)

RetinaVLM is a generative vision-language model designed to assist in the management of patients with age-related macular degeneration (AMD) from retinal OCT images.

![Figure 1](Figure_1.jpg)

## Creating Curriculum part 1 and part 2

To generate curriculum part 1

`run/generate_curriculum_part1.py dataset=retina model/language_model=gpt-4o dataset/task=tabular_annotate dataset/task/curriculum=tabular_annotate_rules`

To generate the modules in curriculum part 2: `1_advanced_biomarkers_guidelines`, `2_specific_qa`... `10_staging_accuracy`:

`run/generate_curriculum_part2.py dataset=retina model/language_model=gpt-4o dataset/task=specialist_annotate dataset/task/curriculum=1_advanced_biomarkers_guidelines`

## Running Vision-Language Pretraining (Curriculum part 1 and part 2)

Creating RetinaVLM by training on curriculum part 1, and then curriculum part 2.

`run/vision_language_pretraining.py dataset=retina pretrained_models=new dataset/task=curriculum_part_1_introduction_to_retina`

`run/vision_language_pretraining.py dataset=retina pretrained_models=retinavlm_base_192px dataset/task=curriculum_part_2_advanced_retinal_specialism`

## Testing VLMs

Testing VLMs on tasks `closed_ended_specialist_staging`, `closed_ended_specialist_referral` or `closed_ended_specialist_biomarkers`

`run/closed_ended_evaluation.py pretrained_models=specialist_v5_192px dataset=retina dataset/task=closed_ended_specialist_staging`

## Generating visual-language saliency maps

Code for computing visual saliency maps to passages in the RetinaVLM's generated reports

`run/visual_language_gradcam.py dataset=retina dataset/task=language_visual_attention pretrained_models=specialist_v5_192px`

#### Data availability

Both imaging datasets are currently being curated and maintained by the Vienna Reading Center on behalf of the PINNACLE consortium. The data will be made available once the PINNACLE study concludes in 2026.

#### Pretrained models

RetinaVLM-Base and RetinaVLM-Specialist models used in the paper are accessible at [https://huggingface.co/RobbieHolland/RetinaVLM](https://huggingface.co/RobbieHolland/RetinaVLM)

These versions of the model are not applicable for clinical use, as they were developed for research purposes

#### Dependencies

Code makes references to [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med) and [Med-Flamingo](https://github.com/snap-stanford/med-flamingo)
