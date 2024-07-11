from models.biomedclip import BiomedCLIP
from models.retfound import RETFound
from models.pretrained_resnet import PretrainedResNet
from models.wizardlm import load_wizardlm
from models.llms import Vicuna
from models.llama3 import Llama3
from ast import literal_eval
import wandb
from types import SimpleNamespace

vision_models = {
    'biomedclip': lambda config: BiomedCLIP(config),
    'pretrained_resnet': lambda config: PretrainedResNet(config),
    'retfound': lambda config: RETFound(config)
}

language_models = {
    'WizardLM-70B-GPTQ': lambda config: load_wizardlm(config),
    'lmsys/vicuna-13b-v1.3': lambda config: Vicuna(config),
    'lmsys/vicuna-13b-v1.5': lambda config: Vicuna(config),
    'meta-llama/Meta-Llama-3-8B-Instruct': lambda config: Llama3(config),
}

def get_vision_model(config):
    return vision_models[config.model.vision_encoder.name](config)

def get_language_model(config):
    return language_models[config.model.language_model.model_id](config)

def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

def deep_update(d, u):
    return {k: deep_update(d.get(k, {}), v) if isinstance(v, dict) else v for k, v in {**d, **u}.items()}

def find_base_config(config, api):
    project, run_name = config['model']['checkpoint_path'][0].split("/")
    runs = api.runs(f'robbieholland/{project}')
    target_run = next((run for run in runs if run.name == run_name), None)
    if target_run is None:
        raise Exception(f"Run {run_name} not found.")
    run_config = {k: literal_eval(v) if isinstance(v, str) and v.startswith('{') and v.endswith('}') else v for k, v in (target_run.config.items() if target_run else {})}
    if run_config['model']['checkpoint_path'] is None:
        return run_config
    return find_base_config(run_config, api)

def get_run_config(config):
    api = wandb.Api()
    run_config = find_base_config(config, api)

    run_config = deep_update(config, run_config)
    run_config = dict_to_namespace(run_config)

    return run_config
