from models.mini_gpt4 import MiniGPT4
import torch
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader
import sys
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import random
import numpy as np
from models.get_model import get_run_config
import pandas as pd
import types
import omegaconf
# import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn', force=True)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "0"

from pytorch_lightning import Trainer

import hydra

import pytorch_lightning as pl
import torch
import torchmetrics
import glob

class TrainableSave(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def find_checkpoint(self, checkpoint_code=None):
        checkpoint_code = self.config.model.checkpoint_path if checkpoint_code is None else checkpoint_code
        glob_string = f'{self.config.pretrained_model_dir}/{checkpoint_code[0]}/*{checkpoint_code[1]}*'
        checkpoints = glob.glob(glob_string)
        if len(checkpoints) == 0:
            raise Exception(f'No model checkpoints in {glob_string}')
        return checkpoints[0]

    def load_from_checkpoint_file(self, checkpoint_path=None):
        checkpoint_path = self.find_checkpoint() if checkpoint_path is None else checkpoint_path
        print(f'Loading pretrained model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)

        # Get the names of all parameters that require gradients
        grad_param_keys = {name for name, param in self.named_parameters() if param.requires_grad}

        # Check that each grad parameter has a corresponding key in the loaded state_dict
        loaded_keys = set(checkpoint['state_dict'].keys())
        missing_keys = grad_param_keys - loaded_keys
        if missing_keys:
            print(f"Warning: Missing keys in loaded state_dict: {missing_keys}")
        extra_keys = loaded_keys - grad_param_keys
        if extra_keys:
            print(f"Warning: Extra keys in loaded state_dict: {extra_keys}")
        
        self.load_state_dict(checkpoint['state_dict'], strict=False)

    def on_save_checkpoint(self, checkpoint: dict):
        grad_params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        checkpoint['state_dict'] = {k: v for k, v in self.state_dict().items() if k in grad_params}

class MiniGPT4Module(TrainableSave):
    def __init__(self, config, closed_ended_evaluator=None, device=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.closed_ended_evaluator = closed_ended_evaluator
        self.previous_closed_ended_performance = 0
        self.validation_epochs = 0
        self.tokenizers = {}

        if config.model.checkpoint_path is not None:
            model_config = config.copy()
            run_config = get_run_config(config)

            model_config.model.update(omegaconf.OmegaConf.create({k: (lambda x: {j: (lambda y: {m: n if not isinstance(n, types.SimpleNamespace) else vars(n) for m, n in vars(y).items()})(m) if isinstance(m, types.SimpleNamespace) else m for j, m in vars(x).items()})(v) if isinstance(v, types.SimpleNamespace) else v for k, v in vars(run_config.model).items()}))
            model_config.model.checkpoint_path = config.model.checkpoint_path
            
            self.model = MiniGPT4(model_config, device=device)
            self.load_from_checkpoint_file()
        else:
            self.model = MiniGPT4(config, device=device)
            print('Using an entirely new MiniGPT4 adapter.')

        self.loss_metrics = {
            'train': torchmetrics.MeanMetric(),
            'val': torchmetrics.MeanMetric(),
            'test': torchmetrics.MeanMetric(),
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx, prefix):
        output = self.model(batch)#, rank=self.global_rank)
        loss = output  # Assuming 'loss' is a field in the output

        self.loss_metrics[prefix].cpu().update(loss.detach().cpu())
        return output

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        self.log(f"train/batch_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     instruction = 'Create a radiological report for this OCT image.'
        #     sample_output = self.model.query(batch[0][0], {'Question': instruction})
        #     print('Instruction', instruction)
        #     print('Sample output', sample_output)
        return self.shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test')

    def on_train_epoch_end(self):
        self.log("train/epoch_loss", self.loss_metrics['train'].to(self.model.llm.device).compute())
        self.loss_metrics['train'].to(self.model.llm.device).reset()

    def on_validation_epoch_end(self):
        epoch_loss = self.loss_metrics['val'].to(self.model.llm.device).compute()
        self.log("val/epoch_loss", epoch_loss)
        self.log("val_epoch_loss", epoch_loss)
        self.loss_metrics['val'].to(self.model.llm.device).reset()

        if (self.closed_ended_evaluator is not None) and (self.validation_epochs % self.config.dataset.task.closed_ended_every_n_epoch == 0):
            from run.closed_ended_figures import aggregate_predictions, aggregate_f1

            closed_ended_results = self.closed_ended_evaluator.run_tasks(self.model)
            results = pd.DataFrame(closed_ended_results)
            results['model_display_name'] = 'Training now'

            results_ixed = results.set_index(['CoT_multi', 'task_type', 'task_name', 'model_display_name'])
            
            biomarker_results = results_ixed.loc[self.config.dataset.task.cot].loc['SpecialistDetection']
            aggregated_biomarker_results = aggregate_predictions(biomarker_results)
            biomarker_f1 = aggregate_f1(aggregated_biomarker_results, positive_classes=['present'], n_bootstraps=False)
            biomarker_f1 = biomarker_f1.loc['Training now']
            self.log(f"val_biomarker_f1", biomarker_f1)

            staging_results = results_ixed.loc[self.config.dataset.task.cot].loc['SpecialistOther']
            aggregated_staging_results = aggregate_predictions(staging_results)
            staging_f1 = aggregate_f1(aggregated_staging_results, positive_classes=aggregated_staging_results.iloc[0]['options'], n_bootstraps=False)
            staging_f1 = staging_f1.loc['Training now']
            self.log(f"val_staging_f1", staging_f1)

            self.previous_closed_ended_performance = (biomarker_f1 + staging_f1) / 2

        self.log("val_average_closed_ended", self.previous_closed_ended_performance)
        
        self.previous_closed_ended_performance -= 1e-8
        self.validation_epochs += 1

    def on_test_epoch_end(self):
        epoch_loss = self.loss_metrics['test'].to(self.model.llm.device).compute()
        self.log("test/epoch_loss", epoch_loss)
        self.loss_metrics['test'].to(self.model.llm.device).reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.dataset.task.learning_rate)

def load_pretrained_vlm_module(config):
    run_config = get_run_config(config)
    run_config.model.checkpoint_path = config.model.checkpoint_path
    minigpt4 = MiniGPT4(run_config)
    module = MiniGPT4Module(run_config, minigpt4)
    return module

def load_pretrained_vlm(config):
    model = load_pretrained_vlm_module(config).model.eval()
    return model

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def train(config):
    import dill
    torch.multiprocessing.get_context().set_forkserver_preload(['dill'])

    wandb.init(project=config.wandb_project, config=dict(config), mode=config.wandb_mode, settings=wandb.Settings(_service_wait=300))
    sys.path.append(config['octlatent_dir'])
    # Split my module across 4 gpus, one layer each
    from dataset.retinal_text_dataset import RetinalLLMDataset

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load or create model
    device = torch.device('cuda:0')
    # device = None
    if not config.model.language_model.load_in_8bit:
        device = torch.device('cpu')
    model = MiniGPT4Module(config, device=device)

    # Load datasets
    datasets, data_loaders = {}, {}
    for set_ in ['train', 'validation', 'test']:
        datasets[set_] = RetinalLLMDataset(config.copy(), set_=set_, deterministic=set_ in ['validation', 'test'], model_config=model.model.config)
        data_loaders[set_] = DataLoader(datasets[set_], batch_size=config.model.batch_size, shuffle=False, collate_fn=datasets[set_].create_custom_collate(), 
                                persistent_workers=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)

    # Standard callbacks
    model_save = f"{config['pretrained_model_dir']}/{config['wandb_project']}/{wandb.run.name}"
    min_val_loss_checkpoint = ModelCheckpoint(dirpath=model_save, filename='best_step={step}-{val_epoch_loss:.2f}', monitor='val_epoch_loss', mode='min', save_weights_only=True)
    last_checkpoint = ModelCheckpoint(dirpath=model_save, filename='last_step={step}-{val_epoch_loss:.2f}', save_last=True, save_weights_only=True)
    every_10k_steps_checkpoint = ModelCheckpoint(dirpath=model_save, filename='step={step}-{val_epoch_loss:.2f}', every_n_train_steps=10000, save_weights_only=True)
    callbacks = [min_val_loss_checkpoint, last_checkpoint, every_10k_steps_checkpoint]

    # Closed ended performance
    from run.closed_ended_evaluation import ClosedEndedEvaluator
    evaluator = None
    if len(config.dataset.task.validation_tasks) > 0:
        tabular_config = config.copy()
        tabular_config.dataset.task.target = 'TabularAnnotated'
        tabular_config.dataset.task.llm_qs_as = []
        tabular_validation_dataset = RetinalLLMDataset(tabular_config, set_='validation', deterministic=True, model_config=model.model.config)
        evaluator = ClosedEndedEvaluator(config, tabular_validation_dataset, specific_tasks=config.dataset.task.validation_tasks)

        for metric in ['val_average_closed_ended', 'val_biomarker_f1', 'val_staging_f1']:
            max_closed_ended_checkpoint = ModelCheckpoint(dirpath=model_save, filename=f'best_{metric}=' + '{step}-{' + metric + ':.2f}', monitor='val_average_closed_ended', mode='max', save_weights_only=True)
            callbacks.append(max_closed_ended_checkpoint)
    model.closed_ended_evaluator = evaluator

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
    # from torch.distributed.fsdp.wrap import default_auto_wrap_policy

    # model = DistributedDataParallel(model)
    # model = FullyShardedDataParallel(
    #     model,
    #     # auto_wrap_policy=default_auto_wrap_policy,
    #     cpu_offload=CPUOffload(offload_params=True),
    # )

    # Assume fsdp_model is your model wrapped with FSDP
    # trainer = Trainer(max_steps=10, accelerator='ddp', gpus=1)
        
    # Test load images
    # batch = next(iter(data_loaders["train"]))
    # print('Batch images', batch['Image'].shape)

    trainer = Trainer(
        accelerator="gpu",
        devices=config.devices, 
        strategy="dp",   # Setting strategy to Data Parallel
        precision=16,
        auto_select_gpus = True,

        num_sanity_val_steps=0,
        max_steps=config.dataset.task.max_steps, 
        log_every_n_steps=1,
        callbacks=callbacks,
        val_check_interval=config.dataset.task.val_check_interval,#min(config.dataset.task.val_check_interval, len(data_loaders['train'])),
        check_val_every_n_epoch=config.dataset.task.check_val_every_n_epoch,
        limit_val_batches=config.dataset.task.limit_val_batches,
        logger=WandbLogger())
        
    if config.dataset.task.validate_first:
        with torch.no_grad():
            model.eval()
            trainer.validate(model, dataloaders=data_loaders['validation'])

    model = MiniGPT4Module(config, device=device)
    model.closed_ended_evaluator = evaluator
    model.train()
    trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'])

if __name__ == "__main__":
    train()