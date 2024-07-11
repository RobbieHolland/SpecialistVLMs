import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError, AUROC
from models.get_model import get_vision_model
from run.vision_language_pretraining import TrainableSave

import torch.nn as nn
from models.projection_heads import LinearClassifier
import hydra
from pytorch_lightning import LightningModule
import torchmetrics

class DownstreamFit(TrainableSave):
    def __init__(self, config, pretrained_encoder, linear_eval=True, regression=False):
        LightningModule.__init__(self)
        device = torch.device('cuda:0')

        self.config = config
        self.encoder = pretrained_encoder
        if linear_eval:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            self.encoder = self.encoder.eval()

        self.projector = LinearClassifier(config.model.vision_encoder.projection_dim, use_softmax=False, use_sigmoid=False, use_tanh=False, num_classes=config.dataset.task.num_outs)

        self.loss_metrics = {phase: torchmetrics.MeanMetric().to(device) for phase in ['train', 'val', 'test']}
        self.metrics = None

        if config.dataset.task.regression:
            self.loss_fn = nn.MSELoss()
            self.metrics = {phase: {'mae': torchmetrics.MeanAbsoluteError().to(device)} for phase in ['train', 'val', 'test']}
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.metrics = {phase: {'auc': AUROC(task="multiclass", num_classes=2).to(device)} for phase in ['train', 'val', 'test']}

        self.save_hyperparameters()

    def update_metrics(self, phase, loss, y_hat, y):
        self.loss_metrics[phase].update(loss)
        if self.metrics is not None:
            for _, metric in self.metrics[phase].items():
                metric.update(y_hat, y)

    def epoch_metrics(self, phase):
        epoch_loss = self.loss_metrics[phase].compute()
        self.log(f"{phase}/epoch_loss", epoch_loss)
        self.loss_metrics[phase].reset()
        if self.metrics is not None:
            for metric_name, metric in self.metrics[phase].items():
                self.log(f"{phase}/epoch_{metric_name}", metric.compute())
                metric.reset()
        return epoch_loss

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.projector(input)

    def shared_step(self, batch, phase):
        xs, ys = batch
        ys = ys[0]
        ys = ys.view(ys.shape[0], -1).squeeze().half()

        zs = self.encoder.embed_image(xs)
        y_hat = self.projector(zs)

        if not self.config.dataset.task.regression:
            ys = ys.long()
        loss = self.loss_fn(y_hat.squeeze(), ys.squeeze())

        self.update_metrics(phase, loss, y_hat, ys)
        self.log(f"{phase}/batch_loss", loss)

        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        step_outputs = self.shared_step(batch, 'train')
        return step_outputs

    def validation_step(self, batch, batch_idx):
        step_outputs = self.shared_step(batch, 'val')
        return step_outputs

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def on_train_epoch_end(self):
        self.epoch_metrics('train')

    def on_validation_epoch_end(self):
        epoch_loss = self.epoch_metrics('val')
        self.log("val_epoch_loss", epoch_loss)

    def on_test_epoch_end(self):
        self.epoch_metrics('test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.dataset.task.lr)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from models.clip import CLIPModule
    import wandb
    wandb.init(project=config.wandb_project, config=dict(config), mode=config.wandb_mode)
    from dataset.retinal_text_dataset import RetinalTextDataset

    device = torch.device('cuda:0')

    # Load datasets
    datasets, data_loaders = {}, {}
    for set_ in ['train', 'validation', 'test']:
        datasets[set_] = RetinalTextDataset(config.copy(), set_=set_)
        data_loaders[set_] = DataLoader(datasets[set_], batch_size=config.model.batch_size, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                                persistent_workers=False, pin_memory=False, num_workers=config.num_workers, drop_last=True)

    batch = next(iter(data_loaders["train"]))
    print('Batch images', batch[0].shape)

    # Load vision encoder
    visual_encoder = get_vision_model(config).to(device)
    model = DownstreamFit(config, visual_encoder).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {trainable_params} parameters')

    # Finetune
    callbacks = []
    model_save = f"{config['pretrained_model_dir']}/{config['wandb_project']}/{wandb.run.name}"
    min_val_loss_checkpoint = ModelCheckpoint(dirpath=model_save, filename='best_step={step}-{val_epoch_loss:.2f}', monitor='val_epoch_loss', mode='min', save_weights_only=True)
    # last_checkpoint = ModelCheckpoint(dirpath=model_save, filename='last_step={step}-{val_epoch_loss:.2f}', save_last=True, save_weights_only=True)
    callbacks = [min_val_loss_checkpoint]
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_steps=config.dataset.task.max_steps, 
        accelerator="gpu",
        devices="auto", 
        precision=16,
        log_every_n_steps=1,
        callbacks=callbacks,
        val_check_interval=config.dataset.task.val_check_interval,
        limit_val_batches=config.dataset.task.limit_val_batches,
        logger=WandbLogger())
    
    trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'])
    model.load_from_checkpoint_file(model.find_checkpoint([f"{config['wandb_project']}/{wandb.run.name}", 'best_step']))
    trainer.test(model, dataloaders=data_loaders['test'])

if __name__ == "__main__":
    test()