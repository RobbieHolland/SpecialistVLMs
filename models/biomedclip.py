import pytorch_lightning as pl
import open_clip
import torch
import torchvision.transforms as T

class BiomedCLIP(pl.LightningModule):
    def __init__(self, config, device):
        super(BiomedCLIP, self).__init__()
        
        model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=config['pretrained_model_dir'])
        self.model = model.to(device)
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.num_features = 768
        self.is_generative = False

        self.feature_tokens_model = torch.nn.Sequential(*list(self.model.visual.trunk.children()))

    def process_images(self, images):
        return torch.stack([self.preprocess(T.ToPILImage()(image)) for image in images]).to(images.device)

    def process_images_and_text(self, images, texts):
        images = self.process_images(images)
        texts = self.tokenizer(texts, context_length=256).to(images.device)
        return images, texts
    
    def forward(self, images, texts):
        images, texts = self.process_images_and_text(images, texts)
        image_features, text_features, logit_scale = self.model(images, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        return logits
    
    def embed_image(self, images):
        images = self.process_images(images)
        return self.model.visual(images)
    
    def feature_tokens(self, images):
        images = self.process_images(images)
        return self.feature_tokens_model(images)
    