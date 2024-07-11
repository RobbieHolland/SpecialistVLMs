import pytorch_lightning as pl
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T

class PubmedCLIP(pl.LightningModule):
    def __init__(self, config):
        super(PubmedCLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32", cache_dir=config['pretrained_model_dir'])
        self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32", cache_dir=config['pretrained_model_dir'])

    def forward(self, images, texts):
        inputs = self.processor(text=texts, images=[T.ToPILImage()(image) for image in images], return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.model.device)
        inputs['input_ids'] = inputs['input_ids'].to(self.model.device)
        logits = self.model(**inputs).logits_per_image.softmax(dim=1)
        return logits
    