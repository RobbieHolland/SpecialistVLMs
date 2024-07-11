from models.med_flamingo import MedFlamingo
from models.llava_med import LLavaMed
from run.vision_language_pretraining import MiniGPT4Module
from models.biomedclip import BiomedCLIP

class VLM():
    def __init__(self, config):
        self.config = config

    def load(self, device=None):
        if self.config.model.checkpoint_path is not None:
            if self.config.model.checkpoint_path[0]=='Med-Flamingo':
                return MedFlamingo(self.config, device=device).eval()
            elif self.config.model.checkpoint_path[0]=='LLaVA-Med':
                return LLavaMed(self.config, device=device).eval()
            elif self.config.model.checkpoint_path[0]=='BioMedCLIP':
                return BiomedCLIP(self.config, device=device).eval()

        print(f'-----> Loading:   {self.config.model.checkpoint_path[-1]}')
        model = MiniGPT4Module(self.config, device=device).model.eval()
        return model