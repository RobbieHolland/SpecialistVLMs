from evaluation.tasks.specialist.base import BiomarkerClassification, CoT
from dataset.text_util import list_or
import textwrap

class CoTAMDStage(BiomarkerClassification, CoT):
    def __init__(self, *args, **kwargs):
        self.display_options = ['\"Healthy\"', '\"Early\"', '\"Intermediate\"', '\"Late dry\"', '\"Late wet (inactive)\"', '\"Late wet (active)\"']

        super().__init__(*args, f'Does the image show {list_or([s.lower() for s in self.display_options], randomise=False)} AMD?', **kwargs)
        self.options = ['Healthy', 'Early', 'Intermediate', 'Late dry', 'Late wet (inactive)', 'Late wet (active)']
        self.variable = 'AMD stage'

        if self.cot:
            self.cot_question = f'Describe the OCT image in detail and list any biomarkers or abnormalities, including the most likely AMD stage of the patient.'
            self.actual_question = f'Then, based on those observations, tell me if the patient\'s most likely AMD stage is {list_or([s.lower() for s in self.display_options], randomise=False)}?'
            self.answer_preamble = f'Based off the image and those findings, the patient\'s most likely AMD stage is'
        else:
            self.form_question = self.form_multiclass_question

class SpecialistOther:
    class AMDStage(CoTAMDStage):
        def __init__(self, config):
            super().__init__(config)

    # class PathologyUnrelatedToAMD(CoTDetection):
    #     def __init__(self, config):
    #         super().__init__(config, binary_word='contain', biomarker='any pathology unrelated to AMD')
    #         self.variable = 'Pathology unrelated to AMD?'
    
    # class FoveaNotShownInImage(CoTDetection):
    #     def __init__(self, config):
    #         super().__init__(config, binary_word='contain', biomarker='the fovea')
    #         self.variable = 'Fovea not shown in image?'

    #     def process_default(self, label):
    #         if label == 'Y':
    #             return 'N'
    #         return 'Y'
