from dataset.text_util import list_and, valid_variable
from torchmetrics.classification import MulticlassAccuracy
from dataset.text_util import valid_variable

class ClosedEndedQuestion():
    def __init__(self, config):
        self.config = config
        self.variable = 'Undefined'
        self.invalid_response = 'Invalid response'
        self.reset()
        
        self.cot = False
        if self.__class__.__mro__[self.__class__.__mro__.index(ClosedEndedQuestion)+1] is not object:
            super().__init__(config) 
        else:
            super().__init__()

    @property
    def name(self):
        return self.variable
    
    @property
    def full_name(self):
        return self.__class__.__qualname__.split('.')

    def process_default(self, label):
        if not valid_variable(label):
            return None
        return label 
    
    def reset(self):
        self.image_ids = []
        self.labels = []
        self.predictions = []
        self.questions = []
        self.inputs = []
        self.outputs = []

class BiomarkerClassification(ClosedEndedQuestion):
    def __init__(self, config, question_phrase, answer_preamble='', options=['Yes', 'No']):
        super().__init__(config)
        self.question_phrase = question_phrase
        self.answer_preamble = answer_preamble
        self.options = options
        self.num_classes = len(self.options)

    def form_output(self, labels):
        return labels
    
    def form_multiclass_question(self):
        return {'Question': self.question_phrase,
                'Answer': self.answer_preamble,
                'Input': None}

class Binary(BiomarkerClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.form_binary_question = self.form_multiclass_question

    def process_default(self, label):
        if not valid_variable(label):
            return 'N'
        return label 
    
class CoT():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cot = dict(args[0])['dataset']['task']['cot']

    def form_cot_question(self, cot_answer):
        return {'Question': self.cot_question + ' ' + self.actual_question,
                'Answer': cot_answer + ' ' + self.answer_preamble,
                'Input': None}

    def form_question(self):
        return {'Question': self.cot_question,
                'Answer': '',
                'Input': None}
    