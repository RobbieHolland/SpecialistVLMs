from evaluation.tasks.specialist.base import CoT, Binary

class CoTDetection(Binary, CoT):
    def __init__(self, *args, **kwargs):
        biomarker = kwargs.pop('biomarker', '')
        binary_word = kwargs.pop('binary_word', '')
        article = kwargs.pop('article', 'is')
        super().__init__(*args, f"Tell me if the image 'does' or 'does not' {binary_word} {biomarker}?", answer_preamble='The image does', **kwargs)

        self.options = ['not present', 'present']
        self.biomarker = biomarker
        self.binary_word = binary_word

        if self.cot:
            self.cot_question = f'Describe the OCT image in detail and list all biomarkers or abnormalities. Detail if there are any signs indicating that {biomarker} might be present, even if there is only a small amount.'
            self.answer_preamble = f'To conclude these findings, in the OCT image {biomarker} {article}'
            self.actual_question = f'Finally, conclude your findings by telling me if {biomarker} {article} \"not present\", or if potentially any amount of {biomarker} {article} \"present\" in the OCT image.'
        else:
            self.form_question = self.form_binary_question

    def form_output(self, labels):
        return {'Y': self.options[1], 'N': self.options[0]}[labels]
    
    def form_statement(self, label):
        presence = {self.options[1]: 'does', self.options[0]: 'does not'}
        return f'There image {presence[label]} show {self.biomarker}'

class SpecialistDetection:
    class RPEAtrophy(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='show', biomarker='RPE atrophy or degeneration')
            self.variable = 'RPE state'

        def form_output(self, labels):
            return {'Atrophy': self.options[1], 'Degeneration': self.options[1], 'N': self.options[0]}[labels]
        
    class RPEElevation(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='show', biomarker='RPE elevation or irregularity')
            self.variable = 'RPE disruption/ elevation'

    class DoubleLayerSign(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='show', biomarker='RPE elevation or irregularity')
            self.variable = 'RPE disruption/ elevation'

    # Not optimised for grammar
    class PED(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='pigment epithelial detachment (PED)')
            self.variable = 'PED?'

    class SubretinalFluid(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='subretinal fluid')
            self.variable = 'Subretinal fluid?'

    class Hypertransmission(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='hypertransmission')
            self.variable = 'Hypertrans-mission?'

    class IntraretinalFluid(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='intraretinal fluid')
            self.variable = 'Intraretinal fluid?'

    class Drusen(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='drusen', article='are')
            self.variable = 'Drusen?'

    class SHRM(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='subretinal hyperreflective material (SHRM)')
            self.variable = 'Subretinal hyperreflective material (SHRM)?'

    class HyperreflectiveFoci(CoTDetection):
        def __init__(self, config):
            super().__init__(config, binary_word='contain', biomarker='hyperreflective foci', article='are')
            self.variable = 'Hyperreflective foci?'

import hydra
@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def debug(config):
    task = SpecialistDetection.Drusen(config)
    print('Finished.')

if __name__ == "__main__":
    debug()