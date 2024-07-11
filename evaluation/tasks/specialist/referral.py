from evaluation.tasks.specialist.base import BiomarkerClassification, CoT
import textwrap

class CoTSouthamptonReferral(BiomarkerClassification, CoT):
    def __init__(self, *args, **kwargs):
        options = ['not', 'routine referral', 'next two weeks']
        super().__init__(*args, f"Does the patient in the image \"urgent treatment\", or do they \"not need referral\" for treatment?", answer_preamble=f'The patient in the image does', options=options, **kwargs)

        if self.cot:
            self.actual_question = textwrap.dedent(f'''
                Being seen by a specialist at the Southampton clinic:
                A. The Southampton clinic requires that patients with any sign of intraretinal fluid, any sign of subretinal fluid, or any sign of cyst(s), MUST be seen by a specialist at the Southampton clinic within the next two weeks.
                B. The Southampton clinic requires that patients who do not have any sign of intraretinal fluid, any sign of subretinal fluid, or any sign of cyst(s), but do have some biomarkers of early or intermediate AMD, should be seen by a specialist at the Southampton clinic for routine referral.
                C. The Southampton clinic requires that patients who do not have any sign of intraretinal fluid, any sign of subretinal fluid, or any sign of cyst(s), but do have medium to large drusen, drusenoid PED, hypertransmission or atrophy, should be seen by a specialist at the Southampton clinic for routine referral.
                D. The Southampton clinic does not need to see patients who have no biomarkers and healthy retinas at all.
                
                Southampton specialist visit: Next, tell me if your initial report of the OCT image indicates that the patient should be seen by a specialist at the Southampton clinic within the next two weeks, for routine referral, or not seen at all?
            ''')

            self.cot_question = "Write an extensive report describing the OCT image and listing any present biomarkers or other observations. Do not provide a disease stage, or referral recommendation yet."

            self.answer_preamble = 'My report indicates that the patient'
        else:
            self.form_question = self.form_multiclass_question
        
        self.variable = 'Referral recommendation'

    def form_output(self, labels):
        return {'Likely needing treatment': self.options[2], 
                'General attention by specialist': self.options[1],
                'No referral': self.options[0]}[labels]

class SpecialistReferral:
    class SouthamptonReferral(CoTSouthamptonReferral):
        def __init__(self, config):
            super().__init__(config)
