import numpy as np
from dataset.text_util import list_and, valid_variable

class SilverBiomarker():
    def __init__(self, config):
        self.config = config
        self.name = 'SilverBiomarker'
        self.variable = 'SilverBiomarkerTags'
        self.options = ['large subretinal fluid', 'deep foveal pit', 'medium drusen', 'subretinal fluid', 'PED', 'no fovea', 'macular scar', 'subretinal hyperreflective material', 'fibrovascular PED', 'small drusen', 'RPE elevation', 'no AMD', 'cRORA', 'poor image contrast', 'disciform scar', 'small intraretinal fluid', 'intraretinal fluid', 'large drusen', 'iRORA', 'tiny drusen', 'hypertransmission', 'small RPE elevation', 'optical disc', 'drusenoid PED', 'grainy image quality', 'poor image quality', 'double-layer sign', 'vitreomacular interface abnormalities']
        self.label_mapping_dict = {option: option for option in self.options}
        self.label_mapping_dict.update({'fibrovascular PED': 'fibrovascular pigment epithelial detachment', 'drusenoid PED': 'drusenoid pigment epithelial detachment', 'PED': 'pigment epithelial detachment', 'no AMD': 'no visible AMD biomarkers'})
        self.num_classes = len(self.options)

    def form_statement(self, label):
        return f'The AMD biomarkers present in this image are {list_and(label)}'
    
    def form_few_shot_example(self, label):
        return f'Here is an encoding of an image. The encoding indicates that this image shows the biomarker(s) {list_and(label)}.'

class TabularToPrompt():
    def __init__(self, config, data_csv):
        self.config = config
        self.data_csv = data_csv
        self.va_quartile_values = [np.percentile(self.data_csv.loc[~np.isnan(self.data_csv['VALogMAR']), 'VALogMAR'], q * 25) for q in range(1, 5)]
        self.qi_quartile_values = [np.percentile(self.data_csv.loc[~np.isnan(self.data_csv['QualityIndex']), 'QualityIndex'], q * 25) for q in range(1, 5)]
        self.all_biomarkers = sum([b for b in self.data_csv['SilverBiomarkerTags'] if valid_variable(b)], [])
        self.all_biomarker_counts = np.unique(self.all_biomarkers, return_counts=True)
        
    def sex_to_str(self, sex):
        return 'female' if sex == 0 else 'male'
    
    def quartile_to_ordinal(self, variable, quartiles, mapping = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}):
        if variable is None or variable is np.nan:
            return None

        quartile_number = sum(variable > quartile_value for quartile_value in quartiles) + 1
        return mapping.get(quartile_number, 'unknown')

    def natural_language_equivalent(self, AMDStageGroup):
        if AMDStageGroup is None or AMDStageGroup is np.nan or AMDStageGroup == 'NoDiagnosis':
            return None
        return AMDStageGroup.replace('EarlyIntermediate', 'early intermediate AMD').replace('LateDry', 'late dry AMD').replace('LateWet', 'late wet AMD').replace('Healthy', 'healthy eyes')

    def map_valogmar_to_letter_score(self, valogmar):
        if not valid_variable(valogmar):
            return valogmar
        return int(max(0, ((valogmar - 1.6) / (-0.2 - 1.6)) * (95 - 5) + 5))
    
    def generate_variables(self, row):
        sex = self.sex_to_str(row['Sex'])
        age = int(row['CurrentAge'])
        eye_position = {0: 'left', 1: 'right'}[row['EyePosition']]
        letter_score = self.map_valogmar_to_letter_score(row['VALogMAR'])
        quality_index = self.quartile_to_ordinal(row['QualityIndex'], self.qi_quartile_values, mapping={1: 'very poor', 2: 'ok', 3: 'good', 4: 'excellent'})
        amd_stage = self.natural_language_equivalent(row['AMDStageGroup'])
        silver_biomarkers = row['SilverBiomarkerTags'] if type(row['SilverBiomarkerTags']) is list else []
        sb = SilverBiomarker(self.config)

        silver_biomarkers = [sb.label_mapping_dict[b] for b in silver_biomarkers]
        all_biomarkers = [sb.label_mapping_dict[b] for b in sb.options]

        silver_biomarkers_str = list_and(silver_biomarkers)

        all_biomarker_count_strings = [sb.label_mapping_dict[b] for b in self.all_biomarker_counts[0]]
        biomarker_count_strings = list(set(all_biomarker_count_strings) - set(['no fovea', 'optical disc', 'poor image contrast', 'poor image quality', 'no visible AMD biomarkers', 'grainy image quality']))
        filtered_biomarkers = [sb for sb in biomarker_count_strings if sb not in silver_biomarkers and not any(word in sb.split() for word in {w for s in silver_biomarkers for w in s.split()})]
        weights = np.array([(self.all_biomarker_counts[1]/(self.all_biomarker_counts[1].sum()))[all_biomarker_count_strings.index(b)] for b in filtered_biomarkers])
        weights /= weights.sum()
        not_present_biomarkers = list_and(np.random.choice(filtered_biomarkers, size=3, replace=False, p=weights))

        # variables = np.array([age, sex, amd_stage, letter_score, eye_position, quality_index, silver_biomarkers])
        variables = np.array([silver_biomarkers_str, not_present_biomarkers, amd_stage, letter_score, age, sex, quality_index, eye_position])
        return variables

