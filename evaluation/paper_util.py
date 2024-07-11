display_names = {
    'Drusen': 'Drusen',
    'PED': 'Pigment\nEpithelial\nDetachment',
    'SubretinalFluid': 'Subretinal\nFluid',
    'IntraretinalFluid': 'Intraretinal\nFluid',
    'Hypertransmission': 'Hypertransmission',
    'RPEElevation': 'RPE\nElevation',
    'Fibrosis': 'Fibrosis',
    'SHRM': 'Subretinal\nHyperreflective\nMaterial',
    'HyperreflectiveFoci': 'Hyperreflective\nFoci',
    'DrusenSize': 'Drusen\nSize',
    'DrusenNumber': 'Drusen\nNumber',
    'PEDSize': 'PED\nSize',
    'SubretinalFluidVolume': 'Subretinal\nFluid\nVolume',
    'IntraretinalFluidVolume': 'Intraretinal\nFluid\nVolume',
    'HypertransmissionSeverity': 'Hypertransmission\nSeverity',
    'PEDType': 'PED\nType',
    'DrusenConfluent': 'Drusen\nConfluent',
    'HypertransmissionType': 'Hypertransmission\nType',
    'RPEState': 'RPE\nState',
    'PathologyUnrelatedToAMD': 'Pathology\nUnrelated to AMD',
    'FoveaNotShownInImage': 'Fovea\nNot Shown\nin Image',
    'ImageQuality': 'Image\nQuality',
    'AMDStage': 'AMD\nStage',
}

curriculum_names = {
    'WizardLM-70B-GPTQ_response': 'Trainee', 
    'advanced_biomarkers_guidelines': 'Advanced biomarkers',
    'general_qa': 'General QA', 
    'specific_qa': 'Specific QA', 
    'staging_introduction_guidelines': 'Staging introduction',
    'staging_logic_guidelines': 'Disease staging guidelines',
    'referral_reasoning_guidelines': 'Patient referral reasoning',
    'report_writing': 'Report writing',
    'complex_qa_guidelines': 'Complex QA',
}

curriculum_colors = {
    'Trainee': '#706F6F', 
    'Advanced biomarkers': '#FF499E',
    'General QA': '#706F6F', 
    'Specific QA': '#FFB10A', 
    'Staging introduction': '#363635',
    'Disease staging guidelines': '#B1C530',
    'Patient referral reasoning': '#FE343B',
    'Report writing': '#752ACB',
    'Complex QA': '#5E976E',
}

def get_font_color(background_color):
    """Determine whether the font color should be white or black based on the luminance of the background color."""
    r, g, b = background_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 140 else None