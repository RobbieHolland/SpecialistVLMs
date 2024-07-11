import yaml

guildines = {
    'observations': yaml.load(open(f"clinical_capabilities/observations.yaml", "r"), Loader=yaml.FullLoader),
    'staging': yaml.load(open(f"clinical_capabilities/disease_staging.yaml", "r"), Loader=yaml.FullLoader),
    'referral': yaml.load(open(f"clinical_capabilities/referral_guidelines.yaml", "r"), Loader=yaml.FullLoader)
}

def add_schema(prompt):       
    observations_schema = guildines['observations']['schema']
    disease_staging_schema = guildines['staging']['schema']
    referral_guidelines_schema = guildines['referral']['schema']

    prompt = prompt.replace('<ObservationGuidelines>', observations_schema)
    prompt = prompt.replace('<DiseaseStagingGuidelines>', disease_staging_schema)
    prompt = prompt.replace('<PatientReferralGuidelines>', referral_guidelines_schema)

    return prompt