def populate_prompt(config, sample):
    return config.model.language_model.prompt.question.replace('<Question>', sample['Question']).replace('<Answer>', sample['Answer'])
