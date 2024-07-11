import pytorch_lightning as pl
from dataset.retinal_text_dataset import RetinalTextDataset
import copy
import torch
import os
import pandas as pd
import random 
from evaluation.tasks.util import subclasses
import dill
from tqdm import tqdm
import spacy 
from sklearn.metrics import confusion_matrix
from clinical_capabilities.clinical_capabilities_util import add_schema

class ClosedEndedExperiment(pl.LightningModule):
    def __init__(self, config, model, task, few_shot_examples=None):
        super(ClosedEndedExperiment, self).__init__()

        self.config = config
        self.model = model
        self.task = task
        self.few_shot_examples = few_shot_examples
        self.nlp = spacy.load("en_core_web_sm")

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels[0]
        image_ids = batch[1][1]
        self.task.image_ids += image_ids

        samples = [self.task.form_question() for i in range(len(labels))]
        samples = {key: [d[key] for d in samples] for key in samples[0]}
        samples['Question'] = [add_schema(q) for q in samples['Question']]
        questions = samples['Question']
        answer_preambles = samples['Answer']

        if self.model:
            images = images.cuda()
            if self.model.is_generative:
                outputs, samples = self.model.query(images, questions, answer_preamble=answer_preambles, max_new_tokens=self.config.dataset.task.max_new_tokens_cot, output_only=True, return_samples=True)
                print(f'Sample output 0: {outputs[0]}')
                if self.task.cot:
                    cot_question_answers = [self.task.form_cot_question(out) for out in outputs]
                    questions = [add_schema(text['Question']) for text in cot_question_answers]
                    answer_preambles = [text['Answer'] for text in cot_question_answers]

                    outputs = []
                    samples = {k: [] for k in samples.keys()}
                    for image, question, answer_preamble in zip(images, questions, answer_preambles):
                        output, sample = self.model.query(image.unsqueeze(0), [question], answer_preamble=[answer_preamble], max_new_tokens=self.config.dataset.task.max_new_tokens_answer, output_only=True, return_samples=True)
                        outputs.append(output[0])
                        for key in sample.keys():
                            samples[key].append(sample[key][0])

                print(len(images))
                print(outputs)
            else:
                retrieval_statements = [self.task.form_statement(o) for o in self.task.options]
                statement_probabilities = self.model(images, retrieval_statements)
                outputs = [self.task.options[i] for i in statement_probabilities.argmax(dim=1)]

                samples['Question'] = [' / '.join(retrieval_statements) for i in range(images.shape[0])]
                samples['Input'] = samples['Question']
        else:
            outputs = [random.choice(self.task.options) if random.random() < 0.35 else label for label in labels]

        self.task.questions += samples['Question']
        self.task.inputs += samples['Input']
        self.task.outputs += outputs

        # Keyword extraction
        max_length = 4
        outputs = [o.replace('.', '') for o in outputs]
        output_tokens = [[str(word).lower() for word in self.nlp(sentence) if not word.is_punct] for sentence in outputs]
        combinations = [[' '.join(sentence[i:j]) for i in range(len(sentence)) for j in range(i+1, min(i+max_length, len(sentence))+1)] for sentence in output_tokens]
        answers = [next((opt for opt in self.task.options if ' '.join([str(word).lower() for word in self.nlp(opt.lower()) if not word.is_punct]) in c), self.task.invalid_response) for c in combinations]

        print(labels)
        print(answers)

        self.task.labels += labels
        self.task.predictions += answers

        x = 3

class ClosedEndedEvaluator():
    def __init__(self, config, dataset, specific_tasks=None):
        self.config = config
        self.dataset = dataset

        from evaluation.tasks.specialist.specialist import SpecialistTasks
        specialist_tasks = [subclass(config) for task_subset in subclasses(SpecialistTasks) for subclass in subclasses(task_subset)]
        
        # Filter tasks based on specific_tasks list from the configuration.
        specialist_tasks = [task for task in specialist_tasks if task.__class__.__qualname__.split('.')[0] in config.dataset.task.specific_tasks]
        if specific_tasks is not None:
            specialist_tasks = [task for task in specialist_tasks if task.__class__.__qualname__ in specific_tasks]
            print('Running closed ended analysis on ', specialist_tasks)
        self.tasks = specialist_tasks

    def run_task(self, model, task):
        from torch.utils.data import DataLoader
        print(f'On task {task}')

        # Load datasets
        dataset = copy.deepcopy(self.dataset)
        dataset.standard = True
        dataset.data_csv = dataset.data_csv[dataset.data_csv['TabularAnnotated'] == True]
        dataset.target = [task.variable, 'ImageId']
        dataset.target_columns = [dataset.target]
        dataset.data_csv[task.variable] = dataset.data_csv[task.variable].apply(task.process_default)

        dataset.data_csv = dataset.data_csv.sample(n=min(len(dataset.data_csv), self.config.dataset.number_train_labels), ignore_index=True, random_state=self.config.seed).reset_index()
        dataset.data_csv = dataset.filter_dataset(lambda s: ~s.isna(), task.variable).copy()
        dataset.data_csv[task.variable] = dataset.data_csv[task.variable].apply(task.form_output)
        dataset.data_csv = dataset.filter_dataset(lambda s: ~s.isna(), task.variable).copy()

        few_shot_examples=None

        data_loader = DataLoader(dataset, batch_size=self.config.model.batch_size, shuffle=False, collate_fn=RetinalTextDataset.custom_collate, 
                                persistent_workers=False, pin_memory=False, num_workers=6, drop_last=False)

        closed_ended = ClosedEndedExperiment(self.config, model, task, few_shot_examples=few_shot_examples)

        dl = iter(data_loader)
        for i, batch in enumerate(dl):
            if self.config.dataset.task.closed_ended_limit_val_batches and (i >= self.config.dataset.task.closed_ended_limit_val_batches):
                break
            closed_ended.test_step(batch, 0)

        result = confusion_matrix(task.labels, task.predictions, labels=task.options + [task.invalid_response])

        results_dict = {'task_type': task.full_name[0], 'task_name': task.full_name[1], 
                'options': task.options, 'result': result,
                'questions': task.questions, 'inputs': task.inputs, 'outputs': task.outputs,
                'labels': task.labels, 'predictions': task.predictions,
                'ImageId': [os.path.join(self.config.dataset.image_dir, id) for id in task.image_ids],
                'image_resolution': self.config.dataset.data_aug_shape[0],
                'CoT_multi': task.cot
            }       
        task.reset()
        return results_dict        
    
    def run_tasks(self, model):
        results = []
        for task in tqdm(self.tasks):
            task_results = self.run_task(model, task)
            print(task_results['result'])
            res = pd.DataFrame({k: task_results[k] for k in ['inputs', 'outputs', 'labels', 'predictions']})

            results.append(task_results)
        return results

def save_results(results_df, results_path):
    from evaluation.figure_util import make_folder_if_not_exists
    make_folder_if_not_exists(results_path)

    all_results_path = os.path.join(results_path, 'results.pkl')

    with open(all_results_path, 'wb') as f:
        dill.dump(results_df, f)

    print(f'All results DataFrame written to {all_results_path}')
    
import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def test(config):
    import sys
    from slurm.util import record_job_id
    config = record_job_id(config)
    
    sys.path.append(config['flamingo_dir'])
    sys.path.append(config['octlatent_dir'])
    from models.vlm import VLM

    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)
    
    results_path = os.path.join(config.results_path, config["job_id"])

    # Load dataset
    dataset = RetinalTextDataset(config, set_=config.dataset.task.set)

    evaluator = ClosedEndedEvaluator(config, dataset, specific_tasks=config.dataset.task.validation_tasks)
    device = torch.device('cuda:0')

    all_results = pd.DataFrame()

    if config.mock:
        print('Running mock experiments')

        model = None
        for name in ['General medical VLM (Mock 1)', 'RetinaVLM-Tabular (Mock 2)', 'RetinaVLM-Specialist (Mock 3)']:
            results = evaluator.run_tasks(model)
            
            results_df = pd.DataFrame(results)
            results_df['model'] = name

            all_results = pd.concat((all_results, results_df.copy())).reset_index(drop=True)
            save_results(all_results, results_path)
    else:
        for model_spec in config.pretrained_models:
            print(f'Running {model_spec}')

            # Load model
            config.model.checkpoint_path = model_spec
            model = VLM(config.copy()).load(device=device)

            # Run tasks and collect results
            results = evaluator.run_tasks(model)

            results_df = pd.DataFrame(results)
            results_df['model'] = model_spec[2]
            results_df['model_display_name'] = model_spec[3]

            all_results = pd.concat((all_results, results_df.copy())).reset_index(drop=True)
            save_results(all_results, results_path)

            del model
            torch.cuda.empty_cache()

    print(all_results)
    x = 3

if __name__ == "__main__":
    test()