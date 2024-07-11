import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.statistics import *
from utils.statistics import bootstrap_f1_confidence_interval
from evaluation.paper_util import get_font_color
from evaluation.radar_charts import radar_factory
from evaluation.junior_specialist import junior_referral_predictions, junior_staging_predictions
sns.set(font_scale=1.3)
sns.set_style("whitegrid", {'axes.grid' : True})
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import sklearn
import math
import itertools
import hydra
import os
from evaluation.figure_util import save_fig_path_creation, make_folder_if_not_exists

def aggregate_predictions(results):
    aggregated_biomarker_results = pd.DataFrame()

    for k in ['predictions', 'labels', 'options']:
        aggregated_biomarker_results[k] = results.groupby('model_display_name')[k].apply(list).apply(lambda ls: [item for sublist in ls for item in sublist])

    return aggregated_biomarker_results

def aggregate_f1(aggregated_biomarker_results, positive_classes, n_bootstraps=2000):
    return aggregated_biomarker_results[['labels', 'predictions', 'options']].apply(lambda p: bootstrap_f1_confidence_interval(*p, positive_classes=positive_classes, n_bootstraps=n_bootstraps, ci=95), axis=1)

class ClosedEndedFigures():
    def __init__(self, config, all_results, figure_path):
        self.config = config
        self.dataset = None
        self.figure_path = figure_path
        self.all_results = all_results

        referral_task = self.all_results.loc[self.all_results['task_name'] == config.dataset.task.referral_three_levels_task]
        for i, row in referral_task.iterrows():
            exclude = [im_id.split('/')[-1] in config.dataset.task.filtered_referral_scans for im_id in row['ImageId']]

            for field in ['ImageId', 'labels', 'predictions', 'inputs', 'outputs']:
                referral_task.at[i, field] = [p for p, e in zip(row[field], exclude) if not e]

            referral_task.at[i, 'result'] = sklearn.metrics.confusion_matrix(referral_task.loc[i, 'labels'], referral_task.loc[i, 'predictions'], labels=referral_task.loc[i, 'options'] + ['Invalid response'])

        self.all_results.loc[self.all_results['task_name'] == config.dataset.task.referral_three_levels_task] = referral_task

        self.results_ixed = all_results.set_index(['CoT_multi', 'task_type', 'task_name', 'model_display_name'])

        self.colors = [self.config.dataset.task.colors[m] for m in [self.config.dataset.task.models.base_model[-1], self.config.dataset.task.models.specialist_model[-1], self.config.dataset.task.models.retinal_specialists[-1]]]
        self.cmap = mcolors.LinearSegmentedColormap.from_list("custom_palette", ['#FFFFFF'] + self.colors)

        make_folder_if_not_exists(self.figure_path)

        self.model_order = (
            self.config.dataset.task.models.baseline_models +
            self.config.dataset.task.models.base_model +
            self.config.dataset.task.models.specialist_model + 
            self.config.dataset.task.models.retinal_specialists
        )
            
    def round1(self, x):
        return f'{(1 * round(x, 3)):.1f}'
    
    def round1(self, x):
        return f'{(1 * round(x, 3)):.3f}'

    def paper_f1_statistics(self):
        # To add p-values, use the mcnemar function and pick a base model (perhaps trainee, or such)
        res = self.results_ixed.loc[True]
        
        table = pd.DataFrame()
        show_table = pd.DataFrame()

        def present_ci(r):
            if pd.isna(r):
                return ''
            return f'{self.round1(r[0])} ({self.round1(r[1][0])}, {self.round1(r[1][1])})'

        # Staging
        staging_results = res.loc['SpecialistOther'].loc[self.config.dataset.task.staging_task]
        aggregated_staging_results = aggregate_predictions(staging_results)
        table['Staging F1'] = aggregate_f1(aggregated_staging_results, positive_classes=self.config.dataset.task.staging_task_positive)
        show_table['Staging F1'] = table['Staging F1'].apply(present_ci)

        staging_significance_df = pd.DataFrame(index=aggregated_staging_results.index, columns=aggregated_staging_results.index)
        for (idx1, row1), (idx2, row2) in itertools.product(aggregated_staging_results.iterrows(), repeat=2):
            staging_significance_df.at[idx1, idx2] = mcnemars_test(row1['predictions'], row2['predictions'], row1['labels'])
        (1 * staging_significance_df).to_csv(os.path.join(self.staging_path, 'pairwise_significance.csv'))

        # Referral
        referral_results = res.loc['SpecialistReferral'].loc[self.config.dataset.task.referral_three_levels_task]
        aggregated_referral_results = aggregate_predictions(referral_results)
        table['Referral (Three Levels) F1'] =  aggregate_f1(aggregated_referral_results, positive_classes=self.config.dataset.task.referral_three_levels_positive)
        show_table['Referral (Three Levels) F1'] = table['Referral (Three Levels) F1'].apply(present_ci)

        referral_significance_df = pd.DataFrame(index=referral_results.index, columns=referral_results.index)
        for (idx1, row1), (idx2, row2) in itertools.product(referral_results.iterrows(), repeat=2):
            referral_significance_df.at[idx1, idx2] = mcnemars_test(row1['predictions'], row2['predictions'], row1['labels'])
        (1 * referral_significance_df).to_csv(os.path.join(self.referral_three_levels_path, 'pairwise_significance.csv'))

        # Biomarkers
        if 'SpecialistDetection' in self.all_results['task_type']:
            biomarker_results = res.loc['SpecialistDetection']
            aggregated_biomarker_results = aggregate_predictions(biomarker_results)
            table['Biomarker F1'] = aggregate_f1(aggregated_biomarker_results, positive_classes=['present'])
            show_table['Biomarker F1'] = table['Biomarker F1'].apply(present_ci)

        table_path = os.path.join(self.figure_path, f'paper_f1_table.csv')
        print('Saving summary table to', table_path)
        show_table.to_csv(table_path)

        latex_path = os.path.join(self.figure_path, f'paper_f1_latex_table.txt')
        latex_table = show_table.applymap(lambda x: '\\multicolumn{1}{c|}{' + str(x) + '}')
        latex_table.to_latex(latex_path)

        return table

    def biomarker_detection_per_severity(self, model, cot):
        sns.set(font_scale=2.5)
        sns.set_style("whitegrid", {'axes.grid' : True})

        if self.dataset is None:
            from dataset.retinal_text_dataset import RetinalTextDataset
            self.dataset = RetinalTextDataset(self.config.copy(), set_='all')
            self.dataset.data_csv['Drusen confluent?'] = self.dataset.data_csv['Drusen confluent?'].fillna('N')
            self.dataset.data_csv['Double layer-sign (DLS)'] = self.dataset.data_csv['Double layer-sign (DLS)'].fillna('N')

        res = self.results_ixed.loc[cot].loc['SpecialistDetection']

        task_severity_pairs = [
            ('SubretinalFluid', 'Subretinal fluid volume'),
            ('IntraretinalFluid', 'Intraretinal fluid volume'),
            ('Hypertransmission', 'Hypertrans-mission severity'),
            ('Hypertransmission', 'Hypertrans-mission type'),
            ('Drusen', 'Drusen size'),
            ('Drusen', 'Drusen number'),
            ('PED', 'PED size'),

            ('PED', 'PED type'),
            ('RPEElevation', 'RPE state'),
            ('RPEElevation', 'Double layer-sign (DLS)'),
            ('Drusen', 'Drusen confluent?'),
            ('SHRM', 'SHRM location'),
        ]

        order_list = ['One', 'Two', 'Many', 'Small', 'Mild', 'N', 'iRORA', 'Moderate', 'Medium', 'Y', 'cRORA', 'Large', 'Significant', 'Drusenoid', 'Serous', 'Fibrovascular', 'Atrophy', 'Degeneration', 'Temporal', 'Fovea', 'Nasal', 'All']

        for task, severity in task_severity_pairs:
            fig = plt.figure(figsize=(10, 10))
            biomarker_results = res.loc[task].loc[model, ['ImageId', 'predictions', 'labels']]
            biomarker_results = biomarker_results.apply(pd.Series).T
            biomarker_results['ImageId'] = biomarker_results['ImageId'].apply(lambda i: i.split('/')[-1])

            result_df = pd.merge(biomarker_results, self.dataset.data_csv[['ImageId', severity]], on='ImageId', how='left')

            results_by_severity = result_df[['predictions', 'labels', severity]]
            results_by_severity = results_by_severity.loc[results_by_severity[severity].notna()]

            results_by_severity['Sensitivity'] = results_by_severity['predictions'] == results_by_severity['labels']
            results_by_severity[severity] = results_by_severity[severity].astype('category')

            accuracy_by_severity = results_by_severity.groupby(severity)['Sensitivity'].mean().reset_index()
            accuracy_by_severity['Sensitivity'] *= 1
            accuracy_by_severity = accuracy_by_severity.rename(columns={severity: 'Severity'})

            sns.barplot(x='Severity', y='Sensitivity', data=accuracy_by_severity, order=[o for o in order_list if o in accuracy_by_severity['Severity'].tolist()], palette=self.colors[-len(accuracy_by_severity):])
            plt.ylabel('')
            plt.xticks(fontsize=36)
            plt.yticks(fontsize=36)
            plt.xlabel('')
            plt.ylim(0, 1)

            save_fig_path_creation(os.path.join(self.biomarker_path, 'by_severity', f'{task}-{severity}.png'), pad_inches=0.1)

        x = 3

    def model_comparison_barplots(self, f1_table, experiment):
        sns.set(font_scale=2)
        sns.set_style("whitegrid", {'axes.grid' : True})

        df = f1_table

        colors_dict = self.config.dataset.task.colors

        # Preparing barplot_df
        df = df[experiment].dropna()
        barplot_df = pd.DataFrame()

        barplot_df['F1'] =      df.apply(lambda x: x[0])
        barplot_df['lower'] =   df.apply(lambda x: x[1][0])
        barplot_df['upper'] =   df.apply(lambda x: x[1][1])
        barplot_df['model_display_name'] = df.index

        # Filtering and reordering based on YAML configuration
        barplot_df = barplot_df[barplot_df['model_display_name'].isin(self.model_order)]
        model_order = [model for model in self.model_order if model in barplot_df['model_display_name']]
        barplot_df = barplot_df.set_index('model_display_name').loc[model_order].reset_index()

        # Calculating error bars
        barplot_df['yerr_lower'] = barplot_df['F1'] - barplot_df['lower']
        barplot_df['yerr_upper'] = barplot_df['upper'] - barplot_df['F1']

        # Assigning colors based on model_display_name
        barplot_df['model_display_name'] = barplot_df['model_display_name'].map(self.config.dataset.task.display_names)

        # Plotting
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='model_display_name', y='F1', data=barplot_df, hue='model_display_name', dodge=False, palette=dict(colors_dict))
        ax.set_ylim(-0.02, min((barplot_df['F1'] + barplot_df['yerr_upper']).max() + 0.02, 1))

        plt.errorbar(x=barplot_df['model_display_name'], y=barplot_df['F1'], 
                    yerr=[barplot_df['yerr_lower'], barplot_df['yerr_upper']], 
                    fmt='none', c='black', capsize=5)
        
        plt.xlabel('Image-based clinical decision maker')

        plt.legend([], [], frameon=False)
        return ax

    def radar(self, data):
        plot_data = data.map(lambda x: max(0.01, x))
        plot_data = plot_data.reindex(self.model_order)
        plot_data = plot_data.loc[~plot_data.isna().all(1)]

        plot_data.index = plot_data.index.map(self.config.dataset.task.display_names)

        N = len(plot_data.columns)
        theta = radar_factory(N, frame='polygon')

        # Adding an extra point to theta to close the radar chart loop
        theta = np.append(theta, theta[0])

        colors = self.config.dataset.task.colors

        # Create a single subplot for the radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))

        # Plot each row of the DataFrame with a color from the color map
        plot_data = plot_data.reindex(colors.keys())
        plot_data = plot_data.loc[~plot_data.isna().all(1)]
        markers = ['v', 'X', '^', 'D', 'o', 's', '<', '>', 'p', 'h', 'H', '*', '+', 'x', 'X', '|', '_']
        for (index, d), marker in zip(plot_data.reindex(colors.keys()).iterrows(), markers):
            color = colors[index]
            d = d.tolist() + d.tolist()[:1]  # Close the radar chart by repeating the first value at the end
            ax.plot(theta, d, label=index, color=color, lw=4, marker=marker, markersize=12, markerfacecolor=color, markeredgecolor='#526B7A', markeredgewidth=1.5)
            ax.fill(theta, d, alpha=0.25, color=color)

        # Set the variable labels (spoke labels)
        spoke_labels = plot_data.columns.tolist()
        ax.set_varlabels(spoke_labels)

        max_value = plot_data.values.max()
        max_value = 0.01 * (5 * math.ceil(100 * max_value / 5))
        print('Max value radar', max_value)
        ax.set_rgrids([max_value * i / 5 for i in range(1, 6)], labels=[], fontsize=24)

        # Set the color of the outer ring (spine) to dark gray
        radar_colour = '#A0A0A0'
        ax.spines['polar'].set_color(radar_colour)
        ax.spines['polar'].set_linewidth(1)

        # Set the color and linewidth of the inner gridlines (circular lines)
        for gridline in ax.yaxis.get_gridlines():
            gridline.set_color(radar_colour)
            gridline.set_linewidth(2)  # Optionally set the linewidth

        for label in ax.get_xticklabels():
            label.set_zorder(1)

        return ax

    def closed_ended_figures(self):
        sns.set(font_scale=3)
        sns.set_style("whitegrid", {'axes.grid' : True})
        self.clinical_evaluation_figures()

        cot = True
        self.cot_path = os.path.join(self.figure_path, f'CoT={cot}')
        self.staging_path = os.path.join(self.cot_path, 'staging')
        self.referral_three_levels_path = os.path.join(self.cot_path, 'referral_three_levels')
        self.biomarker_path = os.path.join(self.cot_path, 'biomarkers')

        # Junior specialist preditions
        for referral_levels, task in [(3, self.config.dataset.task.referral_three_levels_task)]:
            jrp_referral, jrp_combined = junior_referral_predictions(self.config, levels=referral_levels)
            for junior_clinician, performance in jrp_combined.items():
                performance['options'] = self.results_ixed.loc[True].loc['SpecialistReferral'].loc[task].iloc[0]['options']
                performance['result'] = sklearn.metrics.confusion_matrix(performance['labels'], performance['predictions'], labels=performance['options'] + ['Invalid response'])

            self.results_ixed.loc[(True, 'SpecialistReferral', task, junior_clinician)] = performance

        for referral_levels, task in [(3, self.config.dataset.task.referral_three_levels_task)]:
            optician = 'Opticians'
            performance = {
                'predictions': ['next two weeks'] * len(self.results_ixed.loc[True].loc['SpecialistReferral'].loc[task].iloc[0]['predictions']),
                'labels': self.results_ixed.loc[True].loc['SpecialistReferral'].loc[task].iloc[0]['labels'],
                'options': self.results_ixed.loc[True].loc['SpecialistReferral'].loc[task].iloc[0]['options']
            }
            performance['result'] = sklearn.metrics.confusion_matrix(performance['labels'], performance['predictions'], labels=performance['options'] + ['Invalid response'])

            self.results_ixed.loc[(True, 'SpecialistReferral', task, optician)] = performance


        jrp_staging, jrp_combined = junior_staging_predictions()
        for junior_clinician, performance in jrp_combined.items():
            performance['options'] = self.results_ixed.loc[True].loc['SpecialistOther'].iloc[0]['options']
            performance['result'] = sklearn.metrics.confusion_matrix(performance['labels'], performance['predictions'], labels=performance['options'] + ['Invalid response'])

            self.results_ixed.loc[(True, 'SpecialistOther', self.config.dataset.task.staging_task, junior_clinician)] = performance

        # Disease staging - extended data figures
        result = self.results_ixed.loc[True].loc['SpecialistOther'].loc[self.config.dataset.task.staging_task].loc['Specialist-RetinaVLM (V4.52 192)']
        comparison_df = pd.DataFrame({k: result[k] for k in ['labels', 'predictions', 'inputs', 'outputs']})
        comparison_df['ImageId'] = result['ImageId'].apply(lambda l: l.split('/')[-1])

        result = self.results_ixed.loc[True].loc['SpecialistOther'].loc[self.config.dataset.task.staging_task].loc['Junior ophthalmologists']
        junior_df = pd.DataFrame({k: result[k] for k in ['labels', 'predictions', 'ImageId']})
        junior_df['Junior prediction'] = junior_df['predictions']
        comparison_df = pd.merge(comparison_df, junior_df[['ImageId', 'Junior prediction']], on='ImageId')
        comparison_df['ImageId'] = self.config.images_for_figures_dir + '/' + comparison_df['ImageId']

        # Patient referral - extended data figures
        result = self.results_ixed.loc[True].loc['SpecialistReferral'].loc[self.config.dataset.task.referral_three_levels_task].loc['Specialist-RetinaVLM (V4.52 192)']
        comparison_df = pd.DataFrame({k: result[k] for k in ['labels', 'predictions', 'inputs', 'outputs']})
        comparison_df['ImageId'] = result['ImageId'].apply(lambda l: l.split('/')[-1])

        result = self.results_ixed.loc[True].loc['SpecialistReferral'].loc[self.config.dataset.task.referral_three_levels_task].loc['Junior ophthalmologists']
        junior_df = pd.DataFrame({k: result[k] for k in ['labels', 'predictions', 'ImageId']})
        junior_df['Junior prediction'] = junior_df['predictions']
        comparison_df = pd.merge(comparison_df, junior_df[['ImageId', 'Junior prediction']], on='ImageId')
        comparison_df['ImageId'] = self.config.paths.retina_referral_median_image_dir + '/' + comparison_df['ImageId']

        # Main figures
        staging_results = self.results_ixed.loc[(True, 'SpecialistOther')]
        staging_f1 = staging_results[['labels', 'predictions', 'options']].apply(lambda p: bootstrap_f1_confidence_interval(*p, positive_classes=self.config.dataset.task.staging_task_positive, n_bootstraps=0, ci=95), axis=1)
        staging_f1 = staging_f1.apply(lambda r: 1 * r)
        staging_f1 = staging_f1.unstack('task_name')
        print(staging_f1)

        referral_three_level_results = self.results_ixed.loc[(True, 'SpecialistReferral')]
        referral_three_levels_f1 = referral_three_level_results[['labels', 'predictions', 'options']].apply(lambda p: bootstrap_f1_confidence_interval(*p, positive_classes=self.config.dataset.task.referral_three_levels_positive, n_bootstraps=0, ci=95), axis=1)
        referral_three_levels_f1 = referral_three_levels_f1.apply(lambda r: 1 * r)
        referral_three_levels_f1 = referral_three_levels_f1.unstack('task_name')
        print(referral_three_levels_f1)

        # Calculate aggregate f1 scores
        f1_table = self.paper_f1_statistics()

        ax = self.model_comparison_barplots(f1_table, 'Staging F1')
        plt.ylabel('Disease staging F1 Score')

        # Individual scores of junior ophthalmologists
        junior_ophthalmologists_scores = []
        for junior in jrp_staging.values():
            junior_f1 = 1 * bootstrap_f1_confidence_interval(junior['labels'], junior['predictions'], self.config.dataset.task.staging_task_positive, positive_classes=self.config.dataset.task.staging_task_positive, n_bootstraps=0, ci=95)
            junior_ophthalmologists_scores.append(junior_f1) # Replace with actual values
            
        junior_df = pd.DataFrame({
            'Image-based clinical decision maker': ['Junior ophthalmologists'] * len(junior_ophthalmologists_scores),
            'Disease staging F1 score': junior_ophthalmologists_scores
        })

        # sns.stripplot(x='Image-based clinical decision maker', y='Disease staging F1 score', data=junior_df, ax=ax, 
        #       jitter=0.5, size=6, dodge=True, edgecolor='#CDD3CE', linewidth=1, facecolors='#2F3061')

        plt.xticks([])
        ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], max(junior_ophthalmologists_scores)) + 0.02)

        save_fig_path_creation(os.path.join(self.staging_path, f'Model_comparison_barplot.png'))

        # Referral results
        ax = self.model_comparison_barplots(f1_table, 'Referral (Three Levels) F1')

        # Individual scores of junior ophthalmologists
        junior_ophthalmologists_scores = []
        for junior in jrp_referral.values():
            junior_f1 = 1 * bootstrap_f1_confidence_interval(junior['labels'], junior['predictions'], ['not', 'next year', 'next two weeks'], positive_classes=self.config.dataset.task.referral_three_levels_positive, n_bootstraps=0, ci=95)
            junior_ophthalmologists_scores.append(junior_f1) # Replace with actual values
            
        junior_df = pd.DataFrame({
            'Image-based clinical decision maker': ['Junior ophthalmologists'] * len(junior_ophthalmologists_scores),
            'Patient referral F1 Score': junior_ophthalmologists_scores
        })
        # sns.stripplot(x='Image-based clinical decision maker', y='Patient referral F1 Score', data=junior_df, ax=ax, 
        #       jitter=0.5, size=6, dodge=True, edgecolor='#CDD3CE', linewidth=1, facecolors='#2F3061')

        plt.xticks([])
        plt.ylabel('Patient referral F1 Score')
        save_fig_path_creation(os.path.join(self.referral_three_levels_path, f'Model_comparison_barplot.png'), transparent=True)

        # Heatmap plots
        sns.set(font_scale=2)
        sns.set_style("whitegrid", {'axes.grid' : True})
        models = [item for sublist in self.config.dataset.task.models.values() for item in sublist]

        # Compare tasks
        if self.all_results['task_type'].apply(lambda x: x == 'SpecialistDetection').any():
            biomarker_results = self.results_ixed.loc[(True, 'SpecialistDetection')]
            biomarker_f1 = biomarker_results[['labels', 'predictions', 'options']].apply(lambda p: bootstrap_f1_confidence_interval(*p, positive_classes=['present'], n_bootstraps=0, ci=95), axis=1)
            biomarker_f1 = biomarker_f1.apply(lambda r: 1 * r)
            biomarker_f1 = biomarker_f1.unstack('task_name')
            # biomarker_f1.round(1).to_csv(os.path.join(self.biomarker_path, 'table.csv'))
            print(biomarker_f1)

            # Boimarker charts
            ax = self.radar(biomarker_f1)
            ax.set_varlabels([''] * len(biomarker_f1.columns))
            save_fig_path_creation(os.path.join(self.biomarker_path, f'radar_plot_no_labels.png'))

            self.radar(biomarker_f1)
            save_fig_path_creation(os.path.join(self.biomarker_path, f'radar_plot.png'))

            self.biomarker_detection_per_severity(self.config.dataset.task.models['specialist_model'][-1], cot)

            self.model_comparison_barplots(f1_table, 'Biomarker F1')
            plt.xticks([])

            plt.ylabel('Biomarker detection F1 Score')
            save_fig_path_creation(os.path.join(self.biomarker_path, f'Model_comparison_barplot.png'))

        for model in models:
            staging_res = self.results_ixed.loc[cot].loc['SpecialistOther'].loc[self.config.dataset.task.staging_task]
            if model in staging_res.index:
                res = staging_res.loc[model]
                plt.figure(figsize=(10, 8.6))
                sns.heatmap(np.array(res['result'])[:-1], annot=True, annot_kws={"size": 30}, cmap=self.cmap, cbar=False, vmax=22)
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])

                # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
                save_fig_path_creation(os.path.join(self.staging_path, f'{model} - AMDStage_heatmap.png'))

            referral_res = self.results_ixed.loc[cot].loc['SpecialistReferral'].loc[self.config.dataset.task.referral_three_levels_task]
            if model in referral_res.index:
                res = referral_res.loc[model]
                options = res['options'] + ['Invalid response']
                label_order = [self.config.dataset.task.referral_order.index(o) for o in options]
                label_order = [sorted(label_order).index(x) for x in label_order]
                confusion_matrix = np.array(res['result'])[label_order, :][:, label_order][:-1]

                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix, annot=True, cmap=self.cmap, cbar=False, annot_kws={"size": 35}, vmax=22)
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                save_fig_path_creation(os.path.join(self.referral_three_levels_path, f'{model} - Referral_heatmap.png'))

        x = 3
    
    def clinical_evaluation_figures(self):
        sns.set(font_scale=2)
        sns.set_style("whitegrid", {'axes.grid' : True})
    
        cot = True
        self.cot_path = os.path.join(self.figure_path, f'CoT={cot}')
        self.clinical_evaluation_path = os.path.join(self.cot_path, 'clinical_evaluation')

        clinical_evaluation_study_results = pd.DataFrame()

        for clinician, study_df, results in self.config.paths.senior_ophthalmologist_evaluations:
            df = pd.read_csv(study_df)
            senior_annotations_1 = pd.read_excel(results)
            senior_annotations_1 = senior_annotations_1.iloc[1:,:6]
            senior_annotations_1['Clinician'] = clinician
            senior_annotations_1['Report #'] = senior_annotations_1['Report #'].astype(int)
            clinical_evaluation_study_results = pd.concat((clinical_evaluation_study_results, pd.merge(df, senior_annotations_1, left_on='Excel_Id', right_on='Report #')))

        clinical_evaluation_study_results['Model'] = clinical_evaluation_study_results['Model'].map({
            'RetinaVLM-MiniGPT4/astral-microwave-524': 'Specialist-RetinaVLM (V4.52 192)',
            'Retinal specialist': 'Junior ophthalmologists',
            'LLaVA-Med': 'Medical Foundation VLM (LLaVA-Med)'
        })
        clinical_evaluation_study_results['Model'] = clinical_evaluation_study_results['Model'].map(self.config.dataset.task.display_names)
        clinical_evaluation_study_results = clinical_evaluation_study_results.replace('Strongly disagree ', 'Strongly disagree')

        colors = ["#686963", "#CDD3CE", "#B1DBFC", "#23A8F8", "#2F3061"]
        labels = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]

        for metric in ['Correctness', 'Completeness', 'Conciseness']:
            res = clinical_evaluation_study_results[['Model', metric]]
            counts = res.groupby('Model').value_counts()

            valid_models = [self.config.dataset.task.display_names[model] for model in self.model_order if self.config.dataset.task.display_names[model] in counts.index]
            reversed_order = list(reversed(valid_models))
            counts = counts.reset_index()

            counts = counts.pivot(index='Model', columns=metric, values='count').fillna(0)
            counts = counts.reindex(columns=labels, fill_value=0)
            counts = counts.loc[reversed_order]
            print(counts)

            fig, ax = plt.subplots(figsize=(6, 2.3))
            bar_container = counts.plot.barh(ax=ax, stacked=True, legend=False, color=colors)
            plt.yticks([])
            plt.xticks(fontsize=14)
            plt.ylabel('')

            # Add labels to each segment manually
            for bar in bar_container.patches:
                width = bar.get_width()
                if width > 0:  # Only label bars with positive width
                    bar_color = bar.get_facecolor()
                    font_color = get_font_color(bar_color)
                    plt.text(
                        bar.get_x() + width / 2,
                        bar.get_y() + bar.get_height() / 2,
                        int(width),
                        ha='center',
                        va='center',
                        fontsize=10,
                        color=font_color,
                    )

            save_fig_path_creation(os.path.join(self.clinical_evaluation_path, f'{metric}.png'))

        fig, ax = plt.subplots()
        [ax.scatter([], [], color=colors[i], label=labels[i], s=1, marker='s') for i in range(len(colors))]
        ax.legend(loc='center', ncol=len(labels))
        ax.axis('off')
        save_fig_path_creation(os.path.join(self.clinical_evaluation_path, f'legend.png'))

        # Per sample
        df = clinical_evaluation_study_results.reset_index(drop=True)
        df.to_csv(os.path.join(self.clinical_evaluation_path, f'full_results.csv'))

        x = 3

    def manual_save_load_predictions(self, task_type, task_name, model):
        self.manual_predictions_path = f'{self.config.manual_predictions_path}/{task_type}/{task_name}/{model}'
        if not os.path.exists(self.manual_predictions_path):
            os.makedirs(self.manual_predictions_path)
            
        self.results_ixed.loc[True].loc[task_type].loc[task_name].loc[model]

        specialist_staging = self.results_ixed.loc[True].loc[task_type].loc[task_name].loc[self.config.dataset.task.models.specialist_model[-1]]

        comparison_df = pd.DataFrame({k: specialist_staging[k] for k in ['ImageId', 'labels', 'predictions', 'inputs', 'outputs']})
        comparison_df['inputs'] = comparison_df['inputs'].apply(lambda x: x.split('assistant<|end_header_id|>\n\n')[-1])

        manual_labels_path = os.path.join(self.manual_predictions_path, 'prediction_extraction.csv')
        if not os.path.exists(manual_labels_path):
            comparison_df.loc[(comparison_df['predictions'] != comparison_df['labels'])].to_csv(manual_labels_path)
        else:
            print('Avoid overwrite', manual_labels_path)

        manual_predictions_extracted_path = f'{self.manual_predictions_path}/prediction_extraction_manual.csv'

        if os.path.exists(manual_predictions_extracted_path):
            print('Loaded manual predictions', manual_predictions_extracted_path)
            print('Before manual extraction')
            print(self.results_ixed.at[(True, task_type, task_name, model), 'result'])

            manual_label_extractions = pd.read_csv(f'{self.manual_predictions_path}/prediction_extraction_manual.csv')
            manual_label_extractions = manual_label_extractions.loc[~manual_label_extractions['actual_prediction'].isna()][['ImageId', 'actual_prediction']]

            correct_df = comparison_df.merge(manual_label_extractions, on='ImageId', how='left')
            correct_df['predictions'] = correct_df['actual_prediction'].combine_first(correct_df['predictions'])
            correct_df = correct_df.drop(columns=['actual_prediction'])

            for k in ['ImageId', 'labels', 'predictions', 'inputs', 'outputs']:
                self.results_ixed.at[(True, task_type, task_name, model), k] = correct_df[k]
            self.results_ixed.at[(True, task_type, task_name, model), 'result'] = confusion_matrix(correct_df['labels'], correct_df['predictions'], labels=specialist_staging['options'] + ["Invalid response"])
            print('After manual extraction')
            print(self.results_ixed.at[(True, task_type, task_name, model), 'result'])

        else:
            print('Manual predictions not yet provided, check at', manual_predictions_extracted_path)

def collect_results(config):
    results_files = {root: pd.read_pickle(os.path.join(root, file)) for path in config.dataset.task.finished_closed_ended_results for root, _, files in os.walk(path) for file in files if file.endswith('.pkl')}
    all_results = pd.concat(results_files.values(), ignore_index=True)
    return all_results

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def generate_figures(config):
    figure_path = os.path.join(config.figure_path, 'closed_ended', "-".join(p.strip('/').split('/')[-1] for p in config.dataset.task.finished_closed_ended_results))
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # Collect all results
    all_results = collect_results(config)
    figures = ClosedEndedFigures(config, all_results, figure_path)

    # Manual prediction extraction where first keyword algorithm failed
    figures.manual_save_load_predictions('SpecialistOther', 'AMDStage', 'Medical Foundation VLM (MedFlamingo)')
    figures.manual_save_load_predictions('SpecialistOther', 'AMDStage', 'Medical Foundation VLM (LLaVA-Med)')
    figures.manual_save_load_predictions('SpecialistOther', 'AMDStage', 'Trainee-RetinaVLM (p=0 s=0 192)')
    figures.manual_save_load_predictions('SpecialistOther', 'AMDStage', 'Specialist-RetinaVLM (V4.52 192)')
    
    figures.manual_save_load_predictions('SpecialistReferral', 'SouthamptonReferral', 'Medical Foundation VLM (MedFlamingo)')
    figures.manual_save_load_predictions('SpecialistReferral', 'SouthamptonReferral', 'Medical Foundation VLM (LLaVA-Med)')
    figures.manual_save_load_predictions('SpecialistReferral', 'SouthamptonReferral', 'Trainee-RetinaVLM (p=0 s=0 192)')
    figures.manual_save_load_predictions('SpecialistReferral', 'SouthamptonReferral', 'Specialist-RetinaVLM (V4.52 192)')

    figures.closed_ended_figures()
    
if __name__ == "__main__":
    generate_figures()