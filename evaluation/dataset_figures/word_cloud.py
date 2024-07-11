from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import hydra
import os
from evaluation.figure_util import save_fig_path_creation

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def generate_figure(config):
    from dataset.retinal_text_dataset import RetinalTextDataset
    dataset = RetinalTextDataset(config, set_='all')

    # Your text data
    questions_super_string = ' '.join(qa['Question'] for lst in dataset.data_csv['LLM_Qs_As'] for qa in lst)
    answers_super_string = ' '.join(qa['Answer'] for lst in dataset.data_csv['LLM_Qs_As'] for qa in lst)
    annotations_super_string = ' '.join(dataset.data_csv['Annotation'])

    # Add any additional stop words if needed
    additional_stopwords = {'the', 'and', 'is', 'it', 'to', 'of', 'image'}
    stopwords = set(STOPWORDS).union(additional_stopwords)
    wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=2000, height=1000, max_words=100)

    # Annotations
    wordcloud_image = wordcloud.generate(annotations_super_string)

    plt.figure()
    plt.imshow(wordcloud_image, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    save_fig_path_creation(os.path.join(config.figure_path, 'specialist_annotations_word_cloud.jpg'), dpi=500)

    # Questions
    wordcloud_image = wordcloud.generate(questions_super_string)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    save_fig_path_creation(os.path.join(config.figure_path, 'specialist_questions_word_cloud.jpg'), dpi=500)

    # Answers
    stopwords.discard('no')
    stopwords.discard('No')

    wordcloud_image = wordcloud.generate(answers_super_string)

    plt.figure()
    plt.imshow(wordcloud_image, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    save_fig_path_creation(os.path.join(config.figure_path, 'specialist_answers_word_cloud.jpg'), dpi=500)

generate_figure()