import os
import matplotlib.pyplot as plt

def make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_fig_path_creation(path, dpi=144, transparent=False, close_fig=True, pad_inches=0):
    make_folder_if_not_exists(os.path.dirname(path))
    plt.savefig(path, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi, transparent=transparent)
    print('Figure saved to', path)
    if close_fig:
        plt.close('all')