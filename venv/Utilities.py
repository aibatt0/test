import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import glob
from typing import List


class DrawPlots:
    def __init__(self, json):
        self.json = json

    def draw_plots(self) -> List[str]:
        # checking if folder not exist create one
        if not os.path.isdir('plots'):
            os.mkdir('plots')

        df = pd.read_json(self.json)

        def plot_columns(x: str, y: str):
            plt.scatter(df[x], df[y])
            plt.title(f'Comparing {x} and {y}')
            plt.xlabel(x.capitalize())
            plt.ylabel(y.capitalize())
            plt.savefig(f'plots\\{plt.gca().get_title().replace(" ", "_")}.jpg')
            plt.show()

        plot_columns('ceiling_max', 'floor_max')
        plot_columns('ceiling_min', 'floor_min')
        plot_columns('ceiling_mean', 'floor_mean')

        plot_columns('max', 'floor_max')
        plot_columns('min', 'floor_min')
        plot_columns('mean', 'floor_mean')

        plot_columns('max', 'ceiling_max')
        plot_columns('min', 'ceiling_min')
        plot_columns('mean', 'ceiling_mean')

        # Plotting confusion matrix for results of model prediction
        df_cm = pd.DataFrame(
            confusion_matrix(df['gt_corners'], df['rb_corners']),
            index=["4 corners", "6 corners", "8 corners", "10 corners"],
            columns=["4 corners", "6 corners", "8 corners", "10 corners"])

        ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')

        plt.title('Confusion matrix', fontsize=20)
        plt.xlabel('Predicted by model', fontsize=15)
        plt.ylabel('Ground truth', fontsize=15)
        plt.savefig(f'plots\\{ax.get_title().replace(" ", "_")}.jpg')
        plt.show()

        f, ax = plt.subplots(figsize=(13, 13))
        corr = df.corr()
        corr_fig = sns.heatmap(
            corr,
            mask=np.zeros_like(corr, dtype=np.bool),
            cmap='Blues',
            square=True,
            ax=ax,
            annot=True,
            fmt='.3f').set_title('Correlation between columns')
        fig = corr_fig.get_figure()
        fig.savefig(f'plots\\{ax.get_title().replace(" ", "_")}.jpg')
        plt.show()

        f1_score(df['gt_corners'], df['rb_corners'], average='weighted')
        # >>> 1.0

        # F1 score is one of the best metrics to measure model performance 
        # in multiclass classification problem.
        # Score 1.0 show that our model predicted all of the values correct.

        return [path for path in glob.glob("plots\*")]