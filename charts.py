import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature


def prec_rec_chart(actual, predicted, filename):
    '''
    Creates a precision recall curve. Based on the sklearn documentation

    @param actual - Pandas Series of actual values
    @param predicted - Pandas Series of predicted values
    @param filename - Filename to save chart to
    '''
    precision, recall, _ = precision_recall_cuve(actual, predicted)

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, color='b', alpha=0.2, **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('2-class Precision-Recall Curve')

    plt.savefig(filename)


if __name__ == '__main__':
    df = pd.read_csv('example.csv')

