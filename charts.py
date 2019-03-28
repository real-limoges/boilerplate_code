import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve, auc
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


def roc_chart(actual, predicted, filename):
    '''
    Creates ROC chart by class
    '''
    fpr, tpr, _ = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)

    lw = 2
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC')
    plt.legend(loc='lower right')

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.savefig(filename)


def feature_importance_chart(df, clf, num_features, filename):
    '''
    Chart of the feature importances
    '''
    importances = clf.feature_importances_
    stds = np.std([tree.feature_importance_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    plt.title('Feature Importances')
    plt.bar(range(num_features), 
            (importances[indices])[:num_features],
            color='r',
            yerr=(std[indices])[:num_features],
            align='center')
    plt.xticks(range(num_features),
               df.columns[indices][:num_features])
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=90)
    plt.xlim([-1, num_features])
    plt.tight_layout

    plt.savefig(filename)

if __name__ == '__main__':
    df = pd.read_csv('example.csv')

