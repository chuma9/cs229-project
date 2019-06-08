import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_confusion_matrix(y_actual, y_pred, title='None', cmap=plt.cm.gray_r):
    df_confusion = pd.crosstab(y_actual, y_pred)
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()