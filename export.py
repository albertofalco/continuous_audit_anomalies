# Función para la exportación de graficos a png.
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

import sys

# Count the arguments
arguments = len(sys.argv) - 1
print ("The script is called with %i arguments" % (arguments))

def export_png(filename : 'str' = 'test', y_test = None, y_pred = None):

    # Obtención de la matriz de confusión
    cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels = ['No Fraude', 'Fraude'])

    fig = plt.figure(figsize = (10,6))
    ax = plt.axes()
    disp.plot(ax = ax, values_format = 'd', colorbar = False, cmap = 'tab20b')
    ax.set_title(label = 'Matriz de confusión', fontdict = {'fontsize' : 20})
    ax.set_xlabel(xlabel = 'Predicciones', fontdict = {'fontsize' : 14})
    ax.set_ylabel(ylabel = 'Observaciones', fontdict = {'fontsize' : 14})
    plt.savefig('.//figures//' + filename + '_confusion_matrix.png')

    # Obtención de la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_pred)
    disp = RocCurveDisplay(fpr = fpr, tpr = tpr)

    fig = plt.figure(figsize = (10,6))
    ax = plt.axes()
    disp.plot(ax = ax, c = 'blue')
    ax.set_title(label = 'Curva ROC', fontdict = {'fontsize' : 20})
    ax.set_xlabel(xlabel = 'Tasa de falsos positivos', fontdict = {'fontsize' : 14})
    ax.set_ylabel(ylabel = 'Tasa de verdaderos positivos', fontdict = {'fontsize' : 14})
    plt.savefig('.//figures//' + filename + '_roc_curve.png')

    # Obtención de la curva Recall - Precision
    prec, recall, _ = precision_recall_curve(y_true = y_test, probas_pred = y_pred)
    disp = PrecisionRecallDisplay(precision = prec, recall = recall)

    fig = plt.figure(figsize = (10,6))
    ax = plt.axes()
    disp.plot(ax = ax, c = 'blue')
    ax.set_title(label = 'Curva Precision-Recall', fontdict = {'fontsize' : 20})
    ax.set_xlabel(xlabel = 'Recall (Sensibilidad)', fontdict = {'fontsize' : 14})
    ax.set_ylabel(ylabel = 'Precision', fontdict = {'fontsize' : 14})
    plt.savefig('.//figures//' + filename + '_precision_recall.png')