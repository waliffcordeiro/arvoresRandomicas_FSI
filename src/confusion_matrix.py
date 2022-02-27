from cv2 import reduce
from numpy import add
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plot

def confusionMatrix(confusion_datas, model):
    # Plot da matriz de confusão
    ConfusionMatrixDisplay(confusion_matrix=add.reduce(confusion_datas)).plot()
    plot.title("Matriz de Confusão - {}".format(model))
    plot.savefig("../results/{}_confusion_matrix.png".format(model))
    plot.show()