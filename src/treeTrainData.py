import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Retorna as listas: AUC score, matrix de confusão de legendas de gráficos
def treeData(tree, xTest, yTest, foldNumber, aucList, confusionList, legendsList, model):

    # Tree predict
    Ypredict = tree.predict(xTest)

    # Matriz de Confusão
    confusionList.append(confusion_matrix(yTest, Ypredict))  

    # AUC ROC 
    aucAux = roc_auc_score(yTest, Ypredict)
    aucList.append(aucAux)
    legendsList.append("Fold {} - AUC: {:.4}".format(foldNumber, aucAux))
    # ROC
    fpr, tpr, _ = roc_curve(yTest, Ypredict)

    # Plot pra cada folder
    plot.plot(fpr, tpr)
    plot.title('Curva ROC de todos Folds - {}'.format(model))

    return (aucList, confusionList, legendsList)
