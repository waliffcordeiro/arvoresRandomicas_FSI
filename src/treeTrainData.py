import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Retorna as listas: AUC score, matrix de confusão de legendas de gráficos
def treeTraining(tree, xTest, yTest, foldNumber, aucList, confusionList, legendsList, featurePerformance, model):

    # Tree predict
    # Retorna probabilidades calibradas de classificação de acordo com cada classe em uma matriz de vetores de teste.
    Ypredict = tree.predict(xTest)

    # Retorna a precisão média nos dados e rótulos de teste fornecidos.
    Yproba = tree.predict_proba(xTest)[:, 1]

    # Matriz de Confusão
    confusionList.append(confusion_matrix(yTest, Ypredict))

    # AUC ROC 
    aucAux = roc_auc_score(yTest, Yproba)
    aucList.append(aucAux)
    legendsList.append("Fold {} - AUC: {:.4}".format(foldNumber, aucAux))
    # ROC
    fpr, tpr, _ = roc_curve(yTest, Yproba)

    # Salvando a performance
    featurePerformance = [(featA + featB)/2 for featA, featB in zip(featurePerformance, tree.feature_importances_)]

    # Plot pra cada folder
    plot.plot(fpr, tpr)
    plot.title('Curva ROC de todos Folds - {}'.format(model))

    return (aucList, confusionList, legendsList, featurePerformance)
