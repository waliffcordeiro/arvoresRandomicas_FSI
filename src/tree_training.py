from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold as KFold
import matplotlib.pyplot as plot

from import_data import getData
from treeTrainData import treeData
from confusion_matrix import confusionMatrix


def treeTraining(tree, x, y, model):

    # Inicializando lists
    aucList = []
    confusionList = []
    legendsList = []

    # Plot settings
    fig = plot.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)

    # O KFold separa uma unidade do split para teste, no caso, 10% do conjunto
    KFolds = KFold(n_splits=10)

    numFolds = 0
    for idxTrain, idxTest in KFolds.split(x, y):
        # Contando os folds para plotar os gráficos
        numFolds += 1
        
        # Separando dados de treino e teste
        trainX, trainY, testX, testY = x[idxTrain], y[idxTrain], x[idxTest], y[idxTest]

        # Treinamento
        tree.fit(trainX, trainY)

        # Obtendo dados para cada fold e adicionando nas lists
        aucList, confusionList, legendsList = treeData(tree, testX, testY, numFolds, aucList, confusionList, legendsList, model)
    plot.legend(legendsList)
    plot.savefig("../results/{}.png".format(model))
    plot.show()

    confusionMatrix(confusionList, model)

    return 

if __name__ == '__main__':
    dataset = getData()

    # Eliminando a coluna que queremos 'prever'
    x = dataset.drop("chd", axis=1).values # Axis = 1 para eliminar coluna e não só a linha
    y = dataset['chd'].values

    # Example
    tree = DecisionTreeClassifier() # Modelo CART
    treeTraining(tree, x, y, "CART")